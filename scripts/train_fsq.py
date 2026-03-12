"""Train the FSQ-VAE on preprocessed Geometry Dash frames.

Uses frame pairs from episodes for GRWM temporal slowness loss.

Usage:
    python scripts/train_fsq.py
    python scripts/train_fsq.py --epochs 400 --alpha-slow 0.1
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepdash.fsq import FSQVAE, fsqvae_loss, grwm_slowness, grwm_uniformity


class FramePairDataset(Dataset):
    """Loads consecutive frame pairs from episodes for GRWM temporal slowness.

    Each sample is (frame_t, frame_t+1). Split by episode, not by frame.
    """

    def __init__(self, episode_dirs, augment=False, device=None):
        self.pairs = []
        for ep_dir in episode_dirs:
            frames = np.load(ep_dir / "frames.npy")  # (T, 64, 64) uint8
            for i in range(len(frames) - 1):
                self.pairs.append((frames[i], frames[i + 1]))

        # Preload as tensors
        t_data = np.array([p[0] for p in self.pairs])
        t1_data = np.array([p[1] for p in self.pairs])
        self.frames_t = torch.from_numpy(t_data).float().unsqueeze(1) / 255.0
        self.frames_t1 = torch.from_numpy(t1_data).float().unsqueeze(1) / 255.0
        self.pairs = None  # free numpy data

        if device and device.type == "cuda":
            self.frames_t = self.frames_t.to(device)
            self.frames_t1 = self.frames_t1.to(device)

        self.augment = augment

    def __len__(self):
        return len(self.frames_t)

    def __getitem__(self, idx):
        ft = self.frames_t[idx]
        ft1 = self.frames_t1[idx]
        if self.augment:
            # Same random shift for both frames
            pad = 4
            i, j = torch.randint(0, 2 * pad + 1, (2,)).tolist()
            ft = transforms.functional.pad(ft, pad, padding_mode='edge')
            ft = transforms.functional.crop(ft, i, j, 64, 64)
            ft1 = transforms.functional.pad(ft1, pad, padding_mode='edge')
            ft1 = transforms.functional.crop(ft1, i, j, 64, 64)
        return ft, ft1


def train_epoch(model, loader, optimizer, alpha_slow, alpha_uniform,
                scaler=None, amp_dtype=None):
    model.train()
    total_recon, total_slow, total_uniform, n = 0, 0, 0, 0
    for ft, ft1 in loader:
        with torch.amp.autocast("cuda", enabled=amp_dtype is not None, dtype=amp_dtype):
            recon_t, z_e_t, _ = model(ft)
            recon_t1, z_e_t1, _ = model(ft1)

            recon_loss = (fsqvae_loss(recon_t, ft) + fsqvae_loss(recon_t1, ft1)) / 2
            slow_loss = grwm_slowness(z_e_t, z_e_t1)
            uniform_loss = grwm_uniformity(z_e_t)

            loss = recon_loss + alpha_slow * slow_loss + alpha_uniform * uniform_loss

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = ft.size(0)
        total_recon += recon_loss.item() * bs
        total_slow += slow_loss.item() * bs
        total_uniform += uniform_loss.item() * bs
        n += bs
    return total_recon / n, total_slow / n, total_uniform / n


@torch.no_grad()
def val_epoch(model, loader, amp_dtype=None):
    model.eval()
    total_recon, n = 0, 0
    for ft, ft1 in loader:
        with torch.amp.autocast("cuda", enabled=amp_dtype is not None, dtype=amp_dtype):
            recon_t, _, _ = model(ft)
            recon_loss = fsqvae_loss(recon_t, ft)
        bs = ft.size(0)
        total_recon += recon_loss.item() * bs
        n += bs
    return total_recon / n


def main():
    parser = argparse.ArgumentParser(description="Train FSQ-VAE on Geometry Dash frames")
    parser.add_argument("--episodes-dir", default="data/episodes")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=4e-3)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--levels", type=int, nargs="+", default=[8, 5, 5, 5],
                        help="FSQ quantization levels per channel")
    parser.add_argument("--alpha-slow", type=float, default=0.1,
                        help="Weight for GRWM temporal slowness loss")
    parser.add_argument("--alpha-uniform", type=float, default=0.01,
                        help="Weight for GRWM uniformity loss")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Fraction of episodes for validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true",
                        help="Enable bf16 automatic mixed precision (A100+)")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile for faster training")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Split episodes into train/val
    episodes_dir = Path(args.episodes_dir)
    all_episodes = sorted(ep for ep in episodes_dir.glob("*")
                          if (ep / "frames.npy").exists())
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(all_episodes))
    val_count = max(1, int(len(all_episodes) * args.val_ratio))
    val_eps = [all_episodes[i] for i in indices[:val_count]]
    train_eps = [all_episodes[i] for i in indices[val_count:]]

    print(f"Episodes: {len(all_episodes)} total, {len(train_eps)} train, {len(val_eps)} val")

    train_dataset = FramePairDataset(train_eps, augment=True, device=device)
    val_dataset = FramePairDataset(val_eps, augment=False, device=device)
    print(f"Frame pairs: {len(train_dataset)} train, {len(val_dataset)} val")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=False)

    model = FSQVAE(levels=args.levels).to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device, weights_only=True))
        print(f"Resumed from {args.resume}")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    print(f"FSQ levels: {args.levels} -> {model.codebook_size} implicit codes")

    if args.compile:
        try:
            model = torch.compile(model)
            print("torch.compile enabled")
        except Exception as e:
            print(f"torch.compile failed ({e}), continuing without it")

    use_amp = args.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else None
    # bf16 doesn't need loss scaling; GradScaler with enabled=False is a no-op passthrough
    scaler = torch.amp.GradScaler(device.type, enabled=False)
    if use_amp:
        print(f"AMP enabled with {amp_dtype}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(exist_ok=True)
    best_val_recon = float("inf")

    log_path = ckpt_dir / "fsq_log.csv"
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "train_recon", "train_slow", "train_uniform",
                         "val_recon", "lr", "time_s"])

    try:
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_recon, train_slow, train_uniform = train_epoch(
                model, train_loader, optimizer, args.alpha_slow, args.alpha_uniform,
                scaler=scaler, amp_dtype=amp_dtype)
            val_recon = val_epoch(model, val_loader, amp_dtype=amp_dtype)
            scheduler.step()
            dt = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch:3d}/{args.epochs} ({dt:.1f}s) | "
                f"Train: recon={train_recon:.4f} slow={train_slow:.4f} unif={train_uniform:.4f} | "
                f"Val: recon={val_recon:.4f} | LR: {lr:.1e}"
            )

            log_writer.writerow([
                epoch, f"{train_recon:.6f}", f"{train_slow:.6f}", f"{train_uniform:.6f}",
                f"{val_recon:.6f}", f"{lr:.1e}", f"{dt:.1f}"
            ])
            log_file.flush()

            if val_recon < best_val_recon:
                best_val_recon = val_recon
                torch.save(model.state_dict(), ckpt_dir / "fsq_best.pt")
    except KeyboardInterrupt:
        print("\nInterrupted — saving final checkpoint...")

    log_file.close()
    torch.save(model.state_dict(), ckpt_dir / "fsq_final.pt")
    print(f"\nTraining complete. Best val recon: {best_val_recon:.4f}")
    print(f"Checkpoints saved to {ckpt_dir}/")


if __name__ == "__main__":
    main()
