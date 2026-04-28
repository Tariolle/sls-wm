"""Train a beta-VAE under V3-deploy's exact FSQ training pipeline.

This script is a 1-to-1 mirror of V3-deploy/scripts/train_fsq.py with
the model and loss swapped for BetaVAE. Same dataset, same global
episode-level split, same shift augmentation, same Adam(lr=4e-3),
same CosineAnnealingLR(T_max=epochs, eta_min=1e-5), same 200 epochs,
same batch size 32, same AMP/compile flags.

The val_recon column is MSE in identical units to FSQ's val_recon, so
the two logs can be plotted on the same axes.

Usage:
    python scripts/train_beta_vae.py --amp
    python scripts/train_beta_vae.py --beta 1.0 --epochs 200
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepdash.beta_vae import BetaVAE, beta_vae_loss


class FramePairDataset(Dataset):
    """Same FramePairDataset as V3-deploy's train_fsq.py.

    Frame pairs let us reuse the exact dataloader with identical batch
    composition. The beta-VAE consumes both frames (averages losses)
    so each batch sees the same number of forward passes as FSQ.
    """

    def __init__(self, episode_dirs, device=None):
        self.pairs = []
        for ep_dir in episode_dirs:
            frames = np.load(ep_dir / "frames.npy")  # (T, 64, 64) uint8
            for i in range(len(frames) - 1):
                self.pairs.append((frames[i], frames[i + 1]))

        t_data = np.array([p[0] for p in self.pairs])
        t1_data = np.array([p[1] for p in self.pairs])
        self.frames_t = torch.from_numpy(t_data).float().unsqueeze(1) / 255.0
        self.frames_t1 = torch.from_numpy(t1_data).float().unsqueeze(1) / 255.0
        self.pairs = None

        if device and device.type == "cuda":
            self.frames_t = self.frames_t.to(device)
            self.frames_t1 = self.frames_t1.to(device)

    def __len__(self):
        return len(self.frames_t)

    def __getitem__(self, idx):
        return self.frames_t[idx], self.frames_t1[idx]


def augment_batch(ft, ft1, pad=4, size=64):
    """Identical batch-level random shift augmentation as V3-deploy."""
    B = ft.size(0)
    di = torch.randint(0, 2 * pad + 1, (B,), device=ft.device, dtype=ft.dtype)
    dj = torch.randint(0, 2 * pad + 1, (B,), device=ft.device, dtype=ft.dtype)
    shift_i = (di - pad) / (size / 2)
    shift_j = (dj - pad) / (size / 2)
    grid_y = torch.linspace(-1, 1, size, device=ft.device)
    grid_x = torch.linspace(-1, 1, size, device=ft.device)
    gy, gx = torch.meshgrid(grid_y, grid_x, indexing='ij')
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
    grid = grid.clone()
    grid[..., 0] += shift_j.view(B, 1, 1)
    grid[..., 1] += shift_i.view(B, 1, 1)
    out_t = F.grid_sample(ft, grid, mode='nearest', padding_mode='border', align_corners=True)
    out_t1 = F.grid_sample(ft1, grid, mode='nearest', padding_mode='border', align_corners=True)
    return out_t, out_t1


def train_epoch(model, loader, optimizer, beta,
                scaler=None, amp_dtype=None, augment=False):
    model.train()
    total_recon, total_kl, n = 0.0, 0.0, 0
    for ft, ft1 in loader:
        if augment:
            ft, ft1 = augment_batch(ft, ft1)
        with torch.amp.autocast("cuda", enabled=amp_dtype is not None, dtype=amp_dtype):
            recon_t, mu_t, logvar_t = model(ft)
            recon_t1, mu_t1, logvar_t1 = model(ft1)

            loss_t, recon_l_t, kl_l_t = beta_vae_loss(recon_t, ft, mu_t, logvar_t, beta=beta)
            loss_t1, recon_l_t1, kl_l_t1 = beta_vae_loss(recon_t1, ft1, mu_t1, logvar_t1, beta=beta)

            recon_loss = (recon_l_t + recon_l_t1) / 2
            kl_loss = (kl_l_t + kl_l_t1) / 2
            loss = (loss_t + loss_t1) / 2

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = ft.size(0)
        total_recon += recon_loss.item() * bs
        total_kl += kl_loss.item() * bs
        n += bs
    return total_recon / n, total_kl / n


@torch.no_grad()
def val_epoch(model, loader, beta, amp_dtype=None):
    model.eval()
    total_recon, total_kl, n = 0.0, 0.0, 0
    for ft, _ft1 in loader:
        with torch.amp.autocast("cuda", enabled=amp_dtype is not None, dtype=amp_dtype):
            recon_t, mu_t, logvar_t = model(ft)
            _, recon_l, kl_l = beta_vae_loss(recon_t, ft, mu_t, logvar_t, beta=beta)
        bs = ft.size(0)
        total_recon += recon_l.item() * bs
        total_kl += kl_l.item() * bs
        n += bs
    return total_recon / n, total_kl / n


def main():
    parser = argparse.ArgumentParser(description="Train beta-VAE under V3-deploy FSQ pipeline")
    parser.add_argument("--episodes-dir", default="data/death_episodes")
    parser.add_argument("--expert-episodes-dir", default="data/expert_episodes")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=4e-3)
    parser.add_argument("--checkpoint-dir", default="checkpoints_beta_vae")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--beta", type=float, default=1.0,
                        help="KL weight. 1.0 = standard VAE; <1 = sharper recon.")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", default="default",
                        choices=["default", "reduce-overhead", "max-autotune"])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    from deepdash.data_split import get_val_episodes, is_val_episode
    val_set = get_val_episodes(args.episodes_dir, args.expert_episodes_dir)

    all_episodes = []
    for ep_dir in [args.episodes_dir, args.expert_episodes_dir]:
        p = Path(ep_dir)
        if p.exists():
            all_episodes.extend(
                ep for ep in sorted(p.glob("*"))
                if (ep / "frames.npy").exists()
            )

    train_eps = [ep for ep in all_episodes if not is_val_episode(ep.name, val_set)]
    val_eps = [ep for ep in all_episodes if is_val_episode(ep.name, val_set)]

    print(f"Episodes: {len(all_episodes)} total, {len(train_eps)} train, {len(val_eps)} val")

    train_dataset = FramePairDataset(train_eps, device=device)
    val_dataset = FramePairDataset(val_eps, device=device)
    print(f"Frame pairs: {len(train_dataset)} train, {len(val_dataset)} val")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=False)

    model = BetaVAE(latent_dim=args.latent_dim).to(device)
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=True)
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
        model.load_state_dict(state)
        print(f"Resumed from {args.resume}")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    print(f"Beta: {args.beta} | Latent dim: {args.latent_dim}")

    if args.compile:
        try:
            model = torch.compile(model, mode=args.compile_mode)
            print(f"torch.compile enabled (mode={args.compile_mode})")
        except Exception as e:
            print(f"torch.compile failed ({e}), continuing without it")

    use_amp = args.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else None
    scaler = torch.amp.GradScaler(device.type, enabled=False)
    if use_amp:
        print(f"AMP enabled with {amp_dtype}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(exist_ok=True)
    best_val_recon = float("inf")

    log_path = ckpt_dir / "beta_vae_log.csv"
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "train_recon", "train_kl",
                         "val_recon", "val_kl", "lr", "time_s"])

    try:
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_recon, train_kl = train_epoch(
                model, train_loader, optimizer, args.beta,
                scaler=scaler, amp_dtype=amp_dtype, augment=True)
            val_recon, val_kl = val_epoch(model, val_loader, args.beta, amp_dtype=amp_dtype)
            scheduler.step()
            dt = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch:3d}/{args.epochs} ({dt:.1f}s) | "
                f"Train: recon={train_recon:.4f} kl={train_kl:.4f} | "
                f"Val: recon={val_recon:.4f} kl={val_kl:.4f} | LR: {lr:.1e}"
            )

            log_writer.writerow([
                epoch, f"{train_recon:.6f}", f"{train_kl:.6f}",
                f"{val_recon:.6f}", f"{val_kl:.6f}", f"{lr:.1e}", f"{dt:.1f}"
            ])
            log_file.flush()

            if val_recon < best_val_recon:
                best_val_recon = val_recon
                clean_state = {k.removeprefix("_orig_mod."): v
                               for k, v in model.state_dict().items()}
                torch.save(clean_state, ckpt_dir / "beta_vae_best.pt")
    except KeyboardInterrupt:
        print("\nInterrupted — saving final checkpoint...")

    log_file.close()
    clean_state = {k.removeprefix("_orig_mod."): v
                   for k, v in model.state_dict().items()}
    torch.save(clean_state, ckpt_dir / "beta_vae_final.pt")
    print(f"\nTraining complete. Best val recon: {best_val_recon:.4f}")
    print(f"Checkpoints saved to {ckpt_dir}/")


if __name__ == "__main__":
    main()
