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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepdash.wandb_utils import wandb_init, wandb_log, wandb_finish
from torch.utils.data import DataLoader, Dataset
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepdash.fsq import FSQVAE, fsqvae_loss, grwm_slowness, grwm_uniformity


class FramePairDataset(Dataset):
    """Loads consecutive frame pairs from episodes for GRWM temporal slowness.

    Each sample is (frame_t, frame_t+1). Uses memory-mapped numpy arrays
    to avoid loading all frames into RAM at once.
    """

    def __init__(self, episode_dirs, device=None):
        self.device = device
        # Group frame pairs by episode file, store as contiguous arrays
        all_ft, all_ft1 = [], []
        for ep_dir in episode_dirs:
            frames = np.load(ep_dir / "frames.npy")  # (T, 64, 64) uint8
            if len(frames) < 2:
                continue
            all_ft.append(frames[:-1])
            all_ft1.append(frames[1:])
        # Single contiguous array in RAM (uint8, ~1 byte/pixel)
        self.ft = np.concatenate(all_ft)    # (N, 64, 64) uint8
        self.ft1 = np.concatenate(all_ft1)  # (N, 64, 64) uint8
        del all_ft, all_ft1
        print(f"  Dataset: {len(self.ft):,} frame pairs "
              f"({self.ft.nbytes / 1e9:.1f} GB RAM)")

    def __len__(self):
        return len(self.ft)

    def __getitem__(self, idx):
        ft = torch.from_numpy(self.ft[idx].copy()).float().unsqueeze(0) / 255.0
        ft1 = torch.from_numpy(self.ft1[idx].copy()).float().unsqueeze(0) / 255.0
        return ft, ft1


def train_epoch(model, loader, optimizer, alpha_slow, alpha_uniform,
                scaler=None, amp_dtype=None):
    model.train()
    total_recon, total_slow, total_uniform, n = 0, 0, 0, 0
    for ft, ft1 in loader:
        ft = ft.cuda(non_blocking=True)
        ft1 = ft1.cuda(non_blocking=True)
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
        ft = ft.cuda(non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp_dtype is not None, dtype=amp_dtype):
            recon_t, _, _ = model(ft)
            recon_loss = fsqvae_loss(recon_t, ft)
        bs = ft.size(0)
        total_recon += recon_loss.item() * bs
        n += bs
    return total_recon / n


def main():
    parser = argparse.ArgumentParser(description="Train FSQ-VAE on Geometry Dash frames")
    parser.add_argument("--config", default=None, help="YAML config path (default: configs/v4.yaml)")
    parser.add_argument("--episodes-dir", default=None)
    parser.add_argument("--expert-episodes-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--levels", type=int, nargs="+", default=None,
                        help="FSQ quantization levels per channel")
    parser.add_argument("--alpha-slow", type=float, default=None,
                        help="Weight for GRWM temporal slowness loss")
    parser.add_argument("--alpha-uniform", type=float, default=None,
                        help="Weight for GRWM uniformity loss")
    parser.add_argument("--val-ratio", type=float, default=None,
                        help="Fraction of episodes for validation")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--amp", action="store_true", default=None,
                        help="Enable bf16 automatic mixed precision (A100+)")
    parser.add_argument("--compile", action="store_true", default=None,
                        help="Use torch.compile for faster training")
    args = parser.parse_args()

    from deepdash.config import apply_config
    apply_config(args, section="fsq")
    args.amp = bool(args.amp)
    setattr(args, 'compile', bool(getattr(args, 'compile', False)))

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    import json
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(ckpt_dir / "fsq_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    wandb_init(project="deepdash", name=f"fsq-{args.levels}", config=vars(args))

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Global episode-level split (shared across all models)
    from deepdash.data_split import get_val_episodes, is_val_episode
    val_set = get_val_episodes(args.episodes_dir, args.expert_episodes_dir)

    import re
    shift_re = re.compile(r"_s[+-]\d+_[+-]\d+$")

    # Verify shift directories exist (created by shift_episodes.py)
    has_shifts = False
    for ep_dir in [args.episodes_dir, args.expert_episodes_dir]:
        p = Path(ep_dir)
        if p.exists():
            for ep in p.glob("*"):
                if ep.is_dir() and shift_re.search(ep.name):
                    has_shifts = True
                    break
        if has_shifts:
            break
    if not has_shifts:
        print("ERROR: No shift augmentation found. "
              "Run scripts/shift_episodes.py first.")
        sys.exit(1)

    # Delete tokens.npy from all episodes (will be re-created after FSQ)
    for ep_dir in [args.episodes_dir, args.expert_episodes_dir]:
        p = Path(ep_dir)
        if not p.exists():
            continue
        for ep in p.glob("*"):
            if not ep.is_dir():
                continue
            tok = ep / "tokens.npy"
            if tok.exists():
                tok.unlink()

    all_episodes = []
    for ep_dir in [args.episodes_dir, args.expert_episodes_dir]:
        p = Path(ep_dir)
        if p.exists():
            all_episodes.extend(
                ep for ep in sorted(p.glob("*"))
                if ep.is_dir() and (ep / "frames.npy").exists()
            )

    train_eps = [ep for ep in all_episodes if not is_val_episode(ep.name, val_set)]
    val_eps = [ep for ep in all_episodes if is_val_episode(ep.name, val_set)]

    print(f"Episodes: {len(all_episodes)} total, {len(train_eps)} train, {len(val_eps)} val")

    train_dataset = FramePairDataset(train_eps, device=device)
    val_dataset = FramePairDataset(val_eps, device=device)
    print(f"Frame pairs: {len(train_dataset)} train, {len(val_dataset)} val")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    model = FSQVAE(levels=args.levels).to(device)
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=True)
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
        model.load_state_dict(state)
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

            wandb_log({
                "epoch": epoch,
                "train/recon": train_recon, "train/slow": train_slow,
                "train/uniform": train_uniform,
                "val/recon": val_recon, "lr": lr,
            })

            if val_recon < best_val_recon:
                best_val_recon = val_recon
                # Strip _orig_mod. prefix from torch.compile for portable checkpoints
                clean_state = {k.removeprefix("_orig_mod."): v
                               for k, v in model.state_dict().items()}
                torch.save(clean_state, ckpt_dir / "fsq_best.pt")
    except KeyboardInterrupt:
        print("\nInterrupted — saving final checkpoint...")

    log_file.close()
    wandb_finish()
    clean_state = {k.removeprefix("_orig_mod."): v
                   for k, v in model.state_dict().items()}
    torch.save(clean_state, ckpt_dir / "fsq_final.pt")
    print(f"\nTraining complete. Best val recon: {best_val_recon:.4f}")
    print(f"Checkpoints saved to {ckpt_dir}/")


if __name__ == "__main__":
    main()
