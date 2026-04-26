"""Train the FSQ-VAE on raw Geometry Dash frame pairs.

V7 Phase 0 reproduction of V3-deploy's FSQ recipe (commit 66d5e2c) with
two corrections relative to the original:

  1. Globbing filter excludes shift-augmented episode dirs
     (``_s[+-]\\d+_[+-]\\d+`` suffix). The V3-deploy code accidentally
     globbed those as if they were independent episodes, multiplying the
     epoch length 5-15x. See ``project_v3_fsq_aug_bug.md``.

  2. ``--steps-per-epoch`` knob caps gradient steps per epoch. Use this
     to restore the lost compute by training longer (e.g. ``--epochs 1000``
     or via the cap), without resurrecting the duplication bug.

YAML config support via ``apply_config(section="fsq")`` — same convention
as ``train_world_model.py``.
"""

import argparse
import csv
import re
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepdash.fsq import FSQVAE, fsqvae_loss, grwm_slowness, grwm_uniformity


SHIFT_AUG_RE = re.compile(r"_s[+-]\d+_[+-]\d+$")


class FramePairDataset(Dataset):
    """Loads consecutive frame pairs from episodes for GRWM temporal slowness.

    Each sample is (frame_t, frame_t+1). Split by episode, not by frame.
    """

    def __init__(self, episode_dirs, device=None):
        pairs_t, pairs_t1 = [], []
        for ep_dir in episode_dirs:
            frames = np.load(ep_dir / "frames.npy")  # (T, 64, 64) uint8
            for i in range(len(frames) - 1):
                pairs_t.append(frames[i])
                pairs_t1.append(frames[i + 1])
        t_data = np.stack(pairs_t)
        t1_data = np.stack(pairs_t1)
        self.frames_t = torch.from_numpy(t_data).float().unsqueeze(1) / 255.0
        self.frames_t1 = torch.from_numpy(t1_data).float().unsqueeze(1) / 255.0
        if device and device.type == "cuda":
            self.frames_t = self.frames_t.to(device)
            self.frames_t1 = self.frames_t1.to(device)

    def __len__(self):
        return len(self.frames_t)

    def __getitem__(self, idx):
        return self.frames_t[idx], self.frames_t1[idx]


def augment_batch(ft, ft1, pad=4, size=64):
    """Per-sample random shift augmentation via grid_sample with edge padding."""
    B = ft.size(0)
    di = torch.randint(0, 2 * pad + 1, (B,), device=ft.device, dtype=ft.dtype)
    dj = torch.randint(0, 2 * pad + 1, (B,), device=ft.device, dtype=ft.dtype)
    shift_i = (di - pad) / (size / 2)
    shift_j = (dj - pad) / (size / 2)
    grid_y = torch.linspace(-1, 1, size, device=ft.device)
    grid_x = torch.linspace(-1, 1, size, device=ft.device)
    gy, gx = torch.meshgrid(grid_y, grid_x, indexing="ij")
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
    grid = grid.clone()
    grid[..., 0] += shift_j.view(B, 1, 1)
    grid[..., 1] += shift_i.view(B, 1, 1)
    out_t = F.grid_sample(ft, grid, mode="nearest", padding_mode="border", align_corners=True)
    out_t1 = F.grid_sample(ft1, grid, mode="nearest", padding_mode="border", align_corners=True)
    return out_t, out_t1


def train_epoch(model, loader, optimizer, alpha_slow, alpha_uniform,
                amp_dtype=None, augment=True, max_steps=0):
    model.train()
    total_recon, total_slow, total_uniform, n = 0.0, 0.0, 0.0, 0
    step = 0
    for ft, ft1 in loader:
        if augment:
            ft, ft1 = augment_batch(ft, ft1)
        with torch.amp.autocast("cuda", enabled=amp_dtype is not None, dtype=amp_dtype):
            recon_t, z_e_t, _ = model(ft)
            recon_t1, z_e_t1, _ = model(ft1)
            recon_loss = (fsqvae_loss(recon_t, ft) + fsqvae_loss(recon_t1, ft1)) / 2
            slow_loss = grwm_slowness(z_e_t, z_e_t1)
            uniform_loss = grwm_uniformity(z_e_t)
            loss = recon_loss + alpha_slow * slow_loss + alpha_uniform * uniform_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        bs = ft.size(0)
        total_recon += recon_loss.item() * bs
        total_slow += slow_loss.item() * bs
        total_uniform += uniform_loss.item() * bs
        n += bs
        step += 1
        if max_steps and step >= max_steps:
            break
    return total_recon / n, total_slow / n, total_uniform / n


@torch.no_grad()
def val_epoch(model, loader, amp_dtype=None):
    model.eval()
    total_recon, n = 0.0, 0
    for ft, _ in loader:
        with torch.amp.autocast("cuda", enabled=amp_dtype is not None, dtype=amp_dtype):
            recon_t, _, _ = model(ft)
            recon_loss = fsqvae_loss(recon_t, ft)
        bs = ft.size(0)
        total_recon += recon_loss.item() * bs
        n += bs
    return total_recon / n


def main():
    parser = argparse.ArgumentParser(description="Train FSQ-VAE on Geometry Dash frames")
    parser.add_argument("--config", default=None, help="YAML config path")
    parser.add_argument("--episodes-dir", default="data/death_episodes")
    parser.add_argument("--expert-episodes-dir", default="data/expert_episodes")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr-min", type=float, default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--levels", type=int, nargs="+", default=None)
    parser.add_argument("--alpha-slow", type=float, default=None)
    parser.add_argument("--alpha-uniform", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps-per-epoch", type=int, default=None,
                        help="Cap gradient steps per epoch (0/None = full loader). "
                             "Useful for restoring V3-deploy's accidental 5-15x "
                             "compute via longer training without aug-dir duplication.")
    parser.add_argument("--amp-dtype", choices=["bfloat16", "float16", "none"],
                        default=None, help="Default bfloat16 (A100).")
    parser.add_argument("--compile-mode",
                        choices=["reduce-overhead", "default", "none"],
                        default=None, help="torch.compile mode. Default reduce-overhead.")
    args = parser.parse_args()

    from deepdash.config import apply_config
    apply_config(args, section="fsq")

    # Defaults if config missing them
    args.epochs = args.epochs or 200
    args.batch_size = args.batch_size or 2048
    args.lr = args.lr or 1e-3
    args.lr_min = args.lr_min if args.lr_min is not None else 1e-5
    args.checkpoint_dir = args.checkpoint_dir or "checkpoints"
    args.levels = args.levels or [8, 5, 5, 5]
    args.alpha_slow = args.alpha_slow if args.alpha_slow is not None else 0.1
    args.alpha_uniform = args.alpha_uniform if args.alpha_uniform is not None else 0.01
    args.amp_dtype = args.amp_dtype or "bfloat16"
    args.compile_mode = args.compile_mode or "reduce-overhead"

    def _sigterm_handler(sig, frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, _sigterm_handler)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    from deepdash.data_split import get_val_episodes, is_val_episode
    val_set = get_val_episodes(args.episodes_dir, args.expert_episodes_dir)

    all_episodes = []
    for ep_dir in [args.episodes_dir, args.expert_episodes_dir]:
        p = Path(ep_dir)
        if p.exists():
            all_episodes.extend(
                ep for ep in sorted(p.glob("*"))
                if (ep / "frames.npy").exists()
                and not SHIFT_AUG_RE.search(ep.name)  # bug fix: skip aug_dirs
            )

    train_eps = [ep for ep in all_episodes if not is_val_episode(ep.name, val_set)]
    val_eps = [ep for ep in all_episodes if is_val_episode(ep.name, val_set)]
    print(f"Episodes: {len(all_episodes)} total, {len(train_eps)} train, {len(val_eps)} val "
          f"(shift-aug dirs filtered out)")

    train_dataset = FramePairDataset(train_eps, device=device)
    val_dataset = FramePairDataset(val_eps, device=device)
    print(f"Frame pairs: {len(train_dataset)} train, {len(val_dataset)} val")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=False)

    model = FSQVAE(levels=args.levels).to(device)
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=True)
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
        model.load_state_dict(state)
        print(f"Resumed from {args.resume}")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    print(f"FSQ levels: {args.levels} -> {model.codebook_size} implicit codes")

    if args.compile_mode != "none":
        try:
            model = torch.compile(model, mode=args.compile_mode)
            print(f"torch.compile enabled (mode={args.compile_mode})")
        except Exception as e:
            print(f"torch.compile failed ({e}), continuing without it")

    amp_dtype = None
    if args.amp_dtype == "bfloat16":
        amp_dtype = torch.bfloat16
    elif args.amp_dtype == "float16":
        amp_dtype = torch.float16
    if amp_dtype is not None:
        print(f"AMP enabled with {amp_dtype}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr_min)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_recon = float("inf")

    log_path = ckpt_dir / "fsq_log.csv"
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "train_recon", "train_slow", "train_uniform",
                         "val_recon", "lr", "time_s"])

    max_steps = args.steps_per_epoch or 0
    try:
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_recon, train_slow, train_uniform = train_epoch(
                model, train_loader, optimizer, args.alpha_slow, args.alpha_uniform,
                amp_dtype=amp_dtype, augment=True, max_steps=max_steps)
            val_recon = val_epoch(model, val_loader, amp_dtype=amp_dtype)
            scheduler.step()
            dt = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch:4d}/{args.epochs} ({dt:.1f}s) | "
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
                clean_state = {k.removeprefix("_orig_mod."): v
                               for k, v in model.state_dict().items()}
                torch.save(clean_state, ckpt_dir / "fsq_best.pt")
    except KeyboardInterrupt:
        print("\nInterrupted - saving final checkpoint...")

    log_file.close()
    clean_state = {k.removeprefix("_orig_mod."): v
                   for k, v in model.state_dict().items()}
    torch.save(clean_state, ckpt_dir / "fsq_final.pt")
    print(f"\nTraining complete. Best val recon: {best_val_recon:.4f}")
    print(f"Checkpoints saved to {ckpt_dir}/")


if __name__ == "__main__":
    main()
