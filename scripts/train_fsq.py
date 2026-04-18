"""Train the FSQ-VAE on preprocessed Geometry Dash frames.

Uses frame pairs from episodes for GRWM temporal slowness loss.

Usage:
    python scripts/train_fsq.py
    python scripts/train_fsq.py --epochs 400 --alpha-slow 0.1
"""

import argparse
import csv
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepdash.wandb_utils import wandb_init, wandb_log, wandb_finish
from torch.utils.data import DataLoader, Dataset
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepdash.fsq import FSQVAE, fsqvae_loss, grwm_slowness, grwm_uniformity


_SHIFT_RE = re.compile(r"_s[+-]\d+_[+-]\d+$")


class FramePairDataset(Dataset):
    """Loads consecutive frame pairs from episodes for GRWM temporal slowness.

    Each sample is (frame_t, frame_t+1). Supports per-epoch K-of-N shift
    variant subsampling: set k_shifts=K and call set_epoch(e) each epoch
    to draw K distinct variants per base episode.
    """

    def __init__(self, episode_dirs, k_shifts=None, seed=42, device=None):
        self.device = device
        self.k_shifts = k_shifts
        self.seed = seed

        # Group episode dirs by base name (strip shift suffix)
        groups = defaultdict(list)
        for ep_dir in episode_dirs:
            base = _SHIFT_RE.sub("", ep_dir.name)
            groups[base].append(ep_dir)

        all_ft, all_ft1 = [], []
        # gv_ranges[gid] -> list of (start, end) slices for each variant of this base ep
        self.gv_ranges = []
        offset = 0
        for base in sorted(groups):
            variants = sorted(groups[base], key=lambda p: p.name)
            group_slices = []
            for ep_dir in variants:
                frames = np.load(ep_dir / "frames.npy")  # (T, 64, 64) uint8
                if len(frames) < 2:
                    continue
                n = len(frames) - 1
                all_ft.append(frames[:-1])
                all_ft1.append(frames[1:])
                group_slices.append((offset, offset + n))
                offset += n
            if group_slices:
                self.gv_ranges.append(group_slices)

        self.ft = np.concatenate(all_ft)    # (N, 64, 64) uint8
        self.ft1 = np.concatenate(all_ft1)  # (N, 64, 64) uint8
        del all_ft, all_ft1

        max_variants = max((len(g) for g in self.gv_ranges), default=0)
        self.set_epoch(0)
        subsample_note = (f", K={self.k_shifts}/{max_variants} per base ep"
                          if self.k_shifts is not None else "")
        print(f"  Dataset: {len(self.ft):,} pairs total, "
              f"{len(self.active_idx):,} active "
              f"({self.ft.nbytes / 1e9:.1f} GB RAM, "
              f"{len(self.gv_ranges)} base eps{subsample_note})")

    def set_epoch(self, epoch):
        """Rebuild active index list. No-op if k_shifts is None or >= max variants."""
        if self.k_shifts is None:
            self.active_idx = np.arange(len(self.ft), dtype=np.int64)
            return
        rng = np.random.default_rng(self.seed + epoch)
        chunks = []
        for variants in self.gv_ranges:
            n_avail = len(variants)
            k = min(self.k_shifts, n_avail)
            chosen = rng.choice(n_avail, size=k, replace=False)
            for ci in chosen:
                start, end = variants[ci]
                chunks.append(np.arange(start, end, dtype=np.int64))
        self.active_idx = np.concatenate(chunks) if chunks else np.array([], dtype=np.int64)

    def __len__(self):
        return len(self.active_idx)

    def __getitem__(self, idx):
        real = self.active_idx[idx]
        # Return uint8 (1, 64, 64). Float conversion + /255 happens on GPU in the
        # training loop (saves ~4x on H2D traffic and the expensive CPU-side divide).
        ft = torch.from_numpy(self.ft[real].copy()).unsqueeze(0)
        ft1 = torch.from_numpy(self.ft1[real].copy()).unsqueeze(0)
        return ft, ft1


def _augment_pair_gpu(ft, ft1, brightness_range, contrast_range):
    """GPU-side augmentation. Brightness/contrast sampled per-sample, pair-coherent
    (same value for (f_t, f_{t+1}), different across batch elements).
    """
    B = ft.shape[0]
    device = ft.device
    if brightness_range > 0.0:
        b = (torch.rand(B, 1, 1, 1, device=device) * 2.0 - 1.0) * brightness_range
        ft = ft + b
        ft1 = ft1 + b
    if contrast_range > 0.0:
        c = 1.0 + (torch.rand(B, 1, 1, 1, device=device) * 2.0 - 1.0) * contrast_range
        ft = (ft - 0.5) * c + 0.5
        ft1 = (ft1 - 0.5) * c + 0.5
    return ft.clamp(0.0, 1.0), ft1.clamp(0.0, 1.0)


def train_epoch(model, loader, optimizer, alpha_slow, alpha_uniform,
                scaler=None, amp_dtype=None,
                brightness_range=0.0, contrast_range=0.0):
    model.train()
    total_recon, total_slow, total_uniform, n = 0, 0, 0, 0
    t_data = t_aug = t_fwd = t_bwd = t_opt = 0.0
    sync = torch.cuda.synchronize
    t_prev = time.perf_counter()

    for ft, ft1 in loader:
        ft = ft.cuda(non_blocking=True).float().mul_(1.0 / 255.0)
        ft1 = ft1.cuda(non_blocking=True).float().mul_(1.0 / 255.0)
        sync()
        t0 = time.perf_counter()
        t_data += t0 - t_prev

        ft, ft1 = _augment_pair_gpu(ft, ft1, brightness_range, contrast_range)
        sync()
        t1 = time.perf_counter()
        t_aug += t1 - t0

        with torch.amp.autocast("cuda", enabled=amp_dtype is not None, dtype=amp_dtype):
            recon_t, z_e_t, _ = model(ft)
            recon_t1, z_e_t1, _ = model(ft1)
            recon_loss = (fsqvae_loss(recon_t, ft) + fsqvae_loss(recon_t1, ft1)) / 2
            slow_loss = grwm_slowness(z_e_t, z_e_t1)
            uniform_loss = grwm_uniformity(z_e_t)
            loss = recon_loss + alpha_slow * slow_loss + alpha_uniform * uniform_loss
        sync()
        t2 = time.perf_counter()
        t_fwd += t2 - t1

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        sync()
        t3 = time.perf_counter()
        t_bwd += t3 - t2

        scaler.step(optimizer)
        scaler.update()
        sync()
        t4 = time.perf_counter()
        t_opt += t4 - t3

        bs = ft.size(0)
        total_recon += recon_loss.item() * bs
        total_slow += slow_loss.item() * bs
        total_uniform += uniform_loss.item() * bs
        n += bs
        t_prev = time.perf_counter()

    timing = {"data": t_data, "aug": t_aug, "fwd": t_fwd, "bwd": t_bwd, "opt": t_opt}
    return total_recon / n, total_slow / n, total_uniform / n, timing


@torch.no_grad()
def val_epoch(model, loader, amp_dtype=None):
    model.eval()
    total_recon, n = 0, 0
    all_indices = []
    for ft, ft1 in loader:
        ft = ft.cuda(non_blocking=True).float().mul_(1.0 / 255.0)
        with torch.amp.autocast("cuda", enabled=amp_dtype is not None, dtype=amp_dtype):
            recon_t, _, _ = model(ft)
            recon_loss = fsqvae_loss(recon_t, ft)
        # Collect codebook indices for utilization metric
        _, indices = model.fsq(model.encoder(ft))
        all_indices.append(indices.reshape(-1).cpu())
        bs = ft.size(0)
        total_recon += recon_loss.item() * bs
        n += bs
    # Codebook utilization: fraction of codes actually used
    codebook_size = 1
    for L in model.fsq.levels.tolist():
        codebook_size *= int(L)
    all_indices = torch.cat(all_indices)
    usage = all_indices.unique().numel() / codebook_size
    counts = torch.bincount(all_indices, minlength=codebook_size).float()
    probs = counts / counts.sum()
    entropy = -(probs[probs > 0] * probs[probs > 0].log()).sum()
    perplexity = entropy.exp().item() / codebook_size
    return total_recon / n, usage, perplexity


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
                        help="Enable automatic mixed precision")
    parser.add_argument("--amp-dtype", choices=["bfloat16", "float16"], default=None,
                        help="AMP dtype: bfloat16 (A100 default, no GradScaler needed) "
                             "or float16 (Turing/pre-Ampere, requires GradScaler)")
    parser.add_argument("--compile", action="store_true", default=None,
                        help="Use torch.compile for faster training")
    parser.add_argument("--compile-mode",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        default=None,
                        help="torch.compile mode (max-autotune uses cudagraphs; "
                             "fails on Turing backward for this model — use default)")
    parser.add_argument("--k-shifts", type=int, default=None,
                        help="Per-epoch, draw K of the available shift variants "
                             "per base ep (None or >=all variants = use all)")
    parser.add_argument("--lr-schedule", choices=["cosine", "constant"], default=None,
                        help="LR schedule for FSQ training")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="DataLoader workers (use 0 on Windows when dataset "
                             "exceeds a few hundred MB to avoid pickle issues)")
    parser.add_argument("--max-base-episodes", type=int, default=None,
                        help="Cap the dataset to the first N base episodes "
                             "(shift variants of kept bases are all included). "
                             "Counts base eps across death + expert combined.")
    parser.add_argument("--brightness-range", type=float, default=None,
                        help="Uniform brightness offset in [-r, +r] applied to "
                             "both input and target, same value per frame pair")
    parser.add_argument("--contrast-range", type=float, default=None,
                        help="Uniform contrast factor in [1-r, 1+r] applied to "
                             "both input and target, same value per frame pair")
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

    max_base = getattr(args, "max_base_episodes", None)
    if max_base is not None and max_base > 0:
        base_seen = set()
        capped = []
        for ep in all_episodes:
            base = _SHIFT_RE.sub("", ep.name)
            if base not in base_seen:
                if len(base_seen) >= max_base:
                    continue
                base_seen.add(base)
            capped.append(ep)
        print(f"Capped to {len(base_seen)} base episodes "
              f"({len(capped)} total incl shifts; was {len(all_episodes)})")
        all_episodes = capped

    train_eps = [ep for ep in all_episodes if not is_val_episode(ep.name, val_set)]
    val_eps = [ep for ep in all_episodes if is_val_episode(ep.name, val_set)]

    print(f"Episodes: {len(all_episodes)} total, {len(train_eps)} train, {len(val_eps)} val")

    brightness = getattr(args, "brightness_range", None) or 0.0
    contrast = getattr(args, "contrast_range", None) or 0.0
    print(f"Augmentation: brightness_range=±{brightness}, contrast_range=±{contrast}")
    train_dataset = FramePairDataset(
        train_eps, k_shifts=args.k_shifts, seed=args.seed, device=device)
    val_dataset = FramePairDataset(val_eps, device=device)
    print(f"Frame pairs: {len(train_dataset)} train, {len(val_dataset)} val")

    num_workers = getattr(args, "num_workers", None)
    if num_workers is None:
        num_workers = 4
    print(f"DataLoader workers: {num_workers}")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)

    model = FSQVAE(levels=args.levels).to(device)
    if args.resume:
        resume_state_path = ckpt_dir / "fsq_state.pt"
        if resume_state_path.exists():
            state = torch.load(resume_state_path, map_location=device, weights_only=False)
            weights = {k.removeprefix("_orig_mod."): v for k, v in state["model"].items()}
            model.load_state_dict(weights)
            print(f"Resumed model from {resume_state_path}")
        elif Path(args.resume).exists():
            state = torch.load(args.resume, map_location=device, weights_only=True)
            state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
            model.load_state_dict(state)
            print(f"Resumed from {args.resume}")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    print(f"FSQ levels: {args.levels} -> {model.codebook_size} implicit codes")

    if args.compile:
        compile_mode = getattr(args, "compile_mode", None) or "default"
        # Inductor's automatic channels_last conversion hits a stride assertion in
        # conv_backward on Turing for this model. Force NCHW layout throughout.
        try:
            import torch._inductor.config as inductor_cfg
            inductor_cfg.layout_optimization = False
        except Exception:
            pass
        try:
            model = torch.compile(model, mode=compile_mode)
            print(f"torch.compile enabled ({compile_mode}, layout_optimization=False)")
        except Exception as e:
            print(f"torch.compile failed ({e}), continuing without it")

    use_amp = args.amp and device.type == "cuda"
    amp_dtype_name = getattr(args, "amp_dtype", None) or "bfloat16"
    amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[amp_dtype_name] if use_amp else None
    # fp16 needs loss scaling; bf16 doesn't (passthrough)
    scaler_enabled = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler(device.type, enabled=scaler_enabled)
    if use_amp:
        print(f"AMP enabled with {amp_dtype_name} (GradScaler={'on' if scaler_enabled else 'off'})")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_schedule = getattr(args, "lr_schedule", None) or "cosine"
    if lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-5)
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    print(f"LR schedule: {lr_schedule} (start lr={args.lr})")

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(exist_ok=True)
    best_val_recon = float("inf")
    start_epoch = 1

    if args.resume:
        resume_path = ckpt_dir / "fsq_state.pt"
        if resume_path.exists():
            resume_state = torch.load(resume_path, map_location=device, weights_only=False)
            optimizer.load_state_dict(resume_state["optimizer"])
            scheduler.load_state_dict(resume_state["scheduler"])
            start_epoch = resume_state["epoch"] + 1
            best_val_recon = resume_state["best_val_recon"]
            print(f"Resumed optimizer/scheduler from epoch {resume_state['epoch']}")

    log_path = ckpt_dir / "fsq_log.csv"
    log_file = open(log_path, "a" if start_epoch > 1 else "w", newline="")
    log_writer = csv.writer(log_file)
    if start_epoch == 1:
        log_writer.writerow(["epoch", "train_recon", "train_slow", "train_uniform",
                             "val_recon", "codebook_usage", "codebook_ppl", "lr", "time_s"])

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            t0 = time.time()
            train_dataset.set_epoch(epoch)
            train_recon, train_slow, train_uniform, timing = train_epoch(
                model, train_loader, optimizer, args.alpha_slow, args.alpha_uniform,
                scaler=scaler, amp_dtype=amp_dtype,
                brightness_range=brightness, contrast_range=contrast)
            val_recon, codebook_usage, codebook_ppl = val_epoch(model, val_loader, amp_dtype=amp_dtype)
            scheduler.step()
            dt = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]

            t_total = sum(timing.values())
            pct = {k: (v / t_total * 100 if t_total > 0 else 0) for k, v in timing.items()}
            print(
                f"Epoch {epoch:3d}/{args.epochs} ({dt:.1f}s) | "
                f"Train: recon={train_recon:.4f} slow={train_slow:.4f} unif={train_uniform:.4f} | "
                f"Val: recon={val_recon:.4f} usage={codebook_usage:.1%} ppl={codebook_ppl:.1%} | LR: {lr:.1e}"
            )
            if device.type == "cuda":
                peak_gb = torch.cuda.max_memory_allocated() / 1e9
                reserved_gb = torch.cuda.max_memory_reserved() / 1e9
                torch.cuda.reset_peak_memory_stats()
            else:
                peak_gb = reserved_gb = 0.0
            print(
                f"  Timing ({t_total:.1f}s train): "
                f"data {timing['data']:.1f}s ({pct['data']:.0f}%) | "
                f"aug {timing['aug']:.1f}s ({pct['aug']:.0f}%) | "
                f"fwd {timing['fwd']:.1f}s ({pct['fwd']:.0f}%) | "
                f"bwd {timing['bwd']:.1f}s ({pct['bwd']:.0f}%) | "
                f"opt {timing['opt']:.1f}s ({pct['opt']:.0f}%) | "
                f"VRAM peak {peak_gb:.2f}GB (reserved {reserved_gb:.2f}GB)"
            )

            log_writer.writerow([
                epoch, f"{train_recon:.6f}", f"{train_slow:.6f}", f"{train_uniform:.6f}",
                f"{val_recon:.6f}", f"{codebook_usage:.4f}", f"{codebook_ppl:.4f}",
                f"{lr:.1e}", f"{dt:.1f}"
            ])
            log_file.flush()

            wandb_log({
                "epoch": epoch,
                "train/recon": train_recon, "train/slow": train_slow,
                "train/uniform": train_uniform,
                "val/recon": val_recon, "val/codebook_usage": codebook_usage,
                "val/codebook_ppl": codebook_ppl,
                "lr": lr,
                "time/data": timing["data"], "time/aug": timing["aug"],
                "time/fwd": timing["fwd"], "time/bwd": timing["bwd"],
                "time/opt": timing["opt"],
                "vram/peak_gb": peak_gb, "vram/reserved_gb": reserved_gb,
            })

            if val_recon < best_val_recon:
                best_val_recon = val_recon
                # Strip _orig_mod. prefix from torch.compile for portable checkpoints
                clean_state = {k.removeprefix("_orig_mod."): v
                               for k, v in model.state_dict().items()}
                torch.save(clean_state, ckpt_dir / "fsq_best.pt")

            # Save full state for resume
            torch.save({
                "epoch": epoch,
                "model": {k.removeprefix("_orig_mod."): v
                          for k, v in model.state_dict().items()},
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_recon": best_val_recon,
            }, ckpt_dir / "fsq_state.pt")
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
