"""Train the Transformer world model on tokenized episode data.

Usage:
    python scripts/tokenize_episodes.py --model fsq   # must run first
    python scripts/train_transformer.py
    python scripts/train_transformer.py --context-frames 4 --epochs 400
"""

import argparse
import csv
import os
import re
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepdash.wandb_utils import wandb_init, wandb_log, wandb_finish

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepdash.world_model import WorldModel


def _unwrap(model):
    """Access underlying model whether torch.compiled or not."""
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def build_structured_smooth_targets(levels, full_vocab_size, sigma=1.0, smoothing=0.1):
    """Precompute FSQ-structured soft target distributions.

    Visual tokens (0..prod(levels)-1): Gaussian kernel over squared FSQ
    coordinate distance. Status tokens (ALIVE, DEATH): hard targets.

    Args:
        levels: FSQ quantization levels, e.g. [8, 5, 5, 5].
        full_vocab_size: Total vocab including status tokens (1002).
        sigma: Gaussian kernel width. Controls how fast tolerance decays.
        smoothing: Total probability mass redistributed from correct token.

    Returns:
        soft_targets: (full_vocab_size, full_vocab_size) float tensor.
            Row i is the target distribution when the correct token is i.
    """
    import math
    vocab_size = math.prod(levels)
    n_dims = len(levels)

    # Decompose all visual token indices to FSQ coordinates
    divisors = []
    acc = 1
    for L in reversed(levels):
        divisors.append(acc)
        acc *= L
    divisors.reverse()

    coords = torch.zeros(vocab_size, n_dims)
    for idx in range(vocab_size):
        remainder = idx
        for d in range(n_dims):
            coords[idx, d] = remainder // divisors[d]
            remainder = remainder % divisors[d]

    # Pairwise squared Euclidean distance in FSQ coordinate space
    diff = coords.unsqueeze(1) - coords.unsqueeze(0)  # (V, V, D)
    sq_dist = (diff ** 2).sum(dim=-1)  # (V, V)

    # Gaussian kernel weights (zero on diagonal — handled separately)
    weights = torch.exp(-sq_dist / (2 * sigma ** 2))
    weights.fill_diagonal_(0)

    # Normalize each row to sum to 1, then scale by smoothing
    row_sums = weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    weights = weights / row_sums

    # Build full vocab soft targets (visual + status tokens)
    soft = torch.zeros(full_vocab_size, full_vocab_size)
    # Visual tokens: structured smoothing
    soft[:vocab_size, :vocab_size] = smoothing * weights
    diag_idx = torch.arange(vocab_size)
    soft[diag_idx, diag_idx] = 1 - smoothing
    # Status tokens: hard targets (no smoothing — death prediction must be exact)
    for i in range(vocab_size, full_vocab_size):
        soft[i, i] = 1.0

    return soft


def focal_cross_entropy(logits, targets, gamma=2.0, soft_target_matrix=None,
                        label_smoothing=0.0):
    """Focal loss with optional FSQ-structured soft targets.

    If soft_target_matrix is provided, uses structured label smoothing
    (Gaussian kernel over FSQ distance). Otherwise falls back to uniform
    label smoothing via F.cross_entropy.

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    """
    if soft_target_matrix is not None:
        # Look up soft target distributions for each target token
        soft_targets = soft_target_matrix[targets]  # (N, C)
        log_probs = F.log_softmax(logits, dim=-1)
        ce = -(soft_targets * log_probs).sum(dim=-1)  # (N,)
    else:
        ce = F.cross_entropy(logits, targets, reduction='none',
                             label_smoothing=label_smoothing)
    if gamma == 0:
        return ce.mean()
    pt = torch.exp(-ce)
    return (((1 - pt) ** gamma) * ce).mean()


def build_fsq_neighbor_table(levels):
    """Precompute valid FSQ neighbors for each token index.

    For each of the prod(levels) tokens, stores its valid neighbors
    (±1 in each FSQ dimension, boundary-checked). Returns a padded tensor
    and a count tensor for efficient random selection.

    Args:
        levels: list of ints, e.g. [8, 5, 5, 5].

    Returns:
        neighbor_table: (codebook_size, max_neighbors) long — padded neighbor indices.
        neighbor_counts: (codebook_size,) long — number of valid neighbors per token.
    """
    import math
    codebook_size = math.prod(levels)
    n_dims = len(levels)

    # Precompute divisors for mixed-radix decomposition
    divisors = []
    acc = 1
    for L in reversed(levels):
        divisors.append(acc)
        acc *= L
    divisors.reverse()

    all_neighbors = []
    for idx in range(codebook_size):
        # Decompose index to coordinates
        coords = []
        remainder = idx
        for d in range(n_dims):
            coords.append(remainder // divisors[d])
            remainder = remainder % divisors[d]

        # Generate valid neighbors
        neighbors = []
        for d in range(n_dims):
            for delta in [-1, +1]:
                new_val = coords[d] + delta
                if 0 <= new_val < levels[d]:
                    new_coords = list(coords)
                    new_coords[d] = new_val
                    new_idx = sum(c * div for c, div in zip(new_coords, divisors))
                    neighbors.append(new_idx)
        all_neighbors.append(neighbors)

    max_neighbors = max(len(n) for n in all_neighbors)
    neighbor_table = torch.zeros(codebook_size, max_neighbors, dtype=torch.long)
    neighbor_counts = torch.zeros(codebook_size, dtype=torch.long)

    for idx, neighbors in enumerate(all_neighbors):
        neighbor_counts[idx] = len(neighbors)
        for j, n in enumerate(neighbors):
            neighbor_table[idx, j] = n

    return neighbor_table, neighbor_counts


def apply_fsq_noise(tokens, neighbor_table, neighbor_counts, prob, device):
    """Replace tokens with random FSQ neighbors at given probability.

    Args:
        tokens: (*, T) long — token indices (visual tokens only).
        neighbor_table: (codebook_size, max_neighbors) long.
        neighbor_counts: (codebook_size,) long.
        prob: float — probability of replacing each token.
        device: torch device.

    Returns:
        Perturbed tokens, same shape.
    """
    mask = torch.rand(tokens.shape, device=device) < prob
    if not mask.any():
        return tokens

    flat = tokens[mask]  # (n_selected,)
    counts = neighbor_counts[flat]  # (n_selected,)
    # Random neighbor index for each selected token
    rand_idx = (torch.rand(flat.shape, device=device) * counts.float()).long()
    rand_idx = rand_idx.clamp(max=neighbor_table.shape[1] - 1)
    replacements = neighbor_table[flat, rand_idx]

    result = tokens.clone()
    result[mask] = replacements.to(device)
    return result


def train_epoch(model, loader, optimizer, scaler, cpc_weight, device,
                token_noise=0.0, fsq_noise=0.0, neighbor_table=None,
                neighbor_counts=None, label_smoothing=0.0, focal_gamma=2.0,
                soft_target_matrix=None):
    model.train()
    m = _unwrap(model)
    tpf = m.tokens_per_frame
    vocab = m.full_vocab_size
    vs = m.vocab_size
    total_loss, total_correct, total_tokens = 0, 0, 0
    total_cpc_loss = 0.0
    death_tp, death_fp, death_fn = 0, 0, 0

    use_amp = device.type == "cuda"

    for frame_tokens, actions in loader:
        frame_tokens = frame_tokens.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)

        target = frame_tokens[:, -1]  # (B, 65)

        # Noise augmentation on context visual tokens (not status tokens)
        if token_noise > 0 or fsq_noise > 0:
            ctx = frame_tokens[:, :-1].clone()
            visual = ctx[:, :, :tpf]
            # Random token noise: replace with any token uniformly
            if token_noise > 0:
                mask = torch.rand_like(visual, dtype=torch.float) < token_noise
                random_tokens = torch.randint(0, vs, visual.shape, device=device)
                visual = torch.where(mask, random_tokens, visual)
            # FSQ neighbor noise: replace with ±1 neighbor in one FSQ dim
            if fsq_noise > 0 and neighbor_table is not None:
                visual = apply_fsq_noise(
                    visual, neighbor_table, neighbor_counts, fsq_noise, device)
            ctx[:, :, :tpf] = visual
            frame_tokens = torch.cat([ctx, frame_tokens[:, -1:]], dim=1)

        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            logits, cpc_loss = model(frame_tokens, actions)
            token_loss = focal_cross_entropy(
                logits.reshape(-1, logits.size(-1)),  # (B*65, vocab)
                target.reshape(-1),                    # (B*65,)
                gamma=focal_gamma,
                soft_target_matrix=soft_target_matrix,
                label_smoothing=label_smoothing,
            )
            loss = token_loss + cpc_weight * cpc_loss

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        bs = frame_tokens.size(0)
        with torch.no_grad():
            visual_preds = logits[:, :tpf].argmax(dim=-1)
            visual_target = target[:, :tpf]
            total_correct += (visual_preds == visual_target).sum().item()
            total_tokens += bs * tpf
            total_loss += token_loss.item() * bs

            status_target = target[:, tpf]
            status_pred = logits[:, tpf].argmax(dim=-1)
            is_death = status_target == m.DEATH_TOKEN
            pred_death = status_pred == m.DEATH_TOKEN
            death_tp += (pred_death & is_death).sum().item()
            death_fp += (pred_death & ~is_death).sum().item()
            death_fn += (~pred_death & is_death).sum().item()
            total_cpc_loss += cpc_loss.item() * bs

    n_train = death_tp + death_fp + death_fn + 1e-8
    death_prec = death_tp / (death_tp + death_fp + 1e-8)
    death_rec = death_tp / (death_tp + death_fn + 1e-8)
    death_f1 = 2 * death_prec * death_rec / (death_prec + death_rec + 1e-8)
    n_samples = total_tokens // tpf
    return (total_loss / n_samples, total_correct / total_tokens,
            death_prec, death_rec, death_f1,
            total_cpc_loss / n_samples)


@torch.no_grad()
def val_epoch(model, loader, device, label_smoothing=0.0, focal_gamma=2.0,
              soft_target_matrix=None):
    model.eval()
    m = _unwrap(model)
    tpf = m.tokens_per_frame
    total_loss, total_correct, total_tokens = 0, 0, 0
    total_cpc_loss = 0.0
    death_tp, death_fp, death_fn = 0, 0, 0

    use_amp = device.type == "cuda"

    for frame_tokens, actions in loader:
        frame_tokens = frame_tokens.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)

        target = frame_tokens[:, -1]

        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            logits, cpc_loss = model(frame_tokens, actions)

        token_loss = focal_cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target.reshape(-1),
            gamma=focal_gamma,
            soft_target_matrix=soft_target_matrix,
            label_smoothing=label_smoothing,
        )

        bs = frame_tokens.size(0)
        total_loss += token_loss.item() * bs
        total_cpc_loss += cpc_loss.item() * bs

        visual_preds = logits[:, :tpf].argmax(dim=-1)
        visual_target = target[:, :tpf]
        total_correct += (visual_preds == visual_target).sum().item()
        total_tokens += bs * tpf

        status_target = target[:, tpf]
        status_pred = logits[:, tpf].argmax(dim=-1)
        is_death = status_target == m.DEATH_TOKEN
        pred_death = status_pred == m.DEATH_TOKEN
        death_tp += (pred_death & is_death).sum().item()
        death_fp += (pred_death & ~is_death).sum().item()
        death_fn += (~pred_death & is_death).sum().item()

    death_prec = death_tp / (death_tp + death_fp + 1e-8)
    death_rec = death_tp / (death_tp + death_fn + 1e-8)
    death_f1 = 2 * death_prec * death_rec / (death_prec + death_rec + 1e-8)
    n_samples = total_tokens // tpf
    return (total_loss / n_samples, total_correct / total_tokens,
            death_prec, death_rec, death_f1,
            total_cpc_loss / n_samples)


def main():
    parser = argparse.ArgumentParser(description="Train Transformer world model")
    parser.add_argument("--episodes-dir", default="data/death_episodes")
    parser.add_argument("--expert-episodes-dir", default="data/expert_episodes",
                        help="Directory with expert episodes (no death on last frame)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=4e-3)
    parser.add_argument("--lr-min", type=float, default=5e-4)
    parser.add_argument("--context-frames", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=1000,
                        help="Tokenizer vocabulary size (1000 for FSQ, 1024 for VQ-VAE)")
    parser.add_argument("--tokens-per-frame", type=int, default=64,
                        help="Tokens per frame (64 for 8x8 FSQ, 36 for 6x6 VQ-VAE)")
    parser.add_argument("--embed-dim", type=int, default=384)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--cpc-weight", type=float, default=1.0)
    parser.add_argument("--token-noise", type=float, default=0.05,
                        help="Random token replacement noise rate")
    parser.add_argument("--fsq-noise", type=float, default=0.05,
                        help="FSQ neighbor substitution noise rate")
    parser.add_argument("--fsq-levels", type=int, nargs="+", default=[8, 5, 5, 5],
                        help="FSQ quantization levels (must match tokenizer)")
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                        help="Label smoothing amount (used with --fsq-sigma for structured smoothing)")
    parser.add_argument("--fsq-sigma", type=float, default=1.0,
                        help="Gaussian kernel width for structured label smoothing (0 = uniform)")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Focal loss gamma (0 = standard CE)")
    parser.add_argument("--death-oversample", type=int, default=5,
                        help="Repeat death-frame samples this many times (1 = no oversampling)")
    parser.add_argument("--steps-per-epoch", type=int, default=0,
                        help="Cap training steps per epoch (0 = full dataset)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Handle SIGTERM (from SLURM timeout) gracefully
    def _sigterm_handler(sig, frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, _sigterm_handler)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Global episode-level split (shared across all models)
    from deepdash.data_split import get_val_episodes, is_val_episode
    val_set = get_val_episodes(args.episodes_dir, args.expert_episodes_dir)

    episodes_dir = Path(args.episodes_dir)
    all_episodes = sorted(ep for ep in episodes_dir.glob("*")
                          if (ep / "tokens.npy").exists())
    death_episodes = set(ep.name for ep in all_episodes)

    expert_dir = Path(args.expert_episodes_dir)
    if expert_dir.exists():
        expert_eps = sorted(ep for ep in expert_dir.glob("*")
                            if (ep / "tokens.npy").exists())
        all_episodes.extend(expert_eps)

    n_train = sum(1 for ep in all_episodes if not is_val_episode(ep.name, val_set))
    n_val = sum(1 for ep in all_episodes if is_val_episode(ep.name, val_set))
    print(f"Total tokenized episodes: {len(all_episodes)} ({n_train} train, {n_val} val)")

    K = args.context_frames
    TPF = args.tokens_per_frame
    train_frames, train_actions, train_weights = [], [], []
    val_frames, val_actions = [], []

    n_deaths = 0
    for ep in all_episodes:
        tokens = np.load(ep / "tokens.npy")  # (T, TPF)
        actions = np.load(ep / "actions.npy")  # (T,)
        T = len(tokens)
        if T < K + 1:
            continue

        is_val = is_val_episode(ep.name, val_set)
        f_list = val_frames if is_val else train_frames
        a_list = val_actions if is_val else train_actions

        for i in range(T - K):
            frame_window = tokens[i:i + K + 1].astype(np.int64)  # (K+1, TPF)
            action_window = actions[i:i + K].astype(np.int64)

            # Append status token to each frame
            # Target frame (index K in window): DEATH if last frame of a death episode
            status = np.full((K + 1, 1), 0, dtype=np.int64)
            is_death_episode = ep.name in death_episodes
            is_death_frame = is_death_episode and (i + K == T - 1)
            if is_death_frame:
                status[K] = 1  # will be mapped to DEATH_TOKEN
            n_deaths += int(is_death_frame)

            # Pack: (K+1, TPF+1) where last col is status
            frame_with_status = np.concatenate([frame_window, status], axis=1)
            f_list.append(frame_with_status)
            a_list.append(action_window)
            if not is_val:
                train_weights.append(
                    float(args.death_oversample) if is_death_frame else 1.0)

    # Create model first to get token indices
    model = WorldModel(
        vocab_size=args.vocab_size,
        n_actions=2,
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        context_frames=args.context_frames,
        dropout=args.dropout,
        tokens_per_frame=args.tokens_per_frame,
    ).to(device)

    # Pre-stack into contiguous tensors (avoids per-item numpy->torch overhead)
    print("Stacking into tensors...")
    train_frames_t = torch.from_numpy(np.stack(train_frames))
    train_actions_t = torch.from_numpy(np.stack(train_actions))
    val_frames_t = torch.from_numpy(np.stack(val_frames))
    val_actions_t = torch.from_numpy(np.stack(val_actions))
    del train_frames, train_actions
    del val_frames, val_actions

    # Remap status column upfront: 0 -> ALIVE_TOKEN, 1 -> DEATH_TOKEN
    for t in (train_frames_t, val_frames_t):
        status = t[:, :, -1]
        status[status == 0] = model.ALIVE_TOKEN
        status[status == 1] = model.DEATH_TOKEN

    train_dataset = TensorDataset(train_frames_t, train_actions_t)
    val_dataset = TensorDataset(val_frames_t, val_actions_t)
    print(f"Train samples: {len(train_dataset)} unique, Val samples: {len(val_dataset)}")
    print(f"Death frames: {n_deaths} unique, weighted {args.death_oversample}x via sampler")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    print(f"Context: {args.context_frames} frames, Sequence length: {model.seq_len}")
    print(f"Vocab: {args.vocab_size} visual + 2 status = {model.full_vocab_size}")

    # Cap samples per epoch if --steps-per-epoch is set
    if args.steps_per_epoch > 0:
        num_samples = args.steps_per_epoch * args.batch_size
    else:
        num_samples = int(sum(train_weights))
    train_sampler = WeightedRandomSampler(
        train_weights, num_samples=num_samples, replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              sampler=train_sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=0, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    scaler = torch.GradScaler(device.type)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(exist_ok=True)

    start_epoch = 1
    best_val_loss = float("inf")

    if args.resume:
        resume_path = ckpt_dir / "transformer_state.pt"
        if resume_path.exists():
            state = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            if "scaler" in state:
                scaler.load_state_dict(state["scaler"])
            start_epoch = state["epoch"] + 1
            best_val_loss = state["best_val_loss"]
            print(f"Resumed from epoch {state['epoch']} (best val loss: {best_val_loss:.4f})")
        else:
            print("No checkpoint found, starting fresh.")

    if start_epoch == 1:
        import json
        with open(ckpt_dir / "transformer_args.json", "w") as f:
            json.dump(vars(args), f, indent=2)

    wandb_init(project="deepdash", name=f"transformer-{args.embed_dim}d",
               config=vars(args))

    # torch.compile full model (static graph since masking was removed)
    if sys.platform != "win32":
        try:
            import torch._inductor.config as inductor_cfg
            inductor_cfg.compile_threads = min(os.cpu_count() or 1, 8)
            model = torch.compile(model)
            print(f"torch.compile enabled (full model, {inductor_cfg.compile_threads} compile threads)")
        except Exception as e:
            print(f"torch.compile not available, running eager: {e}")
    else:
        print("Skipping torch.compile (not supported on Windows)")

    # Build FSQ neighbor lookup table for structured noise
    if args.fsq_noise > 0:
        neighbor_table, neighbor_counts = build_fsq_neighbor_table(args.fsq_levels)
        neighbor_table = neighbor_table.to(device)
        neighbor_counts = neighbor_counts.to(device)
        print(f"FSQ neighbor noise: {args.fsq_noise:.0%} "
              f"(avg {neighbor_counts.float().mean():.1f} neighbors/token)")
    else:
        neighbor_table, neighbor_counts = None, None

    # Build structured label smoothing matrix
    if args.fsq_sigma > 0 and args.label_smoothing > 0:
        soft_target_matrix = build_structured_smooth_targets(
            args.fsq_levels, model.full_vocab_size,
            sigma=args.fsq_sigma, smoothing=args.label_smoothing,
        ).to(device)
        # Check how concentrated the smoothing mass is
        visual_row = soft_target_matrix[0, :model.vocab_size]
        top5 = visual_row.topk(6).values  # includes self
        print(f"Structured label smoothing: σ={args.fsq_sigma}, ε={args.label_smoothing}")
        print(f"  Top-6 target probs for token 0: {top5.tolist()}")
    else:
        soft_target_matrix = None

    warmup_epochs = 5
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-2, total_iters=warmup_epochs)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - warmup_epochs, eta_min=args.lr_min)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
        last_epoch=start_epoch - 2 if start_epoch > 1 else -1)

    log_path = ckpt_dir / "transformer_log.csv"
    log_header = ["epoch", "train_total", "train_loss", "train_acc",
                  "train_death_prec", "train_death_rec", "train_death_f1",
                  "train_cpc",
                  "val_total", "val_loss", "val_acc",
                  "val_death_prec", "val_death_rec", "val_death_f1",
                  "val_cpc",
                  "gap", "lr", "time_s"]

    # On resume: keep all rows up to start_epoch-1, then append
    if log_path.exists() and start_epoch > 1:
        with open(log_path) as f:
            rows = list(csv.reader(f))
        # Keep header + rows with epoch < start_epoch
        kept = [rows[0]] if rows else []
        for row in rows[1:]:
            try:
                if int(row[0]) < start_epoch:
                    kept.append(row)
            except (ValueError, IndexError):
                continue
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(kept)
        log_file = open(log_path, "a", newline="")
        print(f"Resuming log from epoch {start_epoch} "
              f"(kept {len(kept) - 1} previous rows)")
    else:
        log_file = open(log_path, "w", newline="")
        log_file.write(",".join(log_header) + "\n")

    log_writer = csv.writer(log_file)

    patience_counter = 0

    print("First epoch will be slower due to torch.compile tracing...")
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            t0 = time.time()
            train_loss, train_acc, train_d_prec, train_d_rec, train_d_f1, train_cpc = train_epoch(
                model, train_loader, optimizer, scaler,
                args.cpc_weight, device, token_noise=args.token_noise,
                fsq_noise=args.fsq_noise,
                neighbor_table=neighbor_table, neighbor_counts=neighbor_counts,
                label_smoothing=args.label_smoothing,
                focal_gamma=args.focal_gamma,
                soft_target_matrix=soft_target_matrix)
            val_loss, val_acc, val_d_prec, val_d_rec, val_d_f1, val_cpc = val_epoch(
                model, val_loader, device,
                label_smoothing=args.label_smoothing,
                focal_gamma=args.focal_gamma,
                soft_target_matrix=soft_target_matrix)
            scheduler.step()
            dt = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]

            cpc_w = args.cpc_weight
            train_total = train_loss + cpc_w * train_cpc
            val_total = val_loss + cpc_w * val_cpc
            gap = val_total - train_total

            print(
                f"Epoch {epoch:3d}/{args.epochs} ({dt:.1f}s) | "
                f"Train: total={train_total:.4f} loss={train_loss:.4f} acc={train_acc:.3f} "
                f"death[P={train_d_prec:.3f} R={train_d_rec:.3f} F1={train_d_f1:.3f}] "
                f"cpc={train_cpc:.3f} | "
                f"Val: total={val_total:.4f} loss={val_loss:.4f} acc={val_acc:.3f} "
                f"death[P={val_d_prec:.3f} R={val_d_rec:.3f} F1={val_d_f1:.3f}] "
                f"cpc={val_cpc:.3f} | "
                f"gap={gap:+.4f} | LR: {lr:.1e}"
            )

            log_writer.writerow([
                epoch, f"{train_total:.6f}", f"{train_loss:.6f}", f"{train_acc:.4f}",
                f"{train_d_prec:.4f}", f"{train_d_rec:.4f}", f"{train_d_f1:.4f}",
                f"{train_cpc:.4f}",
                f"{val_total:.6f}", f"{val_loss:.6f}", f"{val_acc:.4f}",
                f"{val_d_prec:.4f}", f"{val_d_rec:.4f}", f"{val_d_f1:.4f}",
                f"{val_cpc:.4f}",
                f"{gap:.4f}", f"{lr:.1e}", f"{dt:.1f}"
            ])
            log_file.flush()

            wandb_log({
                "epoch": epoch,
                "train/total": train_total, "train/loss": train_loss,
                "train/acc": train_acc, "train/death_f1": train_d_f1,
                "train/cpc": train_cpc,
                "val/total": val_total, "val/loss": val_loss,
                "val/acc": val_acc, "val/death_f1": val_d_f1,
                "val/cpc": val_cpc,
                "gap": gap, "lr": lr,
            })

            # Save full state
            torch.save({
                "epoch": epoch,
                "model": _unwrap(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_val_loss": best_val_loss,
            }, ckpt_dir / "transformer_state.pt")

            if val_total < best_val_loss:
                best_val_loss = val_total
                patience_counter = 0
                torch.save(_unwrap(model).state_dict(), ckpt_dir / "transformer_best.pt")
            else:
                patience_counter += 1
                if args.patience > 0 and patience_counter >= args.patience:
                    print(f"\nEarly stopping: val total loss did not improve for {args.patience} epochs.")
                    break

            # Defragment CUDA memory periodically
            if epoch % 2 == 0 and device.type == "cuda":
                torch.cuda.empty_cache()

    except (KeyboardInterrupt, SystemExit):
        print("\nInterrupted — saving checkpoint...")
        torch.save({
            "epoch": epoch,
            "model": _unwrap(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_val_loss": best_val_loss,
        }, ckpt_dir / "transformer_state.pt")

    log_file.close()
    wandb_finish()
    torch.save(_unwrap(model).state_dict(), ckpt_dir / "transformer_final.pt")
    print(f"\nTraining complete. Best val total loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to {ckpt_dir}/")


if __name__ == "__main__":
    main()
