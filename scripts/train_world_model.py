"""Train the world model (FSQ tokenizer + Transformer dynamics).

With --joint, the FSQ and Transformer train simultaneously from raw
frames; transformer gradient routes back into the encoder via STE (see
JointStep / WorldModel.fsq_grad_proj). Without --joint, the Transformer
trains on pre-tokenised episode data (tokens.npy).

Usage:
    python scripts/train_world_model.py --config configs/e6.7-recon-cauchysls.yaml
"""

import argparse
import csv
import math
import os
import re
import signal
import sys
import time
from pathlib import Path

# Persistent Inductor cache so compiled-kernel sources survive across
# SLURM jobs (cluster wipes /tmp between jobs).
os.environ.setdefault(
    "TORCHINDUCTOR_CACHE_DIR",
    str(Path.home() / ".cache" / "torchinductor"),
)
# Expandable CUDA allocator segments: reduces fragmentation between the
# peak activation burst and later allocations. Helpful on Linux/A100; on
# Windows the allocator emits "expandable_segments not supported on this
# platform" and ignores it. Harmless either way.
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True",
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepdash.wandb_utils import wandb_init, wandb_log, wandb_finish, wandb_run_id
from deepdash.world_model import WorldModel

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler


def _unwrap(model):
    """Access underlying model whether torch.compiled or not."""
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def build_optimizer(transformer, args, fsq_encoder=None, fsq_lr=None,
                    single_group=False):
    """AdamW optimizer. Three shapes:
      - transformer-only (fsq_encoder=None): one group.
      - joint + single_group=True: one group over all params, uses args.lr /
        args.weight_decay. LeWM-style recipe.
      - joint + single_group=False (default): two groups (transformer at
        args.lr/wd, fsq_encoder at fsq_lr/wd=0) sharing the LambdaLR factor.
    """
    if fsq_encoder is not None and single_group:
        return torch.optim.AdamW([{
            "params": list(transformer.parameters())
                      + list(fsq_encoder.parameters()),
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "name": "all",
        }])
    groups = [{
        "params": list(transformer.parameters()),
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "name": "transformer",
    }]
    if fsq_encoder is not None:
        if fsq_lr is None:
            raise ValueError("fsq_lr is required when fsq_encoder is provided")
        groups.append({
            "params": list(fsq_encoder.parameters()),
            "lr": fsq_lr,
            "weight_decay": 0.0,
            "name": "fsq_encoder",
        })
    return torch.optim.AdamW(groups)


def build_structured_smooth_targets(levels, full_vocab_size, sigma=1.0, smoothing=0.1,
                                    dim_weights=None, kernel="gaussian"):
    """Precompute FSQ-structured soft target distributions.

    Visual tokens (0..prod(levels)-1): one of four kernels over weighted FSQ
    coordinate distance. Status tokens (ALIVE, DEATH): hard targets.

    Args:
        levels: FSQ quantization levels, e.g. [8, 5, 5, 5].
        full_vocab_size: Total vocab including status tokens.
        sigma: Kernel bandwidth. For kernel='aniso_gaussian', a list/tuple of
            length len(levels) with per-dim bandwidths. Otherwise a scalar.
        smoothing: Total probability mass redistributed from correct token.
        dim_weights: Per-dimension distance weights from sensitivity analysis.
            Higher weight = more sensitive dim = neighbors further apart.
            If None, all dimensions weighted equally.
        kernel: One of 'gaussian', 'laplace', 'cauchy', 'aniso_gaussian'.

    Returns:
        soft_targets: (full_vocab_size, full_vocab_size) float tensor.
            Row i is the target distribution when the correct token is i.
    """
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

    # Pairwise weighted distance components in FSQ coordinate space
    diff = coords.unsqueeze(1) - coords.unsqueeze(0)  # (V, V, D)
    if dim_weights is not None:
        w = torch.tensor(dim_weights, dtype=torch.float32)
        diff = diff * w.view(1, 1, -1)

    if kernel == "gaussian":
        sq_dist = (diff ** 2).sum(dim=-1)
        weights = torch.exp(-sq_dist / (2.0 * float(sigma) ** 2))
    elif kernel == "laplace":
        abs_dist = diff.abs().sum(dim=-1)
        weights = torch.exp(-abs_dist / float(sigma))
    elif kernel == "cauchy":
        sq_dist = (diff ** 2).sum(dim=-1)
        weights = 1.0 / (1.0 + sq_dist / (float(sigma) ** 2))
    elif kernel == "aniso_gaussian":
        sigma_d = torch.tensor(sigma, dtype=torch.float32)
        if sigma_d.numel() != n_dims:
            raise ValueError(
                f"aniso_gaussian needs sigma of length {n_dims}, got {sigma_d.numel()}")
        per = (diff / sigma_d.view(1, 1, -1)) ** 2
        weights = torch.exp(-per.sum(dim=-1) / 2.0)
    else:
        raise ValueError(f"unknown kernel: {kernel}")
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
    # Status tokens: hard targets (no smoothing - death prediction must be exact)
    for i in range(vocab_size, full_vocab_size):
        soft[i, i] = 1.0

    return soft


def focal_cross_entropy(logits, targets, gamma=2.0, soft_target_matrix=None,
                        label_smoothing=0.0, soft_targets=None):
    """Focal loss with optional FSQ-structured soft targets.

    Target distribution source priority (first non-None wins):
      1. `soft_targets` (N, C) - precomputed per-row distribution. Used
         by the learnable-γ path in JointStep where the target kernel is
         rebuilt per batch from γ.
      2. `soft_target_matrix` (C, C) - static path: lookup by target index.
      3. Fall back to standard F.cross_entropy with uniform smoothing.

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    """
    if soft_targets is not None:
        log_probs = F.log_softmax(logits, dim=-1)
        ce = -(soft_targets * log_probs).sum(dim=-1)  # (N,)
    elif soft_target_matrix is not None:
        soft_tgt = soft_target_matrix[targets]  # (N, C)
        log_probs = F.log_softmax(logits, dim=-1)
        ce = -(soft_tgt * log_probs).sum(dim=-1)
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
        neighbor_table: (codebook_size, max_neighbors) long - padded neighbor indices.
        neighbor_counts: (codebook_size,) long - number of valid neighbors per token.
    """
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

    Branch-free rewrite so torch.compile + CUDA-graph reduce-overhead can
    capture the joint-training step as a single graph. At tiny prob the
    extra lookup work is negligible; the eliminated ``mask.any()`` /
    boolean indexing used to trigger graph breaks and dynamic shapes.

    Args:
        tokens: (*, T) long - token indices (visual tokens only).
        neighbor_table: (codebook_size, max_neighbors) long.
        neighbor_counts: (codebook_size,) long.
        prob: float - probability of replacing each token.
        device: torch device.

    Returns:
        Perturbed tokens, same shape.
    """
    apply_mask = torch.rand(tokens.shape, device=device) < prob
    counts = neighbor_counts[tokens]  # (*, T)
    rand_idx = (torch.rand(tokens.shape, device=device) * counts.float()).long()
    rand_idx = rand_idx.clamp(max=neighbor_table.shape[1] - 1)
    replacements = neighbor_table[tokens, rand_idx]  # (*, T)
    return torch.where(apply_mask, replacements, tokens)


# -------------------------------------------------------------------------
# Learnable-γ SLS helpers: FSQ coord table + per-batch soft-target builder
# -------------------------------------------------------------------------

def _fsq_coords(levels):
    """Build the (vocab_size, D) table of FSQ coordinates.

    Matches the decomposition used in `build_structured_smooth_targets`
    so the learnable-γ path produces identical outputs to the fixed
    precomputed path when γ = ln(dim_weights).
    """
    vocab_size = math.prod(levels)
    n_dims = len(levels)
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
    return coords


def _build_soft_targets(target_indices, coords, gamma, sigma, smoothing,
                        full_vocab_size, vocab_size):
    """On-the-fly kernel-smoothed target distribution with learnable γ.

    Same semantics as the precomputed `build_structured_smooth_targets`
    but recomputed per forward pass so γ receives gradient. Visual
    target rows (< vocab_size) get the full SLS kernel over FSQ
    coordinates; status rows (>= vocab_size) get hard one-hot (never
    smoothed - death prediction must be exact, status tokens are exact).

    Args:
        target_indices: (N,) long - flattened target token indices.
        coords: (vocab_size, D) buffer - precomputed FSQ coords.
        gamma: (D,) nn.Parameter - per-dim log-precision; w = exp(γ).
        sigma: float - kernel bandwidth.
        smoothing: float - total mass redistributed to neighbours for
            visual targets (default 0.1).
        full_vocab_size: int - visual codes + status tokens.
        vocab_size: int - visual codes only.

    Returns:
        soft: (N, full_vocab_size) float - per-target distribution,
            rows sum to 1.
    """
    N = target_indices.shape[0]
    device = target_indices.device
    dtype = gamma.dtype
    w = gamma.exp()  # (D,)
    # Clamped indices for coord lookup; bogus values on status rows will
    # be overridden by the is_visual mask below.
    safe_idx = target_indices.clamp(max=vocab_size - 1)
    target_coords = coords[safe_idx]  # (N, D)
    diff = coords.unsqueeze(0) - target_coords.unsqueeze(1)  # (N, V, D)
    weighted_sq = ((diff * w.view(1, 1, -1)) ** 2).sum(dim=-1)
    kernel = torch.exp(-weighted_sq / (2.0 * sigma * sigma))
    # Zero the self position via multiplicative mask (avoids in-place
    # op on `kernel` which is downstream of an autograd-tracked Exp).
    self_mask = torch.zeros(N, vocab_size, device=device, dtype=dtype)
    self_mask.scatter_(1, safe_idx.unsqueeze(1), 1.0)
    kernel = kernel * (1.0 - self_mask)
    row_sums = kernel.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    neighbours = smoothing * kernel / row_sums  # (N, V)

    # Mask: 1.0 for visual targets, 0.0 for status targets
    is_visual = (target_indices < vocab_size).to(dtype).unsqueeze(1)  # (N, 1)
    visual_part = neighbours * is_visual  # (N, vocab_size), zeros on status rows
    status_part = torch.zeros(N, full_vocab_size - vocab_size,
                              device=device, dtype=dtype)
    soft = torch.cat([visual_part, status_part], dim=1)  # (N, full_vocab_size)

    # Place target mass via out-of-place scatter (preserves autograd).
    # Visual targets: 1 - smoothing (complement to the ε spread in
    # neighbours). Status targets: 1.0 (hard one-hot).
    target_mass = 1.0 - smoothing * is_visual.squeeze(1)  # (N,)
    soft = soft.scatter(1, target_indices.unsqueeze(1),
                         target_mass.unsqueeze(1))
    return soft


def train_epoch(model, loader, optimizer, scaler, device,
                token_noise=0.0, fsq_noise=0.0, neighbor_table=None,
                neighbor_counts=None, label_smoothing=0.0, focal_gamma=2.0,
                soft_target_matrix=None, amp_dtype=torch.bfloat16,
                grad_clip=1.0):
    model.train()
    m = _unwrap(model)
    tpf = m.tokens_per_frame
    vocab = m.full_vocab_size
    vs = m.vocab_size
    total_loss, total_correct, total_tokens = 0, 0, 0
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

        with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(frame_tokens, actions)
            token_loss = focal_cross_entropy(
                logits.reshape(-1, logits.size(-1)),  # (B*65, vocab)
                target.reshape(-1),                    # (B*65,)
                gamma=focal_gamma,
                soft_target_matrix=soft_target_matrix,
                label_smoothing=label_smoothing,
            )
            loss = token_loss

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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

    n_train = death_tp + death_fp + death_fn + 1e-8
    death_prec = death_tp / (death_tp + death_fp + 1e-8)
    death_rec = death_tp / (death_tp + death_fn + 1e-8)
    death_f1 = 2 * death_prec * death_rec / (death_prec + death_rec + 1e-8)
    n_samples = total_tokens // tpf
    return (total_loss / n_samples, total_correct / total_tokens,
            death_prec, death_rec, death_f1)


@torch.no_grad()
def val_epoch(model, loader, device, label_smoothing=0.0, focal_gamma=2.0,
              soft_target_matrix=None, amp_dtype=torch.bfloat16):
    model.eval()
    m = _unwrap(model)
    tpf = m.tokens_per_frame
    total_loss, total_correct, total_tokens = 0, 0, 0
    death_tp, death_fp, death_fn = 0, 0, 0

    use_amp = device.type == "cuda"

    for frame_tokens, actions in loader:
        frame_tokens = frame_tokens.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)

        target = frame_tokens[:, -1]

        with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(frame_tokens, actions)

            token_loss = focal_cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1),
                gamma=focal_gamma,
                soft_target_matrix=soft_target_matrix,
                label_smoothing=label_smoothing,
            )

        bs = frame_tokens.size(0)
        total_loss += token_loss.item() * bs

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
            death_prec, death_rec, death_f1)


# -------------------------------------------------------------------------
# Joint training helpers (unfreeze FSQ, train simultaneously)
# -------------------------------------------------------------------------

def _encode_joint(fsq, raw_frames, K, tpf, D, fsq_dim_side=8, run_decoder=True):
    """Run raw frame windows through the FSQ encoder and produce everything
    the transformer + aux losses need.

    Args:
        fsq: FSQVAE module (in the same train/eval mode as the outer loop).
        raw_frames: (B, K+1, 64, 64) uint8 - raw frame windows.
        K: context_frames.
        tpf: tokens_per_frame (== fsq_dim_side ** 2).
        D: FSQ latent dim (= len(levels)).
        fsq_dim_side: spatial side of the encoder output (8 for 64x64 -> 8x8).
        run_decoder: if False, skip the decoder forward and return recon_all=None.
            Used when the decoder is not part of training.

    Returns:
        z_e_all:   (B, K+1, D, side, side)  - continuous encoder output.
        z_q_all:   (B, K+1, tpf, D)         - quantized codes (STE inside fsq).
        indices_all: (B, K+1, tpf)          - discrete code indices.
        recon_all: (B, K+1, 1, 64, 64) or None  - decoded frames (None if
                   run_decoder=False).
        frames_f:  (B, K+1, 1, 64, 64)      - normalized float [0, 1] frames
                                              (matches what decoder targets).
    """
    B = raw_frames.size(0)
    frames_f = raw_frames.float().mul_(1.0 / 255.0).unsqueeze(2)  # (B, K+1, 1, 64, 64)
    flat = frames_f.view(B * (K + 1), 1, 64, 64)
    z_e = fsq.encoder(flat)                         # (B*(K+1), D, side, side)
    z_q, indices = fsq.fsq(z_e)                     # (B*(K+1), D, side, side), (B*(K+1), side, side)
    side = fsq_dim_side
    z_e_all = z_e.view(B, K + 1, D, side, side)
    # (B*(K+1), D, side, side) -> (B, K+1, side*side, D) for transformer consumption
    z_q_all = z_q.view(B, K + 1, D, side * side).permute(0, 1, 3, 2).contiguous()
    indices_all = indices.view(B, K + 1, side * side)
    if run_decoder:
        recon = fsq.decoder(z_q)                    # (B*(K+1), 1, 64, 64)
        recon_all = recon.view(B, K + 1, 1, 64, 64)
    else:
        recon_all = None
    return z_e_all, z_q_all, indices_all, recon_all, frames_f


def _build_frame_tokens_joint(indices_all, is_death, alive_tok, death_tok):
    """Append the status column (last position of each frame block).

    Context frames are always ALIVE. The target frame's status is DEATH iff
    the window's target is a death frame in a death episode.
    """
    B, K1, TPF = indices_all.shape
    status = torch.full((B, K1, 1), alive_tok, dtype=torch.long,
                        device=indices_all.device)
    # Target frame is index K (last of K+1)
    target_status = torch.where(is_death, death_tok * torch.ones_like(is_death,
                                                                      dtype=torch.long),
                                 alive_tok * torch.ones_like(is_death,
                                                             dtype=torch.long))
    status[:, -1, 0] = target_status
    return torch.cat([indices_all, status], dim=-1)  # (B, K+1, TPF+1)


class JointStep(nn.Module):
    """Wraps FSQ encoder/decoder + WorldModel + loss computation in a
    single nn.Module so torch.compile(mode="reduce-overhead") captures
    one CUDA graph spanning encode -> STE -> transformer -> losses.

    Forward returns (loss, metrics_dict) where loss is the scalar to
    backprop and metrics_dict holds per-batch statistics as 0-dim
    tensors. Metrics are summed across batches by the caller with .item()
    outside the compiled boundary (.item() inside would force sync).

    Noise is generated inside the forward via torch.rand (compile-safe)
    and applied via torch.where (also safe). `apply_fsq_noise` is
    branch-free, so the whole path is capturable.

    Construction stores fixed hparams (alpha_uniform, ...) so the compiled
    graph specialises on them. Changing any of these across calls triggers
    recompilation, which is fine - they are constants.
    """

    def __init__(self, fsq, wm, *, alpha_uniform,
                 label_smoothing, focal_gamma, token_noise, fsq_noise,
                 shift_max=0, shift_max_x=0,
                 use_recon=True, recon_weight=1.0, recon_loss_type='mse',
                 neighbor_table=None, neighbor_counts=None,
                 soft_target_matrix=None,
                 fsq_levels=None, fsq_sigma=0.9):
        super().__init__()
        self.fsq = fsq
        self.wm = wm
        self.alpha_uniform = float(alpha_uniform)
        self.label_smoothing = float(label_smoothing)
        self.focal_gamma = float(focal_gamma)
        self.token_noise = float(token_noise)
        self.fsq_noise = float(fsq_noise)
        self.use_recon = bool(use_recon)
        self.recon_weight = float(recon_weight)
        self.recon_loss_type = str(recon_loss_type)
        self.fsq_sigma = float(fsq_sigma)
        if fsq_levels is not None:
            coords = _fsq_coords(fsq_levels)
            self.register_buffer("sls_coords", coords, persistent=False)
            self._has_coords = True
        else:
            self._has_coords = False
        self.shift_max = int(shift_max)
        self.shift_max_x = int(shift_max_x)
        self._has_fsq_levels = fsq_levels is not None
        if neighbor_table is not None:
            self.register_buffer("neighbor_table", neighbor_table, persistent=False)
            self.register_buffer("neighbor_counts", neighbor_counts, persistent=False)
            self._has_neighbors = True
        else:
            self._has_neighbors = False
        if soft_target_matrix is not None:
            self.register_buffer("soft_target_matrix", soft_target_matrix, persistent=False)
            self._has_soft = True
        else:
            self._has_soft = False

    def forward(self, raw_frames, actions, is_death):
        from deepdash.fsq import fsqvae_loss, fsq_marginal_uniform_reg

        m = _unwrap(self.wm)
        K = m.context_frames
        tpf = m.tokens_per_frame
        vs = m.vocab_size
        alive_tok = m.ALIVE_TOKEN
        death_tok = m.DEATH_TOKEN
        B = raw_frames.size(0)
        fsq_dim = self.wm.fsq_grad_proj.in_features

        if self.training and self.shift_max > 0:
            H = raw_frames.shape[-2]
            dy = torch.randint(-self.shift_max, self.shift_max + 1, (B,),
                               device=raw_frames.device)
            row_idx = (torch.arange(H, device=raw_frames.device)[None, :]
                       - dy[:, None]).clamp(0, H - 1)
            row_idx = row_idx[:, None, :, None].expand_as(raw_frames)
            raw_frames = torch.gather(raw_frames, -2, row_idx)

        if self.training and self.shift_max_x > 0:
            W = raw_frames.shape[-1]
            dx = torch.randint(-self.shift_max_x, self.shift_max_x + 1, (B,),
                               device=raw_frames.device)
            col_idx = (torch.arange(W, device=raw_frames.device)[None, :]
                       - dx[:, None]).clamp(0, W - 1)
            col_idx = col_idx[:, None, None, :].expand_as(raw_frames)
            raw_frames = torch.gather(raw_frames, -1, col_idx)

        z_e_all, z_q_all, indices_all, recon_all, frames_f = _encode_joint(
            self.fsq, raw_frames, K, tpf, fsq_dim,
            run_decoder=self.use_recon)
        # Do not detach z_q in the use_recon=False path: under
        # torch.compile AOT autograd, detaching the only grad-bearing
        # intermediate appears to eliminate the backward graph for the
        # entire function, yielding loss.grad_fn=None. Let z_q carry its
        # grad through; fsq encoder params accumulate grads but LR=0 on
        # the fsq_encoder optimizer group makes them inert. Decoder is
        # still skipped via run_decoder=self.use_recon above, so the net
        # memory footprint is below the pre-freeze phase.

        frame_last = frames_f[:, K - 1]
        frame_tgt = frames_f[:, K]

        if self.alpha_uniform > 0.0:
            B_, K1_, D_, H_, W_ = z_e_all.shape
            z_bounded_all = self.fsq.fsq.bound(
                z_e_all.reshape(B_ * K1_, D_, H_, W_)
            ).view(B_, K1_, D_, H_, W_)
            uniform_loss = fsq_marginal_uniform_reg(
                z_bounded_all, self.fsq.fsq.half_levels)
        else:
            uniform_loss = torch.zeros((), device=raw_frames.device,
                                       dtype=frames_f.dtype)

        if self.use_recon:
            recon_last = recon_all[:, K - 1]
            recon_tgt = recon_all[:, K]
            recon_loss = (fsqvae_loss(recon_last, frame_last, self.recon_loss_type)
                          + fsqvae_loss(recon_tgt, frame_tgt, self.recon_loss_type)) / 2
        else:
            recon_loss = torch.zeros((), device=raw_frames.device,
                                     dtype=frames_f.dtype)

        frame_tokens = _build_frame_tokens_joint(
            indices_all, is_death, alive_tok, death_tok)
        target = frame_tokens[:, -1]

        ctx = frame_tokens[:, :-1]
        visual = ctx[:, :, :tpf]
        if self.token_noise > 0.0:
            mask = torch.rand_like(visual, dtype=torch.float) < self.token_noise
            random_tokens = torch.randint(0, vs, visual.shape, device=visual.device)
            visual = torch.where(mask, random_tokens, visual)
        if self.fsq_noise > 0.0 and self._has_neighbors:
            visual = apply_fsq_noise(
                visual, self.neighbor_table, self.neighbor_counts,
                self.fsq_noise, visual.device)
        ctx = torch.cat([visual, ctx[:, :, tpf:]], dim=-1)
        frame_tokens = torch.cat([ctx, frame_tokens[:, -1:]], dim=1)

        z_q_ste_ctx = z_q_all[:, :K]
        wm_out = self.wm(
            frame_tokens, actions, z_q_ste_context=z_q_ste_ctx,
            return_dense_logits=True)
        if isinstance(wm_out, tuple):
            logits, dense_logits = wm_out
        else:
            logits = wm_out
            dense_logits = None

        target_block_flat_logits = logits.reshape(-1, logits.size(-1))
        target_block_flat_targets = target.reshape(-1)
        if dense_logits is not None:
            dense_vis_flat_logits = (
                dense_logits[:, :, :tpf, :]
                .reshape(-1, dense_logits.size(-1)))
            dense_vis_flat_targets = indices_all[:, 1:K + 1, :].reshape(-1)
            all_flat_logits = torch.cat(
                [dense_vis_flat_logits, target_block_flat_logits], dim=0)
            all_flat_targets = torch.cat(
                [dense_vis_flat_targets, target_block_flat_targets], dim=0)
        else:
            all_flat_logits = target_block_flat_logits
            all_flat_targets = target_block_flat_targets

        if self._has_coords and self.wm.sls_gamma is not None:
            soft_per_batch = _build_soft_targets(
                all_flat_targets, self.sls_coords, self.wm.sls_gamma,
                self.fsq_sigma, self.label_smoothing,
                all_flat_logits.size(-1), vs)
            token_loss = focal_cross_entropy(
                all_flat_logits, all_flat_targets,
                gamma=self.focal_gamma,
                soft_target_matrix=None,
                label_smoothing=self.label_smoothing,
                soft_targets=soft_per_batch,
            )
        else:
            token_loss = focal_cross_entropy(
                all_flat_logits, all_flat_targets,
                gamma=self.focal_gamma,
                soft_target_matrix=(self.soft_target_matrix if self._has_soft else None),
                label_smoothing=self.label_smoothing,
            )

        loss = (self.recon_weight * recon_loss
                + self.alpha_uniform * uniform_loss
                + token_loss)

        # Per-batch metric scalars (0-dim tensors). Caller does .item()
        # outside the compiled boundary.
        with torch.no_grad():
            visual_preds = logits[:, :tpf].argmax(dim=-1)
            visual_target = target[:, :tpf]
            correct = (visual_preds == visual_target).sum()
            n_tokens = torch.tensor(visual_target.numel(), device=loss.device)
            status_target = target[:, tpf]
            status_pred = logits[:, tpf].argmax(dim=-1)
            is_d = status_target == death_tok
            pred_d = status_pred == death_tok
            tp = (pred_d & is_d).sum()
            fp = (pred_d & ~is_d).sum()
            fn = (~pred_d & is_d).sum()
            # FSQ codebook usage histogram over all context + target frames
            # produced by the ONLINE encoder. scatter_add_ with static
            # output size (vs) is compile-friendly.
            flat_idx = indices_all.reshape(-1).long()
            code_counts = torch.zeros(vs, device=loss.device, dtype=torch.long)
            code_counts.scatter_add_(
                0, flat_idx,
                torch.ones_like(flat_idx, dtype=torch.long))

        metrics = {
            "token_loss": token_loss.detach(),
            "recon_loss": recon_loss.detach(),
            "uniform_loss": uniform_loss.detach(),
            "correct": correct,
            "n_tokens": n_tokens,
            "death_tp": tp,
            "death_fp": fp,
            "death_fn": fn,
            "code_counts": code_counts,
        }
        return loss, metrics


def _codebook_stats(code_counts):
    """Usage % and normalized perplexity from a (vocab_size,) long histogram.

    usage_pct: fraction of codes observed at least once, in percent.
    ppl_pct: exp(H) / vocab_size, in percent. 100% = uniform codebook usage,
    ~0% = fully collapsed. Cross-vocab-size comparable (unlike raw ppl).
    """
    vocab_size = int(code_counts.numel())
    total = int(code_counts.sum().item())
    if total == 0:
        return 0.0, 0.0
    n_used = int((code_counts > 0).sum().item())
    usage_pct = 100.0 * n_used / vocab_size
    p = code_counts.to(torch.float64) / float(total)
    p_nz = p[p > 0]
    entropy = float(-(p_nz * p_nz.log()).sum().item())
    ppl_pct = 100.0 * math.exp(entropy) / vocab_size
    return usage_pct, ppl_pct


def train_epoch_joint(joint_step, loader, optimizer, scaler, device,
                      amp_dtype=torch.bfloat16, grad_clip=1.0,
                      single_group=False, grad_skip_threshold=0.0):
    """Run one training epoch by calling the compiled JointStep per batch.

    The joint_step module (typically torch.compile-wrapped) captures
    FSQ encode + aux losses + transformer + token/cpc losses as a single
    graph. The training loop here only handles data transfer, the
    optimizer step, grad-norm diagnostics, and metric accumulation.

    Per-stage timings are accumulated into ``bottlenecks`` (keys: data,
    fwd, bwd, step, metrics). GPU-side stages use CUDA events batched to
    a single synchronize() at epoch end — they don't stall the pipeline.
    """
    joint_step.train()
    js = _unwrap(joint_step)
    tpf = _unwrap(js.wm).tokens_per_frame
    use_amp = device.type == "cuda"
    is_cuda = device.type == "cuda"

    total_token_loss = 0.0
    total_recon = 0.0
    total_unif = 0.0
    total_correct = 0
    total_tokens = 0
    tp = fp = fn = 0
    enc_grad_sq = tr_grad_sq = 0.0
    n_batches = 0
    n_skipped = 0
    code_counts_total = None

    fwd_events, bwd_events, step_events = [], [], []
    cpu_data_s = 0.0
    cpu_metrics_s = 0.0
    code_counts_gpu = None
    t_data_start = time.perf_counter()

    for raw_frames, actions, is_death in loader:
        raw_frames = raw_frames.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        is_death = is_death.to(device, non_blocking=True).bool()
        B = raw_frames.size(0)
        cpu_data_s += time.perf_counter() - t_data_start

        if is_cuda:
            fs = torch.cuda.Event(enable_timing=True)
            fe = torch.cuda.Event(enable_timing=True)
            fs.record()
        with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            loss, metrics = joint_step(raw_frames, actions, is_death)
        if is_cuda:
            fe.record()
            fwd_events.append((fs, fe))

            bs = torch.cuda.Event(enable_timing=True)
            be = torch.cuda.Event(enable_timing=True)
            bs.record()
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if single_group:
            # Global clip across all params. Caller sized grad_clip to
            # match observed encoder-grad scale under recon dominance
            # (transformer grads are far smaller and pass through freely).
            total_norm = torch.nn.utils.clip_grad_norm_(
                list(js.wm.parameters()) + list(js.fsq.parameters()),
                grad_clip)
            wm_norm = fsq_norm = total_norm
        else:
            # Per-group clip: each network gets its own grad_clip budget
            # independently. A global clip would let the encoder's large
            # recon-driven gradient saturate the whole budget and attenuate
            # the transformer's (already smaller) gradient by the encoder-
            # to-transformer ratio, severely slowing transformer learning.
            wm_norm = torch.nn.utils.clip_grad_norm_(
                list(js.wm.parameters()), grad_clip)
            fsq_norm = torch.nn.utils.clip_grad_norm_(
                list(js.fsq.parameters()), grad_clip)
        if is_cuda:
            be.record()
            bwd_events.append((bs, be))

            ss = torch.cuda.Event(enable_timing=True)
            se = torch.cuda.Event(enable_timing=True)
            ss.record()
        # Outlier-batch skip: if the pre-clip norm is far above steady state,
        # the update direction is likely anomalous (exploding gradient from a
        # rare batch). Clipping only bounds magnitude, not direction; a full
        # clip-magnitude step in a bad direction can still destabilize the
        # encoder on slim models. We zero gradients on skip rather than
        # branching on a host-synced bool: this keeps the decision on GPU
        # and folds into the per-batch metric .tolist() below (one sync
        # instead of two). AdamW m/v decay by beta on a zero-grad step
        # rather than staying frozen; for the rare-skip regime this is
        # numerically negligible.
        if grad_skip_threshold > 0:
            norm_for_skip = (total_norm if single_group
                             else torch.maximum(wm_norm, fsq_norm))
            skip_t = (norm_for_skip > grad_skip_threshold).to(loss.dtype)
            keep = 1.0 - skip_t
            for p in (list(js.wm.parameters()) + list(js.fsq.parameters())):
                if p.grad is not None:
                    p.grad.mul_(keep)
        else:
            skip_t = torch.zeros((), device=device, dtype=loss.dtype)
        scaler.step(optimizer)
        scaler.update()
        if is_cuda:
            se.record()
            step_events.append((ss, se))

        # GPU-side accumulator (copied out once at epoch end). Avoids a
        # per-batch D2H sync for code_counts.
        if code_counts_gpu is None:
            code_counts_gpu = torch.zeros_like(metrics["code_counts"])
        code_counts_gpu += metrics["code_counts"]

        t_m = time.perf_counter()
        # Stack every scalar we need off-GPU into one tensor, then one
        # .tolist() call = one CPU-GPU sync instead of 11. clip_grad_norm_
        # returns the pre-clip L2 norm; squared = sum of per-param squared
        # grads (replaces the 800+ per-parameter .item() loop that used
        # to live here).
        scalars = torch.stack([
            fsq_norm.detach().float(),
            wm_norm.detach().float(),
            metrics["token_loss"].detach().float(),
            metrics["recon_loss"].detach().float(),
            metrics["uniform_loss"].detach().float(),
            metrics["correct"].detach().float(),
            metrics["n_tokens"].detach().float(),
            metrics["death_tp"].detach().float(),
            metrics["death_fp"].detach().float(),
            metrics["death_fn"].detach().float(),
            skip_t.detach().float(),
        ]).tolist()
        (fsq_n, wm_n, tok_l, rec_l, unif_l,
         cor, nt, d_tp, d_fp, d_fn, skipped) = scalars
        n_skipped += int(skipped)
        enc_grad_sq += fsq_n ** 2
        tr_grad_sq += wm_n ** 2
        total_token_loss += tok_l * B
        total_recon += rec_l * B
        total_unif += unif_l * B
        total_correct += int(cor)
        total_tokens += int(nt)
        tp += int(d_tp)
        fp += int(d_fp)
        fn += int(d_fn)
        n_batches += 1
        cpu_metrics_s += time.perf_counter() - t_m
        t_data_start = time.perf_counter()

    if code_counts_gpu is not None:
        code_counts_total = code_counts_gpu.detach().cpu()

    if is_cuda:
        torch.cuda.synchronize()
        fwd_s = sum(s.elapsed_time(e) for s, e in fwd_events) / 1000.0
        bwd_s = sum(s.elapsed_time(e) for s, e in bwd_events) / 1000.0
        step_s = sum(s.elapsed_time(e) for s, e in step_events) / 1000.0
    else:
        fwd_s = bwd_s = step_s = 0.0
    bottlenecks = {
        "data": cpu_data_s,
        "fwd": fwd_s,
        "bwd": bwd_s,
        "step": step_s,
        "metrics": cpu_metrics_s,
    }

    n_samples = max(1, total_tokens // tpf)
    death_prec = tp / (tp + fp + 1e-8)
    death_rec = tp / (tp + fn + 1e-8)
    death_f1 = 2 * death_prec * death_rec / (death_prec + death_rec + 1e-8)
    code_usage_pct, code_ppl_pct = _codebook_stats(
        code_counts_total if code_counts_total is not None
        else torch.zeros(1, dtype=torch.long))
    return {
        "loss": total_token_loss / n_samples,
        "acc": total_correct / max(1, total_tokens),
        "death_prec": death_prec,
        "death_rec": death_rec,
        "death_f1": death_f1,
        "recon": total_recon / n_samples,
        "unif": total_unif / n_samples,
        "enc_grad_rms": (enc_grad_sq / max(1, n_batches)) ** 0.5,
        "tr_grad_rms": (tr_grad_sq / max(1, n_batches)) ** 0.5,
        "code_usage_pct": code_usage_pct,
        "code_ppl_pct": code_ppl_pct,
        "n_skipped": n_skipped,
        "n_batches": n_batches,
        "bottlenecks": bottlenecks,
    }


@torch.no_grad()
def val_epoch_joint(joint_step, loader, device, amp_dtype=torch.bfloat16):
    """Validation: same compiled JointStep, eval mode, no backward.

    The caller should pass a joint_step instance constructed with
    noise=0 (or call .eval() on one that has stochastic dropout in its
    modules). Noise in the train JointStep is Python-attr-gated, so the
    attribute check short-circuits when token_noise=0 / fsq_noise=0.
    """
    joint_step.eval()
    js = _unwrap(joint_step)
    tpf = _unwrap(js.wm).tokens_per_frame
    use_amp = device.type == "cuda"

    total_token_loss = 0.0
    total_recon = 0.0
    total_unif = 0.0
    total_correct = 0
    total_tokens = 0
    tp = fp = fn = 0
    code_counts_total = None

    code_counts_gpu = None
    for raw_frames, actions, is_death in loader:
        raw_frames = raw_frames.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        is_death = is_death.to(device, non_blocking=True).bool()
        B = raw_frames.size(0)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            _, metrics = joint_step(raw_frames, actions, is_death)
        if code_counts_gpu is None:
            code_counts_gpu = torch.zeros_like(metrics["code_counts"])
        code_counts_gpu += metrics["code_counts"]
        scalars = torch.stack([
            metrics["token_loss"].detach().float(),
            metrics["recon_loss"].detach().float(),
            metrics["uniform_loss"].detach().float(),
            metrics["correct"].detach().float(),
            metrics["n_tokens"].detach().float(),
            metrics["death_tp"].detach().float(),
            metrics["death_fp"].detach().float(),
            metrics["death_fn"].detach().float(),
        ]).tolist()
        (tok_l, rec_l, unif_l,
         cor, nt, d_tp, d_fp, d_fn) = scalars
        total_token_loss += tok_l * B
        total_recon += rec_l * B
        total_unif += unif_l * B
        total_correct += int(cor)
        total_tokens += int(nt)
        tp += int(d_tp)
        fp += int(d_fp)
        fn += int(d_fn)
    if code_counts_gpu is not None:
        code_counts_total = code_counts_gpu.detach().cpu()

    n_samples = max(1, total_tokens // tpf)
    death_prec = tp / (tp + fp + 1e-8)
    death_rec = tp / (tp + fn + 1e-8)
    death_f1 = 2 * death_prec * death_rec / (death_prec + death_rec + 1e-8)
    code_usage_pct, code_ppl_pct = _codebook_stats(
        code_counts_total if code_counts_total is not None
        else torch.zeros(1, dtype=torch.long))
    return {
        "loss": total_token_loss / n_samples,
        "acc": total_correct / max(1, total_tokens),
        "death_prec": death_prec,
        "death_rec": death_rec,
        "death_f1": death_f1,
        "recon": total_recon / n_samples,
        "unif": total_unif / n_samples,
        "code_usage_pct": code_usage_pct,
        "code_ppl_pct": code_ppl_pct,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Transformer world model")
    parser.add_argument("--episodes-dir", default="data/death_episodes")
    parser.add_argument("--expert-episodes-dir", default="data/expert_episodes",
                        help="Directory with expert episodes (no death on last frame)")
    parser.add_argument("--config", default=None, help="YAML config path (default: configs/e6.7-recon-cauchysls.yaml)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr-min", type=float, default=None)
    parser.add_argument("--context-frames", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--tokens-per-frame", type=int, default=None)
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--token-noise", type=float, default=None)
    parser.add_argument("--fsq-noise", type=float, default=None)
    parser.add_argument("--fsq-levels", type=int, nargs="+", default=None,
                        help="FSQ quantization levels (must match tokenizer)")
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--fsq-sigma", type=float, default=None,
                        help="Kernel bandwidth for structured label smoothing "
                             "(scalar; used for gaussian/laplace/cauchy; 0 = uniform)")
    parser.add_argument("--fsq-sigma-d", type=float, nargs="+", default=None,
                        help="Per-dimension bandwidths for kernel=aniso_gaussian "
                             "(must match len(fsq_levels))")
    parser.add_argument("--fsq-kernel", choices=["gaussian", "laplace", "cauchy",
                                                  "aniso_gaussian"],
                        default=None,
                        help="Kernel family for structured label smoothing "
                             "(default: gaussian)")
    parser.add_argument("--fsq-dim-weights", type=float, nargs="+", default=None,
                        help="Per-dimension distance weights for structured smoothing")
    parser.add_argument("--focal-gamma", type=float, default=None)
    parser.add_argument("--death-oversample", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    # Mixed precision + compile (mirrors train_fsq.py conventions).
    # A100 defaults: bfloat16 + reduce-overhead. RTX 2060 SUPER (Turing) has
    # no native bf16 - it emulates at ~12x slower. Use float16 + default
    # compile mode on Turing; see configs/*-local.yaml.
    parser.add_argument("--amp-dtype", choices=["bfloat16", "float16"],
                        default=None,
                        help="AMP dtype. Default bfloat16 (A100). Use float16 "
                             "on Turing (RTX 20xx) - no native bf16 there.")
    parser.add_argument("--compile-mode",
                        choices=["reduce-overhead", "default", "none"],
                        default=None,
                        help="torch.compile mode. Default reduce-overhead "
                             "(A100 + modern CUDA). Use 'default' on Turing "
                             "to avoid CUDA-graph issues, or 'none' to disable "
                             "compile entirely (eager).")
    parser.add_argument("--joint", action="store_true", default=None,
                        help="Joint FSQ+Transformer training (STE-routed).")
    parser.add_argument("--fsq-lr", type=float, default=None,
                        help="Peak LR for the FSQ encoder param group (joint mode).")
    parser.add_argument("--alpha-uniform", type=float, default=None,
                        help="Weight on CWU anti-collapse regularizer.")
    parser.add_argument("--recon-weight", type=float, default=None,
                        help="Scalar multiplier on FSQ recon loss in the "
                             "joint total. Default 1.0 (no scaling); "
                             "<1 lets the prediction loss dominate.")
    parser.add_argument("--recon-loss-type", type=str, default=None,
                        choices=["mse", "l1"],
                        help="FSQ recon loss form. mse (default) sums squared "
                             "errors; l1 sums absolute errors and is ~30x larger "
                             "in magnitude at convergence -- recalibrate "
                             "recon-weight when switching.")
    parser.add_argument("--grad-skip-threshold", type=float, default=None,
                        help="If pre-clip grad norm exceeds this, skip the "
                             "optimizer step for that batch. 0 or None "
                             "disables. Set to ~3-5x steady-state grad_rms "
                             "to kill outlier-batch cascades on slim WMs.")
    parser.add_argument("--grad-skip-warmup-epochs", type=int, default=None,
                        help="Skip threshold is ignored until this epoch. "
                             "Early training has naturally huge gradients "
                             "(random init, low-LR warmup) that would "
                             "otherwise trip the threshold and stall training. "
                             "Default 15 when grad_skip_threshold is set.")
    parser.add_argument("--shift-max", type=int, default=None,
                        help="Max per-batch vertical shift pixels. 0 disables.")
    parser.add_argument("--shift-max-x", type=int, default=None,
                        help="Max per-batch horizontal shift pixels. 0 disables.")
    parser.add_argument("--use-recon", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="Include FSQ recon loss in the joint training loss. "
                             "When off, train a post-hoc decoder separately.")
    parser.add_argument("--learnable-gamma", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="Gamma as nn.Parameter (target distribution rebuilt "
                             "per forward) rather than fixed dim_weights.")
    parser.add_argument("--single-group", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="Merge encoder + transformer into a single AdamW "
                             "param group with shared LR and weight_decay "
                             "(LeWM-style). Default: two groups.")
    parser.add_argument("--grad-clip", type=float, default=None,
                        help="Gradient clip max_norm. Global (single-group) or "
                             "per-group (two-group). Default 1.0.")
    args = parser.parse_args()

    from deepdash.config import apply_config
    apply_config(args, section="transformer")
    # Map model.levels -> fsq_levels if not set via CLI
    if args.fsq_levels is None:
        args.fsq_levels = getattr(args, "levels", [8, 5, 5, 5])

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

    joint = bool(getattr(args, "joint", False))
    source_file = "frames.npy" if joint else "tokens.npy"

    def _has_data(ep):
        """Skip episodes missing or with zero-byte actions.npy / source_file.
        Shift-augmented episodes use a symlinked actions.npy; if that symlink
        was stripped during a filesystem copy (e.g. robocopy HDD -> SSD
        without /SL), the file becomes 0 bytes and np.load raises EOFError.
        This guard makes the loader robust to such broken copies.
        """
        src = ep / source_file
        act = ep / "actions.npy"
        return (src.exists() and src.stat().st_size > 0
                and act.exists() and act.stat().st_size > 0)

    episodes_dir = Path(args.episodes_dir)
    all_episodes = sorted(ep for ep in episodes_dir.glob("*") if _has_data(ep))
    death_episodes = set(ep.name for ep in all_episodes)

    expert_dir = Path(args.expert_episodes_dir)
    if expert_dir.exists():
        expert_eps = sorted(ep for ep in expert_dir.glob("*") if _has_data(ep))
        all_episodes.extend(expert_eps)

    n_train = sum(1 for ep in all_episodes if not is_val_episode(ep.name, val_set))
    n_val = sum(1 for ep in all_episodes if is_val_episode(ep.name, val_set))
    label = "raw-frame" if joint else "tokenized"
    print(f"Total {label} episodes: {len(all_episodes)} ({n_train} train, {n_val} val)")

    K = args.context_frames
    TPF = args.tokens_per_frame

    n_deaths = 0
    if joint:
        # Raw-frame windows: (T, 64, 64) uint8 + (T,) actions per episode
        train_rframes, train_actions_l, train_death, train_weights = [], [], [], []
        val_rframes, val_actions_l, val_death = [], [], []
        for ep in all_episodes:
            frames_np = np.load(ep / "frames.npy")    # (T, 64, 64) uint8
            actions_np = np.load(ep / "actions.npy")  # (T,)
            T = len(frames_np)
            if T < K + 1:
                continue
            is_val = is_val_episode(ep.name, val_set)
            rf_list = val_rframes if is_val else train_rframes
            a_list = val_actions_l if is_val else train_actions_l
            d_list = val_death if is_val else train_death
            is_death_ep = ep.name in death_episodes
            for i in range(T - K):
                frame_window = frames_np[i:i + K + 1]            # (K+1, 64, 64)
                action_window = actions_np[i:i + K].astype(np.int64)
                is_death_frame = is_death_ep and (i + K == T - 1)
                rf_list.append(frame_window)
                a_list.append(action_window)
                d_list.append(bool(is_death_frame))
                n_deaths += int(is_death_frame)
                if not is_val:
                    train_weights.append(
                        float(args.death_oversample) if is_death_frame else 1.0)
    else:
        # Non-joint tokenized path
        train_frames, train_actions, train_weights = [], [], []
        val_frames, val_actions = [], []
        for ep in all_episodes:
            tokens = np.load(ep / "tokens.npy")
            actions = np.load(ep / "actions.npy")
            T = len(tokens)
            if T < K + 1:
                continue
            is_val = is_val_episode(ep.name, val_set)
            f_list = val_frames if is_val else train_frames
            a_list = val_actions if is_val else train_actions
            for i in range(T - K):
                frame_window = tokens[i:i + K + 1].astype(np.int64)
                action_window = actions[i:i + K].astype(np.int64)
                status = np.full((K + 1, 1), 0, dtype=np.int64)
                is_death_episode = ep.name in death_episodes
                is_death_frame = is_death_episode and (i + K == T - 1)
                if is_death_frame:
                    status[K] = 1  # remapped to DEATH_TOKEN below
                n_deaths += int(is_death_frame)
                frame_with_status = np.concatenate([frame_window, status], axis=1)
                f_list.append(frame_with_status)
                a_list.append(action_window)
                if not is_val:
                    train_weights.append(
                        float(args.death_oversample) if is_death_frame else 1.0)

    use_recon = bool(getattr(args, "use_recon", True) if getattr(args, "use_recon", None) is not None else True)
    learnable_gamma = bool(getattr(args, "learnable_gamma", False) if getattr(args, "learnable_gamma", None) is not None else False)

    # Capacity-ratio gamma init: w_d = D * L_d / sum(L), gamma_d = ln(w_d).
    # Data-independent, gives w=1 on uniform codebooks.
    sls_gamma_init = None
    if learnable_gamma and args.fsq_levels is not None:
        levels_t = torch.tensor(args.fsq_levels, dtype=torch.float32)
        w = levels_t * len(args.fsq_levels) / levels_t.sum()
        sls_gamma_init = torch.log(w)

    # Create model (joint mode wires fsq_dim so fsq_grad_proj is instantiated)
    fsq_dim = len(args.fsq_levels) if joint else None
    model = WorldModel(
        vocab_size=args.vocab_size,
        n_actions=2,
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        context_frames=args.context_frames,
        dropout=args.dropout,
        tokens_per_frame=args.tokens_per_frame,
        adaln=getattr(args, 'adaln', False),
        fsq_dim=fsq_dim,
        sls_gamma_init=sls_gamma_init,
    ).to(device)

    # Pre-stack into contiguous tensors (avoids per-item numpy->torch overhead)
    print("Stacking into tensors...")
    if joint:
        from deepdash.fsq import FSQVAE
        train_frames_t = torch.from_numpy(np.stack(train_rframes))   # uint8 (N, K+1, 64, 64)
        train_actions_t = torch.from_numpy(np.stack(train_actions_l))
        train_death_t = torch.from_numpy(np.array(train_death, dtype=np.bool_))
        val_frames_t = torch.from_numpy(np.stack(val_rframes))
        val_actions_t = torch.from_numpy(np.stack(val_actions_l))
        val_death_t = torch.from_numpy(np.array(val_death, dtype=np.bool_))
        del train_rframes, train_actions_l, train_death
        del val_rframes, val_actions_l, val_death
        train_dataset = TensorDataset(train_frames_t, train_actions_t, train_death_t)
        val_dataset = TensorDataset(val_frames_t, val_actions_t, val_death_t)
        # Fresh FSQ init (no warm-start from checkpoints_v5_fsq)
        fsq = FSQVAE(levels=args.fsq_levels).to(device)
    else:
        fsq = None
        train_frames_t = torch.from_numpy(np.stack(train_frames))
        train_actions_t = torch.from_numpy(np.stack(train_actions))
        val_frames_t = torch.from_numpy(np.stack(val_frames))
        val_actions_t = torch.from_numpy(np.stack(val_actions))
        del train_frames, train_actions
        del val_frames, val_actions
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

    # One weighted pass per epoch. With death_oversample=N, num_samples =
    # sum(weights) = N * n_deaths + n_non_deaths: each unit of weight is
    # drawn once in expectation, giving death frames N expected draws per
    # epoch. Previously --steps-per-epoch capped this; dropped because
    # on-the-fly shift augmentation makes the effective dataset much
    # smaller than the old pre-shifted episode ladder that justified a cap.
    num_samples = int(sum(train_weights))
    train_sampler = WeightedRandomSampler(
        train_weights, num_samples=num_samples, replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              sampler=train_sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)

    single_group = bool(getattr(args, "single_group", False) or False)
    _gc = getattr(args, "grad_clip", None)
    grad_clip = float(_gc) if _gc is not None else 1.0
    _gst = getattr(args, "grad_skip_threshold", None)
    grad_skip_threshold = float(_gst) if _gst is not None else 0.0
    _gswe = getattr(args, "grad_skip_warmup_epochs", None)
    grad_skip_warmup_epochs = int(
        _gswe if _gswe is not None
        else (15 if grad_skip_threshold > 0 else 0))
    if joint:
        fsq_lr_val = args.fsq_lr if args.fsq_lr is not None else 1e-3
        optimizer = build_optimizer(
            model, args, fsq_encoder=fsq, fsq_lr=fsq_lr_val,
            single_group=single_group)
    else:
        optimizer = build_optimizer(model, args)
    print(f"Optimizer groups: {len(optimizer.param_groups)} "
          f"(single_group={single_group}, grad_clip={grad_clip})")
    if grad_skip_threshold > 0:
        print(f"Grad-skip: threshold={grad_skip_threshold} "
              f"(active after epoch {grad_skip_warmup_epochs})")

    # Resolve AMP dtype and compile mode with A100 defaults; local configs
    # (RTX 2060 SUPER / Turing) override via --amp-dtype float16 and
    # --compile-mode default, since Turing emulates bf16 at ~12x slower
    # and reduce-overhead's CUDA graphs have extra pitfalls on smaller SMs.
    amp_dtype_name = getattr(args, "amp_dtype", None) or "bfloat16"
    amp_dtype = {"bfloat16": torch.bfloat16,
                 "float16": torch.float16}[amp_dtype_name]
    _cm = getattr(args, "compile_mode", None)
    compile_mode = _cm if _cm is not None else "reduce-overhead"
    # GradScaler is only needed for float16 (bfloat16 has enough range to
    # skip scaling).
    scaler_enabled = (amp_dtype == torch.float16) and device.type == "cuda"
    scaler = torch.GradScaler(device.type, enabled=scaler_enabled)
    print(f"AMP: {amp_dtype_name} | GradScaler: "
          f"{'on' if scaler_enabled else 'off'} | compile: {compile_mode}")

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(exist_ok=True)

    start_epoch = 1
    best_val_death_f1 = 0.0

    wandb_resume_id = None
    resume_state = None
    if args.resume:
        resume_path = ckpt_dir / "transformer_state.pt"
        if resume_path.exists():
            resume_state = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(resume_state["model"])
            if joint and "fsq" in resume_state:
                fsq.load_state_dict(resume_state["fsq"])
            optimizer.load_state_dict(resume_state["optimizer"])
            if "scaler" in resume_state:
                scaler.load_state_dict(resume_state["scaler"])
            start_epoch = resume_state["epoch"] + 1
            best_val_death_f1 = resume_state.get("best_val_death_f1", 0.0)
            wandb_resume_id = resume_state.get("wandb_run_id")
            print(f"Resumed from epoch {resume_state['epoch']} (best val death F1: {best_val_death_f1:.4f})")
        else:
            print("No checkpoint found, starting fresh.")

    if start_epoch == 1:
        import json
        with open(ckpt_dir / "transformer_args.json", "w") as f:
            json.dump(vars(args), f, indent=2)

    # Derive run name from checkpoint_dir slug (post-"checkpoints_" prefix).
    # This trains the whole world model (FSQ + Transformer in joint mode,
    # Transformer only in non-joint mode), so a single "wm-<experiment>" label
    # matches the script name and reflects all metrics logged.
    _ckpt_name = Path(args.checkpoint_dir).name
    _slug = re.sub(r"^checkpoints[_-]?", "", _ckpt_name) or _ckpt_name
    run_name = f"wm-{_slug}"
    wandb_init(project="deepdash", name=run_name,
               config=vars(args), resume_id=wandb_resume_id)

    # torch.compile the model. In non-joint mode we compile WorldModel alone.
    # In joint mode we defer until JointStep is built (below) and compile
    # the wrapped FSQ+Transformer+loss graph as one unit so CUDA graph
    # capture spans encode -> STE -> transformer -> losses.
    if joint:
        print("Joint mode: model compile deferred to JointStep below")
    elif compile_mode == "none":
        print("torch.compile disabled (--compile-mode none)")
    else:
        try:
            import torch._inductor.config as inductor_cfg
            # Windows: Inductor's parallel-compile SubprocPool uses
            # subprocess `pass_fds`, which raises on Windows. Fall back
            # to single-threaded compile there.
            inductor_cfg.compile_threads = (
                1 if sys.platform == "win32"
                else min(os.cpu_count() or 1, 8))
            model = torch.compile(model, mode=compile_mode, fullgraph=True)
            print(f"torch.compile enabled (mode={compile_mode}, "
                  f"{inductor_cfg.compile_threads} compile threads)")
        except Exception as e:
            print(f"torch.compile not available, running eager: {e}")

    # Build FSQ neighbor lookup table for structured noise
    if args.fsq_noise > 0:
        neighbor_table, neighbor_counts = build_fsq_neighbor_table(args.fsq_levels)
        neighbor_table = neighbor_table.to(device)
        neighbor_counts = neighbor_counts.to(device)
        print(f"FSQ neighbor noise: {args.fsq_noise:.0%} "
              f"(avg {neighbor_counts.float().mean():.1f} neighbors/token)")
    else:
        neighbor_table, neighbor_counts = None, None

    # Build structured label smoothing matrix.
    # Under learnable γ the target distribution is rebuilt per
    # forward pass inside JointStep, so precomputing the matrix is dead
    # work; skip it and log a different message so the run logs reflect
    # what's actually happening.
    kernel = args.fsq_kernel or "gaussian"
    if kernel == "aniso_gaussian":
        sigma_arg = args.fsq_sigma_d
        has_bandwidth = sigma_arg is not None and all(s > 0 for s in sigma_arg)
    else:
        sigma_arg = args.fsq_sigma
        has_bandwidth = sigma_arg is not None and sigma_arg > 0

    if learnable_gamma:
        soft_target_matrix = None
        if args.label_smoothing > 0 and sls_gamma_init is not None:
            levels_str = args.fsq_levels
            mean_levels = sum(args.fsq_levels) / len(args.fsq_levels)
            w_init = [round(v, 4) for v in
                      (sls_gamma_init.exp().tolist())]
            g_init = [round(v, 4) for v in sls_gamma_init.tolist()]
            print(f"Learnable-gamma SLS: kernel=gaussian, sigma={args.fsq_sigma}, "
                  f"epsilon={args.label_smoothing}")
            print(f"  gamma_init = ln(levels / mean(levels)) = "
                  f"ln({levels_str} / {mean_levels:.2f}) = {g_init}  "
                  f"(w_init = {w_init})")
            print(f"  Target distribution rebuilt per forward pass from "
                  f"self.wm.sls_gamma; per-epoch values logged to sls/gamma_* "
                  f"and CSV.")
        else:
            print("Learnable-gamma SLS enabled but label_smoothing=0 or "
                  "fsq_levels=None; CE will be hard one-hot.")
    elif has_bandwidth and args.label_smoothing > 0:
        soft_target_matrix = build_structured_smooth_targets(
            args.fsq_levels, model.full_vocab_size,
            sigma=sigma_arg, smoothing=args.label_smoothing,
            dim_weights=args.fsq_dim_weights,
            kernel=kernel,
        ).to(device)
        visual_row = soft_target_matrix[0, :model.vocab_size]
        top5 = visual_row.topk(6).values  # includes self
        w_str = f", dim_weights={args.fsq_dim_weights}" if args.fsq_dim_weights else ""
        sigma_str = (f"[{', '.join(f'{s:.3f}' for s in sigma_arg)}]"
                     if kernel == "aniso_gaussian" else f"{sigma_arg}")
        print(f"Structured label smoothing: kernel={kernel}, sigma={sigma_str}, "
              f"epsilon={args.label_smoothing}{w_str}")
        print(f"  Top-6 target probs for token 0: {top5.tolist()}")
    else:
        soft_target_matrix = None

    # Build and compile JointStep. Two instances share the same
    # underlying fsq + model parameters; only the noise settings differ
    # so each can be specialised by torch.compile.
    joint_step_train = None
    joint_step_val = None
    if joint:
        _sm = getattr(args, "shift_max", None)
        shift_max = int(_sm) if _sm is not None else 0
        _smx = getattr(args, "shift_max_x", None)
        shift_max_x = int(_smx) if _smx is not None else 0
        # Under learnable γ the soft_target_matrix is irrelevant (rebuilt
        # per forward). Pass None to save the buffer copy.
        soft_tm_for_joint = None if learnable_gamma else soft_target_matrix
        joint_kwargs_common = dict(
            fsq_levels=args.fsq_levels,
            fsq_sigma=float(args.fsq_sigma) if args.fsq_sigma else 0.9,
        )
        _rw = getattr(args, "recon_weight", None)
        recon_weight = float(_rw) if _rw is not None else 1.0
        _rlt = getattr(args, "recon_loss_type", None)
        recon_loss_type = str(_rlt) if _rlt is not None else "mse"
        joint_step_train = JointStep(
            fsq=fsq, wm=model,
            alpha_uniform=args.alpha_uniform,
            label_smoothing=args.label_smoothing,
            focal_gamma=args.focal_gamma,
            token_noise=args.token_noise, fsq_noise=args.fsq_noise,
            shift_max=shift_max, shift_max_x=shift_max_x,
            use_recon=use_recon,
            recon_weight=recon_weight, recon_loss_type=recon_loss_type,
            neighbor_table=neighbor_table, neighbor_counts=neighbor_counts,
            soft_target_matrix=soft_tm_for_joint,
            **joint_kwargs_common,
        ).to(device)
        joint_step_val = JointStep(
            fsq=fsq, wm=model,
            alpha_uniform=args.alpha_uniform,
            label_smoothing=args.label_smoothing,
            focal_gamma=args.focal_gamma,
            token_noise=0.0, fsq_noise=0.0,                # no noise at eval
            shift_max=0, shift_max_x=0,                    # no shift at eval
            use_recon=use_recon,
            recon_weight=recon_weight, recon_loss_type=recon_loss_type,
            neighbor_table=None, neighbor_counts=None,
            soft_target_matrix=soft_tm_for_joint,
            **joint_kwargs_common,
        ).to(device)

        if compile_mode == "none":
            print("JointStep compile disabled (--compile-mode none)")
        else:
            try:
                import torch._inductor.config as inductor_cfg
                # Windows: parallel compile SubprocPool breaks on pass_fds.
                inductor_cfg.compile_threads = (
                    1 if sys.platform == "win32"
                    else min(os.cpu_count() or 1, 8))
                joint_step_train = torch.compile(
                    joint_step_train, mode=compile_mode, fullgraph=True)
                joint_step_val = torch.compile(
                    joint_step_val, mode=compile_mode, fullgraph=True)
                print(f"JointStep compiled (mode={compile_mode}, "
                      f"{inductor_cfg.compile_threads} compile threads, train + val)")
            except Exception as e:
                print(f"JointStep compile failed, running eager: {e}")

    # Closed-form LR factor: linear warmup followed by cosine decay to
    # eta_min. Implemented as a single LambdaLR so resume is state-
    # independent: on load_state_dict the child scheduler state doesn't
    # fully round-trip across SequentialLR milestones, which previously
    # caused the warmup leg to re-trigger from the current LR on resume.
    warmup_epochs = 5
    start_factor = 1e-2
    cosine_length = max(1, args.epochs - warmup_epochs)
    final_ratio = args.lr_min / args.lr
    def lr_factor(epoch):
        if epoch < warmup_epochs:
            return start_factor + (1.0 - start_factor) * epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / cosine_length
        progress = min(max(progress, 0.0), 1.0)
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return final_ratio + (1.0 - final_ratio) * cos
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
    if resume_state is not None:
        if "scheduler" in resume_state:
            scheduler.load_state_dict(resume_state["scheduler"])
        else:
            # Back-compat: pre-fix checkpoint without scheduler state.
            # last_epoch semantics: optimizer.step() already ran
            # `start_epoch - 1` times, so set last_epoch and let LambdaLR
            # recompute the factor analytically from lr_factor().
            scheduler.last_epoch = start_epoch - 1
            scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]

    log_path = ckpt_dir / "transformer_log.csv"
    log_unif = joint and float(getattr(args, "alpha_uniform", 0.0) or 0.0) > 0
    log_recon = joint and use_recon
    log_header = ["epoch", "train_total", "train_loss", "train_acc",
                  "train_death_prec", "train_death_rec", "train_death_f1",
                  "val_total", "val_loss", "val_acc",
                  "val_death_prec", "val_death_rec", "val_death_f1",
                  "gap", "lr", "time_s"]
    if joint:
        log_header.append("fsq_train_total")
        if log_recon:
            log_header.append("train_recon")
        if log_unif:
            log_header.append("train_unif")
        log_header.append("fsq_val_total")
        if log_recon:
            log_header.append("val_recon")
        if log_unif:
            log_header.append("val_unif")
        log_header += ["fsq_gap",
                       "encoder_grad_rms", "transformer_grad_rms",
                       "fsq_lr",
                       "train_code_usage_pct", "train_code_ppl_pct",
                       "val_code_usage_pct", "val_code_ppl_pct"]
        if model.sls_gamma is not None:
            for i in range(model.sls_gamma.shape[0]):
                log_header.append(f"gamma_{i}")

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

    if compile_mode != "none":
        print("First epoch will be slower due to torch.compile tracing...")
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            t0 = time.time()
            if joint:
                effective_skip = (grad_skip_threshold
                                  if epoch > grad_skip_warmup_epochs
                                  else 0.0)
                train_metrics = train_epoch_joint(
                    joint_step_train, train_loader, optimizer, scaler, device,
                    amp_dtype=amp_dtype, grad_clip=grad_clip,
                    single_group=single_group,
                    grad_skip_threshold=effective_skip)
                val_metrics = val_epoch_joint(
                    joint_step_val, val_loader, device, amp_dtype=amp_dtype)
                train_loss = train_metrics["loss"]
                train_acc = train_metrics["acc"]
                train_d_prec = train_metrics["death_prec"]
                train_d_rec = train_metrics["death_rec"]
                train_d_f1 = train_metrics["death_f1"]
                val_loss = val_metrics["loss"]
                val_acc = val_metrics["acc"]
                val_d_prec = val_metrics["death_prec"]
                val_d_rec = val_metrics["death_rec"]
                val_d_f1 = val_metrics["death_f1"]
            else:
                train_loss, train_acc, train_d_prec, train_d_rec, train_d_f1 = train_epoch(
                    model, train_loader, optimizer, scaler,
                    device, token_noise=args.token_noise,
                    fsq_noise=args.fsq_noise,
                    neighbor_table=neighbor_table, neighbor_counts=neighbor_counts,
                    label_smoothing=args.label_smoothing,
                    focal_gamma=args.focal_gamma,
                    soft_target_matrix=soft_target_matrix,
                    amp_dtype=amp_dtype, grad_clip=grad_clip)
                val_loss, val_acc, val_d_prec, val_d_rec, val_d_f1 = val_epoch(
                    model, val_loader, device,
                    label_smoothing=args.label_smoothing,
                    focal_gamma=args.focal_gamma,
                    soft_target_matrix=soft_target_matrix,
                    amp_dtype=amp_dtype)
            scheduler.step()
            dt = time.time() - t0
            # In joint mode both param groups are driven by the same factor
            # but have different base_lrs; log both.
            lr = optimizer.param_groups[0]["lr"]
            fsq_lr_now = (optimizer.param_groups[1]["lr"]
                          if joint and len(optimizer.param_groups) > 1
                          else None)

            train_total = train_loss
            val_total = val_loss
            gap = val_total - train_total

            # Joint-mode FSQ-side totals mirror the transformer's per-
            # component aggregate so the wandb overview and console line
            # are symmetric.
            if joint:
                a_u = args.alpha_uniform
                fsq_train_total = (train_metrics["recon"]
                                   + a_u * train_metrics["unif"])
                fsq_val_total = (val_metrics["recon"]
                                 + a_u * val_metrics["unif"])
                fsq_gap = fsq_val_total - fsq_train_total

            print(
                f"Epoch {epoch:3d}/{args.epochs} ({dt:.1f}s) | "
                f"Train: total={train_total:.4f} loss={train_loss:.4f} acc={train_acc:.3f} "
                f"death[P={train_d_prec:.3f} R={train_d_rec:.3f} F1={train_d_f1:.3f}] "
                f"Val: total={val_total:.4f} loss={val_loss:.4f} acc={val_acc:.3f} "
                f"death[P={val_d_prec:.3f} R={val_d_rec:.3f} F1={val_d_f1:.3f}] "
                f"gap={gap:+.4f} | LR: {lr:.1e}"
            )
            if joint:
                train_recon_str = (f"recon={train_metrics['recon']:.4f} "
                                    if log_recon else "")
                val_recon_str = (f"recon={val_metrics['recon']:.4f} "
                                  if log_recon else "")
                train_unif_str = (f"unif={train_metrics['unif']:.4e} "
                                   if log_unif else "")
                val_unif_str = (f"unif={val_metrics['unif']:.4e} "
                                 if log_unif else "")
                skipped = train_metrics.get("n_skipped", 0)
                skip_str = (f" | skipped={skipped}/{train_metrics['n_batches']}"
                            if skipped > 0 else "")
                print(
                    f"  FSQ | "
                    f"Train: total={fsq_train_total:.4f} {train_recon_str}{train_unif_str}| "
                    f"Val: total={fsq_val_total:.4f} {val_recon_str}{val_unif_str}| "
                    f"gap={fsq_gap:+.4f} | "
                    f"grad_rms[enc={train_metrics['enc_grad_rms']:.2e} "
                    f"tr={train_metrics['tr_grad_rms']:.2e}]"
                    + skip_str
                    + (f" | LR: {fsq_lr_now:.1e}" if fsq_lr_now is not None else "")
                )
                print(
                    f"  Codebook | "
                    f"Train: usage={train_metrics['code_usage_pct']:.1f}% "
                    f"ppl={train_metrics['code_ppl_pct']:.1f}% | "
                    f"Val: usage={val_metrics['code_usage_pct']:.1f}% "
                    f"ppl={val_metrics['code_ppl_pct']:.1f}%"
                )
                b = train_metrics.get("bottlenecks")
                if b is not None:
                    tot = max(1e-6, sum(b.values()))
                    print(
                        f"  Bottleneck | "
                        f"data={b['data']:.1f}s ({100*b['data']/tot:.0f}%) "
                        f"fwd={b['fwd']:.1f}s ({100*b['fwd']/tot:.0f}%) "
                        f"bwd={b['bwd']:.1f}s ({100*b['bwd']/tot:.0f}%) "
                        f"step={b['step']:.1f}s ({100*b['step']/tot:.0f}%) "
                        f"metrics={b['metrics']:.1f}s ({100*b['metrics']/tot:.0f}%)"
                    )

            row = [
                epoch, f"{train_total:.6f}", f"{train_loss:.6f}", f"{train_acc:.4f}",
                f"{train_d_prec:.4f}", f"{train_d_rec:.4f}", f"{train_d_f1:.4f}",
                f"{val_total:.6f}", f"{val_loss:.6f}", f"{val_acc:.4f}",
                f"{val_d_prec:.4f}", f"{val_d_rec:.4f}", f"{val_d_f1:.4f}",
                f"{gap:.4f}", f"{lr:.1e}", f"{dt:.1f}",
            ]
            if joint:
                row.append(f"{fsq_train_total:.6f}")
                if log_recon:
                    row.append(f"{train_metrics['recon']:.6f}")
                if log_unif:
                    row.append(f"{train_metrics['unif']:.6e}")
                row.append(f"{fsq_val_total:.6f}")
                if log_recon:
                    row.append(f"{val_metrics['recon']:.6f}")
                if log_unif:
                    row.append(f"{val_metrics['unif']:.6e}")
                row += [
                    f"{fsq_gap:.4f}",
                    f"{train_metrics['enc_grad_rms']:.6e}",
                    f"{train_metrics['tr_grad_rms']:.6e}",
                    f"{fsq_lr_now:.1e}" if fsq_lr_now is not None else "",
                    f"{train_metrics['code_usage_pct']:.2f}",
                    f"{train_metrics['code_ppl_pct']:.2f}",
                    f"{val_metrics['code_usage_pct']:.2f}",
                    f"{val_metrics['code_ppl_pct']:.2f}",
                ]
                if model.sls_gamma is not None:
                    for v in model.sls_gamma.detach().cpu().tolist():
                        row.append(f"{v:.6f}")
            log_writer.writerow(row)
            log_file.flush()

            wandb_payload = {
                "epoch": epoch,
                "transformer/train/total": train_total,
                "transformer/train/loss": train_loss,
                "transformer/train/acc": train_acc,
                "transformer/train/death_f1": train_d_f1,
                "transformer/val/total": val_total,
                "transformer/val/loss": val_loss,
                "transformer/val/acc": val_acc,
                "transformer/val/death_f1": val_d_f1,
                "transformer/gap": gap,
                "transformer/lr": lr,
            }
            if joint:
                wandb_payload.update({
                    "fsq/train/total": fsq_train_total,
                    "fsq/train/grad_rms": train_metrics["enc_grad_rms"],
                    "fsq/val/total": fsq_val_total,
                    "fsq/gap": fsq_gap,
                    "fsq/lr": fsq_lr_now,
                    "fsq/train/code_usage_pct": train_metrics["code_usage_pct"],
                    "fsq/train/code_ppl_pct": train_metrics["code_ppl_pct"],
                    "fsq/val/code_usage_pct": val_metrics["code_usage_pct"],
                    "fsq/val/code_ppl_pct": val_metrics["code_ppl_pct"],
                    "transformer/train/grad_rms": train_metrics["tr_grad_rms"],
                })
                if log_recon:
                    wandb_payload["fsq/train/recon"] = train_metrics["recon"]
                    wandb_payload["fsq/val/recon"] = val_metrics["recon"]
                if log_unif:
                    wandb_payload["fsq/train/unif"] = train_metrics["unif"]
                    wandb_payload["fsq/val/unif"] = val_metrics["unif"]
                if model.sls_gamma is not None:
                    gamma_vals = model.sls_gamma.detach().cpu().tolist()
                    for i, v in enumerate(gamma_vals):
                        wandb_payload[f"sls/gamma_{i}"] = v
            wandb_log(wandb_payload)

            # Save full state
            state_dict = {
                "epoch": epoch,
                "model": _unwrap(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_val_death_f1": best_val_death_f1,
                "wandb_run_id": wandb_run_id(),
            }
            if joint:
                state_dict["fsq"] = fsq.state_dict()
            torch.save(state_dict, ckpt_dir / "transformer_state.pt")

            if val_d_f1 > best_val_death_f1:
                best_val_death_f1 = val_d_f1
                torch.save(_unwrap(model).state_dict(), ckpt_dir / "transformer_best.pt")
                if joint:
                    torch.save(fsq.state_dict(), ckpt_dir / "fsq_best.pt")

            # Defragment CUDA memory periodically
            if epoch % 2 == 0 and device.type == "cuda":
                torch.cuda.empty_cache()

    except (KeyboardInterrupt, SystemExit):
        print("\nInterrupted - saving checkpoint...")
        interrupt_state = {
            "epoch": epoch,
            "model": _unwrap(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_val_death_f1": best_val_death_f1,
            "wandb_run_id": wandb_run_id(),
        }
        if joint:
            interrupt_state["fsq"] = fsq.state_dict()
        torch.save(interrupt_state, ckpt_dir / "transformer_state.pt")

    log_file.close()
    wandb_finish()
    torch.save(_unwrap(model).state_dict(), ckpt_dir / "transformer_final.pt")
    if joint:
        torch.save(fsq.state_dict(), ckpt_dir / "fsq_final.pt")
    print(f"\nTraining complete. Best val death F1: {best_val_death_f1:.4f}")
    print(f"Checkpoints saved to {ckpt_dir}/")


if __name__ == "__main__":
    main()
