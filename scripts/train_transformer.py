"""Train the Transformer world model on tokenized episode data.

Usage:
    python scripts/tokenize_episodes.py --model fsq   # must run first
    python scripts/train_transformer.py
    python scripts/train_transformer.py --context-frames 4 --epochs 400
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


def build_optimizer(transformer, args, fsq_encoder=None, fsq_lr=None):
    """AdamW with one or two param groups.

    Single-group (default, matches V5 behavior byte-for-byte):
        build_optimizer(transformer, args) -> AdamW(transformer.params,
                                                    lr=args.lr,
                                                    weight_decay=args.weight_decay)

    Two-group (E6.1 joint training): pass an FSQ encoder module and a
    separate peak LR for it. The transformer group keeps args.weight_decay;
    the encoder group uses weight_decay=0.0 to match V5 FSQ (Adam, no WD).
    Both groups are driven by the same LambdaLR factor; different peak LRs
    come from each group's base_lr.
    """
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

    Branch-free rewrite so torch.compile + CUDA-graph reduce-overhead can
    capture the joint-training step as a single graph. At tiny prob the
    extra lookup work is negligible; the eliminated ``mask.any()`` /
    boolean indexing used to trigger graph breaks and dynamic shapes.

    Args:
        tokens: (*, T) long — token indices (visual tokens only).
        neighbor_table: (codebook_size, max_neighbors) long.
        neighbor_counts: (codebook_size,) long.
        prob: float — probability of replacing each token.
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


def train_epoch(model, loader, optimizer, scaler, cpc_weight, device,
                token_noise=0.0, fsq_noise=0.0, neighbor_table=None,
                neighbor_counts=None, label_smoothing=0.0, focal_gamma=2.0,
                soft_target_matrix=None, amp_dtype=torch.bfloat16):
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

        with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
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
              soft_target_matrix=None, amp_dtype=torch.bfloat16):
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

        with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
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


# -------------------------------------------------------------------------
# Joint training helpers (E6.1: unfreeze FSQ, train simultaneously)
# -------------------------------------------------------------------------

def _encode_joint(fsq, raw_frames, K, tpf, D, fsq_dim_side=8):
    """Run raw frame windows through the FSQ encoder and produce everything
    the transformer + aux losses need.

    Args:
        fsq: FSQVAE module (in the same train/eval mode as the outer loop).
        raw_frames: (B, K+1, 64, 64) uint8 — raw frame windows.
        K: context_frames.
        tpf: tokens_per_frame (== fsq_dim_side ** 2).
        D: FSQ latent dim (= len(levels)).
        fsq_dim_side: spatial side of the encoder output (8 for 64x64 -> 8x8).

    Returns:
        z_e_all:   (B, K+1, D, side, side)  — continuous encoder output.
        z_q_all:   (B, K+1, tpf, D)         — quantized codes (STE inside fsq).
        indices_all: (B, K+1, tpf)          — discrete code indices.
        recon_all: (B, K+1, 1, 64, 64)      — decoded frames (for recon loss).
        frames_f:  (B, K+1, 1, 64, 64)      — normalized float [0, 1] frames
                                              (matches what decoder targets).
    """
    B = raw_frames.size(0)
    frames_f = raw_frames.float().mul_(1.0 / 255.0).unsqueeze(2)  # (B, K+1, 1, 64, 64)
    flat = frames_f.view(B * (K + 1), 1, 64, 64)
    z_e = fsq.encoder(flat)                         # (B*(K+1), D, side, side)
    z_q, indices = fsq.fsq(z_e)                     # (B*(K+1), D, side, side), (B*(K+1), side, side)
    recon = fsq.decoder(z_q)                        # (B*(K+1), 1, 64, 64)
    side = fsq_dim_side
    z_e_all = z_e.view(B, K + 1, D, side, side)
    # (B*(K+1), D, side, side) -> (B, K+1, side*side, D) for transformer consumption
    z_q_all = z_q.view(B, K + 1, D, side * side).permute(0, 1, 3, 2).contiguous()
    indices_all = indices.view(B, K + 1, side * side)
    recon_all = recon.view(B, K + 1, 1, 64, 64)
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

    Construction stores fixed hparams (alpha_slow, cpc_weight, ...) so
    the compiled graph specialises on them. Changing any of these across
    calls triggers recompilation, which is fine — they are constants.
    """

    def __init__(self, fsq, wm, *, alpha_slow, alpha_uniform, cpc_weight,
                 label_smoothing, focal_gamma, token_noise, fsq_noise,
                 neighbor_table=None, neighbor_counts=None,
                 soft_target_matrix=None):
        super().__init__()
        self.fsq = fsq
        self.wm = wm
        self.alpha_slow = float(alpha_slow)
        self.alpha_uniform = float(alpha_uniform)
        self.cpc_weight = float(cpc_weight)
        self.label_smoothing = float(label_smoothing)
        self.focal_gamma = float(focal_gamma)
        self.token_noise = float(token_noise)
        self.fsq_noise = float(fsq_noise)
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
        from deepdash.fsq import fsqvae_loss, grwm_slowness, grwm_uniformity

        m = _unwrap(self.wm)
        K = m.context_frames
        tpf = m.tokens_per_frame
        vs = m.vocab_size
        alive_tok = m.ALIVE_TOKEN
        death_tok = m.DEATH_TOKEN
        D = self.fsq.fsq.codebook_size  # not used; kept for clarity
        D = len(self.fsq.fsq.levels) if hasattr(self.fsq.fsq, "levels") else None
        # Reliable: FSQ latent dim is the output channels of the encoder
        # conv, which is len(levels). We infer it from the quantizer's
        # internal levels list.
        fsq_dim = self.wm.fsq_grad_proj.in_features

        z_e_all, z_q_all, indices_all, recon_all, frames_f = _encode_joint(
            self.fsq, raw_frames, K, tpf, fsq_dim)

        frame_last = frames_f[:, K - 1]
        frame_tgt = frames_f[:, K]
        recon_last = recon_all[:, K - 1]
        recon_tgt = recon_all[:, K]
        z_e_last = z_e_all[:, K - 1]
        z_e_tgt = z_e_all[:, K]
        recon_loss = (fsqvae_loss(recon_last, frame_last)
                      + fsqvae_loss(recon_tgt, frame_tgt)) / 2
        slow_loss = grwm_slowness(z_e_last, z_e_tgt)
        uniform_loss = grwm_uniformity(z_e_last)

        frame_tokens = _build_frame_tokens_joint(
            indices_all, is_death, alive_tok, death_tok)
        target = frame_tokens[:, -1]

        # Noise on context visual tokens (branch-free). Cost when noise=0
        # is the noise tensors being computed then discarded via where;
        # still cheap relative to attention.
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
        logits, cpc_loss = self.wm(
            frame_tokens, actions, z_q_ste_context=z_q_ste_ctx)
        token_loss = focal_cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target.reshape(-1),
            gamma=self.focal_gamma,
            soft_target_matrix=(self.soft_target_matrix if self._has_soft else None),
            label_smoothing=self.label_smoothing,
        )
        loss = (recon_loss
                + self.alpha_slow * slow_loss
                + self.alpha_uniform * uniform_loss
                + token_loss
                + self.cpc_weight * cpc_loss)

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

        metrics = {
            "token_loss": token_loss.detach(),
            "cpc_loss": cpc_loss.detach(),
            "recon_loss": recon_loss.detach(),
            "slow_loss": slow_loss.detach(),
            "uniform_loss": uniform_loss.detach(),
            "correct": correct,
            "n_tokens": n_tokens,
            "death_tp": tp,
            "death_fp": fp,
            "death_fn": fn,
        }
        return loss, metrics


def train_epoch_joint(joint_step, loader, optimizer, scaler, device,
                      amp_dtype=torch.bfloat16):
    """Run one training epoch by calling the compiled JointStep per batch.

    The joint_step module (typically torch.compile-wrapped) captures
    FSQ encode + aux losses + transformer + token/cpc losses as a single
    graph. The training loop here only handles data transfer, the
    optimizer step, grad-norm diagnostics, and metric accumulation.
    """
    joint_step.train()
    js = _unwrap(joint_step)
    tpf = _unwrap(js.wm).tokens_per_frame
    use_amp = device.type == "cuda"

    total_token_loss = 0.0
    total_cpc = 0.0
    total_recon = 0.0
    total_slow = 0.0
    total_unif = 0.0
    total_correct = 0
    total_tokens = 0
    tp = fp = fn = 0
    enc_grad_sq = tr_grad_sq = 0.0
    n_batches = 0

    for raw_frames, actions, is_death in loader:
        raw_frames = raw_frames.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        is_death = is_death.to(device, non_blocking=True).bool()
        B = raw_frames.size(0)

        with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            loss, metrics = joint_step(raw_frames, actions, is_death)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        all_params = list(js.wm.parameters()) + list(js.fsq.parameters())
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)

        with torch.no_grad():
            e = sum(p.grad.detach().pow(2).sum().item()
                    for p in js.fsq.parameters() if p.grad is not None)
            t = sum(p.grad.detach().pow(2).sum().item()
                    for p in js.wm.parameters() if p.grad is not None)
            enc_grad_sq += e
            tr_grad_sq += t

        scaler.step(optimizer)
        scaler.update()

        total_token_loss += metrics["token_loss"].item() * B
        total_cpc += metrics["cpc_loss"].item() * B
        total_recon += metrics["recon_loss"].item() * B
        total_slow += metrics["slow_loss"].item() * B
        total_unif += metrics["uniform_loss"].item() * B
        total_correct += int(metrics["correct"].item())
        total_tokens += int(metrics["n_tokens"].item())
        tp += int(metrics["death_tp"].item())
        fp += int(metrics["death_fp"].item())
        fn += int(metrics["death_fn"].item())
        n_batches += 1

    n_samples = max(1, total_tokens // tpf)
    death_prec = tp / (tp + fp + 1e-8)
    death_rec = tp / (tp + fn + 1e-8)
    death_f1 = 2 * death_prec * death_rec / (death_prec + death_rec + 1e-8)
    return {
        "loss": total_token_loss / n_samples,
        "acc": total_correct / max(1, total_tokens),
        "death_prec": death_prec,
        "death_rec": death_rec,
        "death_f1": death_f1,
        "cpc": total_cpc / n_samples,
        "recon": total_recon / n_samples,
        "slow": total_slow / n_samples,
        "unif": total_unif / n_samples,
        "enc_grad_rms": (enc_grad_sq / max(1, n_batches)) ** 0.5,
        "tr_grad_rms": (tr_grad_sq / max(1, n_batches)) ** 0.5,
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
    total_cpc = 0.0
    total_recon = 0.0
    total_slow = 0.0
    total_unif = 0.0
    total_correct = 0
    total_tokens = 0
    tp = fp = fn = 0

    for raw_frames, actions, is_death in loader:
        raw_frames = raw_frames.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        is_death = is_death.to(device, non_blocking=True).bool()
        B = raw_frames.size(0)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            _, metrics = joint_step(raw_frames, actions, is_death)
        total_token_loss += metrics["token_loss"].item() * B
        total_cpc += metrics["cpc_loss"].item() * B
        total_recon += metrics["recon_loss"].item() * B
        total_slow += metrics["slow_loss"].item() * B
        total_unif += metrics["uniform_loss"].item() * B
        total_correct += int(metrics["correct"].item())
        total_tokens += int(metrics["n_tokens"].item())
        tp += int(metrics["death_tp"].item())
        fp += int(metrics["death_fp"].item())
        fn += int(metrics["death_fn"].item())

    n_samples = max(1, total_tokens // tpf)
    death_prec = tp / (tp + fp + 1e-8)
    death_rec = tp / (tp + fn + 1e-8)
    death_f1 = 2 * death_prec * death_rec / (death_prec + death_rec + 1e-8)
    return {
        "loss": total_token_loss / n_samples,
        "acc": total_correct / max(1, total_tokens),
        "death_prec": death_prec,
        "death_rec": death_rec,
        "death_f1": death_f1,
        "cpc": total_cpc / n_samples,
        "recon": total_recon / n_samples,
        "slow": total_slow / n_samples,
        "unif": total_unif / n_samples,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Transformer world model")
    parser.add_argument("--episodes-dir", default="data/death_episodes")
    parser.add_argument("--expert-episodes-dir", default="data/expert_episodes",
                        help="Directory with expert episodes (no death on last frame)")
    parser.add_argument("--config", default=None, help="YAML config path (default: configs/v4.yaml)")
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
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--cpc-weight", type=float, default=None)
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
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    # Mixed precision + compile (mirrors train_fsq.py conventions).
    # A100 defaults: bfloat16 + reduce-overhead. RTX 2060 SUPER (Turing) has
    # no native bf16 — it emulates at ~12x slower. Use float16 + default
    # compile mode on Turing; see configs/*-local.yaml.
    parser.add_argument("--amp-dtype", choices=["bfloat16", "float16"],
                        default=None,
                        help="AMP dtype. Default bfloat16 (A100). Use float16 "
                             "on Turing (RTX 20xx) — no native bf16 there.")
    parser.add_argument("--compile-mode",
                        choices=["reduce-overhead", "default", "none"],
                        default=None,
                        help="torch.compile mode. Default reduce-overhead "
                             "(A100 + modern CUDA). Use 'default' on Turing "
                             "to avoid CUDA-graph issues, or 'none' to disable "
                             "compile entirely (eager).")
    # E6.1 joint training: unfreeze FSQ, train simultaneously, STE-routed
    # gradient from transformer losses back to the encoder via
    # WorldModel.fsq_grad_proj. When --joint is off, the script runs in
    # V5 mode on pre-tokenized episodes (unchanged).
    parser.add_argument("--joint", action="store_true", default=None,
                        help="Enable joint FSQ+Transformer training (E6.1).")
    parser.add_argument("--fsq-lr", type=float, default=None,
                        help="Peak LR for the FSQ encoder param group in joint "
                             "mode. Default: V5 FSQ peak (1e-3).")
    parser.add_argument("--alpha-slow", type=float, default=None,
                        help="GRWM temporal slowness weight (joint mode).")
    parser.add_argument("--alpha-uniform", type=float, default=None,
                        help="GRWM uniformity weight (joint mode).")
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

    # Joint mode loads raw frames (frames.npy) and encodes on-the-fly.
    # V5 mode loads pre-tokenized data (tokens.npy) produced by
    # scripts/tokenize_episodes.py.
    joint = bool(getattr(args, "joint", False))
    source_file = "frames.npy" if joint else "tokens.npy"

    episodes_dir = Path(args.episodes_dir)
    all_episodes = sorted(ep for ep in episodes_dir.glob("*")
                          if (ep / source_file).exists())
    death_episodes = set(ep.name for ep in all_episodes)

    expert_dir = Path(args.expert_episodes_dir)
    if expert_dir.exists():
        expert_eps = sorted(ep for ep in expert_dir.glob("*")
                            if (ep / source_file).exists())
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
        # V5 tokenized path (unchanged)
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

    if joint:
        fsq_lr_val = args.fsq_lr if args.fsq_lr is not None else 1e-3
        optimizer = build_optimizer(model, args, fsq_encoder=fsq, fsq_lr=fsq_lr_val)
    else:
        optimizer = build_optimizer(model, args)

    # Resolve AMP dtype and compile mode with A100 defaults; local configs
    # (RTX 2060 SUPER / Turing) override via --amp-dtype float16 and
    # --compile-mode default, since Turing emulates bf16 at ~12x slower
    # and reduce-overhead's CUDA graphs have extra pitfalls on smaller SMs.
    amp_dtype_name = getattr(args, "amp_dtype", None) or "bfloat16"
    amp_dtype = {"bfloat16": torch.bfloat16,
                 "float16": torch.float16}[amp_dtype_name]
    compile_mode = getattr(args, "compile_mode", None) or "reduce-overhead"
    # GradScaler is only needed for float16 (bfloat16 has enough range to
    # skip scaling).
    scaler_enabled = (amp_dtype == torch.float16) and device.type == "cuda"
    scaler = torch.GradScaler(device.type, enabled=scaler_enabled)
    print(f"AMP: {amp_dtype_name} | GradScaler: "
          f"{'on' if scaler_enabled else 'off'} | compile: {compile_mode}")

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(exist_ok=True)

    start_epoch = 1
    best_val_loss = float("inf")

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
            best_val_loss = resume_state["best_val_loss"]
            wandb_resume_id = resume_state.get("wandb_run_id")
            print(f"Resumed from epoch {resume_state['epoch']} (best val loss: {best_val_loss:.4f})")
        else:
            print("No checkpoint found, starting fresh.")

    if start_epoch == 1:
        import json
        with open(ckpt_dir / "transformer_args.json", "w") as f:
            json.dump(vars(args), f, indent=2)

    wandb_init(project="deepdash", name=f"transformer-{args.embed_dim}d",
               config=vars(args), resume_id=wandb_resume_id)

    # torch.compile the model. In V5 mode we compile WorldModel alone.
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
            inductor_cfg.compile_threads = min(os.cpu_count() or 1, 8)
            model = torch.compile(model, mode=compile_mode)
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

    # Build structured label smoothing matrix
    kernel = args.fsq_kernel or "gaussian"
    if kernel == "aniso_gaussian":
        sigma_arg = args.fsq_sigma_d
        has_bandwidth = sigma_arg is not None and all(s > 0 for s in sigma_arg)
    else:
        sigma_arg = args.fsq_sigma
        has_bandwidth = sigma_arg is not None and sigma_arg > 0
    if has_bandwidth and args.label_smoothing > 0:
        soft_target_matrix = build_structured_smooth_targets(
            args.fsq_levels, model.full_vocab_size,
            sigma=sigma_arg, smoothing=args.label_smoothing,
            dim_weights=args.fsq_dim_weights,
            kernel=kernel,
        ).to(device)
        # Check how concentrated the smoothing mass is
        visual_row = soft_target_matrix[0, :model.vocab_size]
        top5 = visual_row.topk(6).values  # includes self
        w_str = f", dim_weights={args.fsq_dim_weights}" if args.fsq_dim_weights else ""
        sigma_str = (f"[{', '.join(f'{s:.3f}' for s in sigma_arg)}]"
                     if kernel == "aniso_gaussian" else f"{sigma_arg}")
        print(f"Structured label smoothing: kernel={kernel}, σ={sigma_str}, "
              f"ε={args.label_smoothing}{w_str}")
        print(f"  Top-6 target probs for token 0: {top5.tolist()}")
    else:
        soft_target_matrix = None

    # Build and compile JointStep (E6.1). Two instances share the same
    # underlying fsq + model parameters; only the noise settings differ
    # so each can be specialised by torch.compile.
    joint_step_train = None
    joint_step_val = None
    if joint:
        joint_step_train = JointStep(
            fsq=fsq, wm=model,
            alpha_slow=args.alpha_slow, alpha_uniform=args.alpha_uniform,
            cpc_weight=args.cpc_weight,
            label_smoothing=args.label_smoothing,
            focal_gamma=args.focal_gamma,
            token_noise=args.token_noise, fsq_noise=args.fsq_noise,
            neighbor_table=neighbor_table, neighbor_counts=neighbor_counts,
            soft_target_matrix=soft_target_matrix,
        ).to(device)
        joint_step_val = JointStep(
            fsq=fsq, wm=model,
            alpha_slow=args.alpha_slow, alpha_uniform=args.alpha_uniform,
            cpc_weight=args.cpc_weight,
            label_smoothing=args.label_smoothing,
            focal_gamma=args.focal_gamma,
            token_noise=0.0, fsq_noise=0.0,          # no noise at eval
            neighbor_table=None, neighbor_counts=None,
            soft_target_matrix=soft_target_matrix,
        ).to(device)
        if compile_mode == "none":
            print("JointStep compile disabled (--compile-mode none)")
        else:
            try:
                import torch._inductor.config as inductor_cfg
                inductor_cfg.compile_threads = min(os.cpu_count() or 1, 8)
                joint_step_train = torch.compile(joint_step_train, mode=compile_mode)
                joint_step_val = torch.compile(joint_step_val, mode=compile_mode)
                print(f"JointStep compiled (mode={compile_mode}, train + val)")
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
        # last_epoch semantics: optimizer.step() already ran `start_epoch - 1`
        # times before the first pending step, so set last_epoch accordingly
        # and let LambdaLR recompute the factor analytically from lr_factor().
        scheduler.last_epoch = start_epoch - 1
        scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]

    log_path = ckpt_dir / "transformer_log.csv"
    log_header = ["epoch", "train_total", "train_loss", "train_acc",
                  "train_death_prec", "train_death_rec", "train_death_f1",
                  "train_cpc",
                  "val_total", "val_loss", "val_acc",
                  "val_death_prec", "val_death_rec", "val_death_f1",
                  "val_cpc",
                  "gap", "lr", "time_s"]
    if joint:
        # E6.1: merged CSV carries the FSQ aux losses (formerly written to
        # fsq_log.csv by train_fsq.py) and the per-source grad-norm diagnostic
        # used to validate the encoder-LR / transformer-LR ratio.
        log_header += [
            "train_recon", "train_slow", "train_unif",
            "val_recon", "val_slow", "val_unif",
            "encoder_grad_rms", "transformer_grad_rms",
            "fsq_lr",
        ]

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
            if joint:
                train_metrics = train_epoch_joint(
                    joint_step_train, train_loader, optimizer, scaler, device,
                    amp_dtype=amp_dtype)
                val_metrics = val_epoch_joint(
                    joint_step_val, val_loader, device, amp_dtype=amp_dtype)
                train_loss = train_metrics["loss"]
                train_acc = train_metrics["acc"]
                train_d_prec = train_metrics["death_prec"]
                train_d_rec = train_metrics["death_rec"]
                train_d_f1 = train_metrics["death_f1"]
                train_cpc = train_metrics["cpc"]
                val_loss = val_metrics["loss"]
                val_acc = val_metrics["acc"]
                val_d_prec = val_metrics["death_prec"]
                val_d_rec = val_metrics["death_rec"]
                val_d_f1 = val_metrics["death_f1"]
                val_cpc = val_metrics["cpc"]
            else:
                train_loss, train_acc, train_d_prec, train_d_rec, train_d_f1, train_cpc = train_epoch(
                    model, train_loader, optimizer, scaler,
                    args.cpc_weight, device, token_noise=args.token_noise,
                    fsq_noise=args.fsq_noise,
                    neighbor_table=neighbor_table, neighbor_counts=neighbor_counts,
                    label_smoothing=args.label_smoothing,
                    focal_gamma=args.focal_gamma,
                    soft_target_matrix=soft_target_matrix,
                    amp_dtype=amp_dtype)
                val_loss, val_acc, val_d_prec, val_d_rec, val_d_f1, val_cpc = val_epoch(
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
                + (f" fsq_LR: {fsq_lr_now:.1e}" if fsq_lr_now is not None else "")
            )
            if joint:
                print(
                    f"  FSQ: train[recon={train_metrics['recon']:.4f} "
                    f"slow={train_metrics['slow']:.4f} unif={train_metrics['unif']:.4f}] "
                    f"val[recon={val_metrics['recon']:.4f} "
                    f"slow={val_metrics['slow']:.4f} unif={val_metrics['unif']:.4f}] "
                    f"| grad_rms enc={train_metrics['enc_grad_rms']:.3e} "
                    f"tr={train_metrics['tr_grad_rms']:.3e}"
                )

            row = [
                epoch, f"{train_total:.6f}", f"{train_loss:.6f}", f"{train_acc:.4f}",
                f"{train_d_prec:.4f}", f"{train_d_rec:.4f}", f"{train_d_f1:.4f}",
                f"{train_cpc:.4f}",
                f"{val_total:.6f}", f"{val_loss:.6f}", f"{val_acc:.4f}",
                f"{val_d_prec:.4f}", f"{val_d_rec:.4f}", f"{val_d_f1:.4f}",
                f"{val_cpc:.4f}",
                f"{gap:.4f}", f"{lr:.1e}", f"{dt:.1f}",
            ]
            if joint:
                row += [
                    f"{train_metrics['recon']:.6f}",
                    f"{train_metrics['slow']:.6f}",
                    f"{train_metrics['unif']:.6f}",
                    f"{val_metrics['recon']:.6f}",
                    f"{val_metrics['slow']:.6f}",
                    f"{val_metrics['unif']:.6f}",
                    f"{train_metrics['enc_grad_rms']:.6e}",
                    f"{train_metrics['tr_grad_rms']:.6e}",
                    f"{fsq_lr_now:.1e}" if fsq_lr_now is not None else "",
                ]
            log_writer.writerow(row)
            log_file.flush()

            wandb_payload = {
                "epoch": epoch,
                "transformer/train/total": train_total,
                "transformer/train/loss": train_loss,
                "transformer/train/acc": train_acc,
                "transformer/train/death_f1": train_d_f1,
                "transformer/train/cpc": train_cpc,
                "transformer/val/total": val_total,
                "transformer/val/loss": val_loss,
                "transformer/val/acc": val_acc,
                "transformer/val/death_f1": val_d_f1,
                "transformer/val/cpc": val_cpc,
                "transformer/gap": gap,
                "transformer/lr": lr,
            }
            if joint:
                wandb_payload.update({
                    "fsq/train/recon": train_metrics["recon"],
                    "fsq/train/slow": train_metrics["slow"],
                    "fsq/train/unif": train_metrics["unif"],
                    "fsq/val/recon": val_metrics["recon"],
                    "fsq/val/slow": val_metrics["slow"],
                    "fsq/val/unif": val_metrics["unif"],
                    "fsq/lr": fsq_lr_now,
                    "grad/encoder_rms": train_metrics["enc_grad_rms"],
                    "grad/transformer_rms": train_metrics["tr_grad_rms"],
                })
            wandb_log(wandb_payload)

            # Save full state
            state_dict = {
                "epoch": epoch,
                "model": _unwrap(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_val_loss": best_val_loss,
                "wandb_run_id": wandb_run_id(),
            }
            if joint:
                state_dict["fsq"] = fsq.state_dict()
            torch.save(state_dict, ckpt_dir / "transformer_state.pt")

            if val_total < best_val_loss:
                best_val_loss = val_total
                patience_counter = 0
                torch.save(_unwrap(model).state_dict(), ckpt_dir / "transformer_best.pt")
                if joint:
                    torch.save(fsq.state_dict(), ckpt_dir / "fsq_best.pt")
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
        interrupt_state = {
            "epoch": epoch,
            "model": _unwrap(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_val_loss": best_val_loss,
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
    print(f"\nTraining complete. Best val total loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to {ckpt_dir}/")


if __name__ == "__main__":
    main()
