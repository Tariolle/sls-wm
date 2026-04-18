"""Generate the SLS kernel visualization for the paper (Figure 2).

Two panels:
  (a) 2D heatmap slice of SLS weights for a center token
  (b) Sorted SLS vs uniform distribution comparison (log scale)

Defaults reflect the current V5 baseline (FSQ [5,5,5,5] with calibrated
per-dim weights). Override via CLI for other codebooks or ablations.

Usage:
    python scripts/gen_sls_figure.py
    python scripts/gen_sls_figure.py --output paper/figures/sls_kernel.pdf
    python scripts/gen_sls_figure.py --levels 5 5 5 5 \\
        --dim-weights 1.02 0.94 0.83 1.20 --target-coords 2 2 2 2
"""

import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import torch


def build_sls_row(levels, target_idx, sigma=0.9, smoothing=0.1, dim_weights=None):
    """Build the SLS target distribution for a single token (CPU-only)."""
    vocab_size = math.prod(levels)
    n_dims = len(levels)

    # Mixed-radix decomposition
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

    # Weighted squared distance from target
    target_coords = coords[target_idx].unsqueeze(0)  # (1, D)
    diff = coords - target_coords  # (V, D)
    if dim_weights is not None:
        w = torch.tensor(dim_weights, dtype=torch.float32)
        diff = diff * w
    sq_dist = (diff ** 2).sum(dim=-1)  # (V,)

    # Gaussian kernel
    weights = torch.exp(-sq_dist / (2 * sigma ** 2))
    weights[target_idx] = 0.0

    # Normalize and apply smoothing
    row_sum = weights.sum().clamp(min=1e-8)
    weights = smoothing * weights / row_sum
    weights[target_idx] = 1.0 - smoothing

    return weights.numpy(), coords.numpy(), divisors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="paper/figures/sls_kernel.pdf")
    parser.add_argument("--levels", type=int, nargs="+", default=[5, 5, 5, 5],
                        help="FSQ codebook levels (default: V5 baseline [5,5,5,5])")
    parser.add_argument("--sigma", type=float, default=0.9)
    parser.add_argument("--smoothing", type=float, default=0.1,
                        help="Label smoothing epsilon")
    parser.add_argument("--dim-weights", type=float, nargs="+",
                        default=[1.02, 0.94, 0.83, 1.20],
                        help="Per-dim sensitivity weights (default: V5 baseline)")
    parser.add_argument("--target-coords", type=int, nargs="+",
                        default=None,
                        help="Target token coordinates as ints per dim "
                             "(default: center of the lattice)")
    parser.add_argument("--slice-dims", type=int, nargs=2, default=[1, 2],
                        help="Which two FSQ dims to show as the 2D slice "
                             "(default: 1 2, matching the paper's V5 figure)")
    args = parser.parse_args()

    levels = args.levels
    n_dims = len(levels)
    if len(args.dim_weights) != n_dims:
        raise SystemExit(
            f"--dim-weights length {len(args.dim_weights)} must match "
            f"--levels length {n_dims}"
        )

    # Choose target: center coords unless overridden
    if args.target_coords is None:
        target_coords = [L // 2 for L in levels]
    else:
        if len(args.target_coords) != n_dims:
            raise SystemExit(
                f"--target-coords length {len(args.target_coords)} must match "
                f"--levels length {n_dims}"
            )
        target_coords = list(args.target_coords)

    # Decompose target coords into flat index
    divisors = []
    acc = 1
    for L in reversed(levels):
        divisors.append(acc)
        acc *= L
    divisors.reverse()
    target_idx = sum(c * div for c, div in zip(target_coords, divisors))

    print(f"Levels:        {levels}")
    print(f"Dim weights:   {args.dim_weights}")
    print(f"Sigma:         {args.sigma}")
    print(f"Epsilon:       {args.smoothing}")
    print(f"Target coords: {tuple(target_coords)}  ->  index {target_idx}")

    sls_row, coords, divisors = build_sls_row(
        levels, target_idx, args.sigma, args.smoothing, args.dim_weights,
    )

    # --- Paper-quality settings ---
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.4))

    # --- Panel (a): 2D heatmap slice ---
    # Fix all dims EXCEPT the two we're slicing over
    d1, d2 = args.slice_dims
    if d1 == d2 or d1 < 0 or d2 < 0 or d1 >= n_dims or d2 >= n_dims:
        raise SystemExit(f"--slice-dims {d1} {d2} invalid for n_dims={n_dims}")
    grid = np.full((levels[d1], levels[d2]), np.nan)
    for v1 in range(levels[d1]):
        for v2 in range(levels[d2]):
            cur_coords = list(target_coords)
            cur_coords[d1] = v1
            cur_coords[d2] = v2
            idx = sum(c * div for c, div in zip(cur_coords, divisors))
            if (v1, v2) != (target_coords[d1], target_coords[d2]):
                grid[v1, v2] = sls_row[idx]

    im = ax1.imshow(grid, cmap="viridis", origin="lower", aspect="equal")
    ax1.plot(target_coords[d2], target_coords[d1], "s",
             markeredgecolor="red", markerfacecolor="red",
             markersize=12, markeredgewidth=2)
    ax1.set_xlabel(f"Dim {d2} ({levels[d2]} levels)")
    ax1.set_ylabel(f"Dim {d1} ({levels[d1]} levels)")
    ax1.set_title("(a) SLS weights (2D slice)")
    ax1.set_xticks(range(levels[d2]))
    ax1.set_yticks(range(levels[d1]))
    cb = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7)

    # --- Panel (b): Sorted distribution comparison ---
    off_diag = np.delete(sls_row, target_idx)
    sorted_sls = np.sort(off_diag)[::-1]
    uniform_val = args.smoothing / (math.prod(levels) - 1)

    ax2.semilogy(sorted_sls, color="#2196F3", linewidth=1.2, label="SLS (Gaussian)")
    ax2.axhline(uniform_val, color="#FF9800", linewidth=1.2,
                linestyle="--", label="Uniform")
    ax2.set_xlabel("Token rank (by SLS weight)")
    ax2.set_ylabel("Smoothing probability")
    ax2.set_title("(b) SLS vs. uniform smoothing")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.set_xlim(0, len(sorted_sls))

    plt.tight_layout()
    plt.savefig(args.output, bbox_inches="tight")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
