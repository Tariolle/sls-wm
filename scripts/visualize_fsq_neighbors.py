"""Visualize FSQ neighbor similarity: decode a token and its ±1 neighbors in each dim.

For 5 random tokens, shows the original decoded patch and its 8 neighbors
(±1 in each of the 4 FSQ dimensions). Helps verify whether FSQ neighbors
are semantically similar (for structured token noise augmentation).

Usage:
    python scripts/visualize_fsq_neighbors.py
    python scripts/visualize_fsq_neighbors.py --checkpoint checkpoints/fsq_best.pt
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepdash.fsq import FSQVAE, FSQQuantizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/fsq_best.pt")
    parser.add_argument("--levels", type=int, nargs="+", default=[8, 5, 5, 5])
    parser.add_argument("--n-examples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="fsq_neighbors.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FSQVAE(levels=args.levels).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    fsq = model.fsq
    levels = args.levels
    n_dims = len(levels)

    rng = np.random.default_rng(args.seed)
    token_ids = rng.integers(0, fsq.codebook_size, size=args.n_examples)

    # Layout: n_examples rows, (1 original + 2*n_dims neighbors) columns
    n_cols = 1 + 2 * n_dims
    fig, axes = plt.subplots(args.n_examples, n_cols, figsize=(n_cols * 1.5, args.n_examples * 1.5))

    col_labels = ["Original"]
    for d in range(n_dims):
        col_labels.append(f"d{d}-1")
        col_labels.append(f"d{d}+1")

    with torch.no_grad():
        for row, token_id in enumerate(token_ids):
            # Decode original: create a 1x1 spatial grid
            idx = torch.tensor([[[token_id]]], dtype=torch.long, device=device)  # (1,1,1)
            z_q = fsq.indices_to_codes(idx)  # (1, D, 1, 1)
            patch = model.decoder(z_q)  # (1, 1, H, W)
            img = patch[0, 0].cpu().numpy()

            axes[row, 0].imshow(img, cmap="gray", vmin=0, vmax=1)
            axes[row, 0].set_ylabel(f"token {token_id}", fontsize=8)

            # Decode neighbors: ±1 in each dimension
            col = 1
            for d in range(n_dims):
                for delta in [-1, +1]:
                    z_neighbor = z_q.clone()
                    z_neighbor[0, d, 0, 0] += delta

                    # Clamp to valid range
                    half = levels[d] // 2
                    max_val = half if levels[d] % 2 == 1 else half
                    min_val = -half
                    val = z_neighbor[0, d, 0, 0].item()

                    if val < min_val or val > max_val:
                        # Out of range — show blank
                        axes[row, col].imshow(np.zeros_like(img), cmap="gray", vmin=0, vmax=1)
                        axes[row, col].set_title("OOB", fontsize=6, color="red")
                    else:
                        patch_n = model.decoder(z_neighbor)
                        img_n = patch_n[0, 0].cpu().numpy()
                        axes[row, col].imshow(img_n, cmap="gray", vmin=0, vmax=1)
                    col += 1

    for col, label in enumerate(col_labels):
        axes[0, col].set_title(label, fontsize=7)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("FSQ Token Neighbors: ±1 in each dimension", fontsize=10)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
