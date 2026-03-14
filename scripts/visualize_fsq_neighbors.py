"""Visualize FSQ neighbor similarity on real frames.

For 5 random frames, picks a random token position, swaps it with each
FSQ neighbor (±1 in each of the 4 dimensions), and decodes the full 8×8
grid to show the visual effect. Helps verify whether FSQ neighbors
produce semantically similar reconstructions.

Usage:
    python scripts/visualize_fsq_neighbors.py
    python scripts/visualize_fsq_neighbors.py --episodes-dir data/episodes
"""

import argparse
import sys
from pathlib import Path

from matplotlib.patches import Rectangle

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepdash.fsq import FSQVAE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/fsq_best.pt")
    parser.add_argument("--episodes-dir", default="data/episodes")
    parser.add_argument("--levels", type=int, nargs="+", default=[8, 5, 5, 5])
    parser.add_argument("--n-examples", type=int, default=1)
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

    # Load random frames from episodes
    episodes_dir = Path(args.episodes_dir)
    all_frames_files = sorted(episodes_dir.glob("*/frames.npy"))
    rng = np.random.default_rng(args.seed)

    frames = []
    for _ in range(args.n_examples):
        ep_frames = np.load(rng.choice(all_frames_files))
        frames.append(ep_frames[rng.integers(0, len(ep_frames))])
    frames = np.stack(frames)  # (N, 64, 64)

    # Encode all frames
    x = torch.from_numpy(frames).float().unsqueeze(1).to(device) / 255.0  # (N, 1, 64, 64)

    # Layout: n_examples rows, (1 original + 1 recon + 2*n_dims neighbors) columns
    n_cols = 2 + 2 * n_dims
    fig, axes = plt.subplots(args.n_examples, n_cols,
                             figsize=(n_cols * 1.5, args.n_examples * 1.5),
                             squeeze=False)

    col_labels = ["Original", "Recon"]
    for d in range(n_dims):
        col_labels.append(f"d{d}-1")
        col_labels.append(f"d{d}+1")

    with torch.no_grad():
        for row in range(args.n_examples):
            frame = x[row:row + 1]  # (1, 1, 64, 64)

            # Encode to get z_q codes and indices
            z_e = model.encoder(frame)  # (1, D, 8, 8)
            z_q, indices = fsq(z_e)  # z_q: (1, D, 8, 8), indices: (1, 8, 8)

            # Pick a random non-border position for clearer effect
            pos_r = rng.integers(1, 7)
            pos_c = rng.integers(1, 7)
            token_id = indices[0, pos_r, pos_c].item()

            # Show original frame
            axes[row, 0].imshow(frames[row], cmap="gray")
            rect = Rectangle((pos_c * 8, pos_r * 8), 8, 8,
                             linewidth=1, edgecolor='r', facecolor='none')
            axes[row, 0].add_patch(rect)
            axes[row, 0].set_ylabel(f"pos=({pos_r},{pos_c})\ntok={token_id}", fontsize=7)

            # Show reconstruction from original codes
            recon = model.decoder(z_q)
            axes[row, 1].imshow(recon[0, 0].cpu().numpy(), cmap="gray")
            rect2 = Rectangle((pos_c * 8, pos_r * 8), 8, 8,
                               linewidth=1, edgecolor='r', facecolor='none')
            axes[row, 1].add_patch(rect2)

            # Swap token at (pos_r, pos_c) with each neighbor
            col = 2
            for d in range(n_dims):
                for delta in [-1, +1]:
                    z_mod = z_q.clone()
                    new_val = z_mod[0, d, pos_r, pos_c].item() + delta

                    # Check bounds
                    half = levels[d] // 2
                    min_val = -half
                    max_val = half if levels[d] % 2 == 1 else half - 1

                    if new_val < min_val or new_val > max_val:
                        axes[row, col].imshow(np.ones((64, 64)) * 0.5, cmap="gray",
                                              vmin=0, vmax=1)
                        axes[row, col].set_title("OOB", fontsize=6, color="red")
                    else:
                        z_mod[0, d, pos_r, pos_c] = new_val
                        recon_mod = model.decoder(z_mod)
                        img_mod = recon_mod[0, 0].cpu().numpy()
                        axes[row, col].imshow(img_mod, cmap="gray")
                        rect_n = Rectangle((pos_c * 8, pos_r * 8), 8, 8,
                                           linewidth=1, edgecolor='r', facecolor='none')
                        axes[row, col].add_patch(rect_n)
                    col += 1

    for col, label in enumerate(col_labels):
        axes[0, col].set_title(label, fontsize=7)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("FSQ Neighbor Substitution: ±1 in each dim at marked position", fontsize=10)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
