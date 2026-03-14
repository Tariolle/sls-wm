"""Visualize how FSQ semantic similarity degrades with distance.

For each example, shows the reconstruction with perturbations at
increasing FSQ distances: +1 in 1 dim, +2 in 1 dim, +1 in 2 dims,
+1 in 3 dims, +1 in 4 dims. Also computes MSE between original and
perturbed reconstructions to quantify the visual difference.

Usage:
    python scripts/visualize_fsq_distances.py
    python scripts/visualize_fsq_distances.py --n-examples 5
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
    parser.add_argument("--output", default="fsq_distances.png")
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

    episodes_dir = Path(args.episodes_dir)
    all_frames_files = sorted(episodes_dir.glob("*/frames.npy"))
    rng = np.random.default_rng(args.seed)

    frames = []
    for _ in range(args.n_examples):
        ep_frames = np.load(rng.choice(all_frames_files))
        frames.append(ep_frames[rng.integers(0, len(ep_frames))])
    frames = np.stack(frames)

    x = torch.from_numpy(frames).float().unsqueeze(1).to(device) / 255.0

    # Perturbation configs: (label, list of (dim, delta) pairs)
    perturbations = [
        ("Recon", []),
        ("+1 d0", [(0, +1)]),
        ("+2 d0", [(0, +2)]),
        ("+1 d0,d1", [(0, +1), (1, +1)]),
        ("+1 d0,d1,d2", [(0, +1), (1, +1), (2, +1)]),
        ("+1 all", [(0, +1), (1, +1), (2, +1), (3, +1)]),
        ("+2 d0,d1", [(0, +2), (1, +2)]),
        ("max", [(d, levels[d] // 2) for d in range(n_dims)]),
    ]

    n_cols = 1 + len(perturbations)  # original + perturbations
    fig, axes = plt.subplots(args.n_examples, n_cols,
                             figsize=(n_cols * 1.6, args.n_examples * 1.6),
                             squeeze=False)

    with torch.no_grad():
        for row in range(args.n_examples):
            frame = x[row:row + 1]
            z_e = model.encoder(frame)
            z_q, indices = fsq(z_e)

            # Pick position away from borders to allow +2 perturbations
            pos_r = rng.integers(2, 6)
            pos_c = rng.integers(2, 6)
            token_id = indices[0, pos_r, pos_c].item()

            # Original coords for bounds checking
            orig_coords = z_q[0, :, pos_r, pos_c].clone()

            # Decode original reconstruction
            recon_orig = model.decoder(z_q)[0, 0].cpu().numpy()

            # Show original frame
            axes[row, 0].imshow(frames[row], cmap="gray")
            rect = Rectangle((pos_c * 8, pos_r * 8), 8, 8,
                             linewidth=1, edgecolor='r', facecolor='none')
            axes[row, 0].add_patch(rect)
            axes[row, 0].set_ylabel(f"tok={token_id}", fontsize=7)
            if row == 0:
                axes[row, 0].set_title("Original", fontsize=7)

            for col_idx, (label, deltas) in enumerate(perturbations):
                col = col_idx + 1
                z_mod = z_q.clone()

                for dim, delta in deltas:
                    new_val = orig_coords[dim].item() + delta
                    half = levels[dim] // 2
                    # Match FSQ valid range: odd L → [-half, half], even L → [-half, half-1]
                    max_val = half if levels[dim] % 2 == 1 else half - 1
                    new_val = max(-half, min(max_val, new_val))
                    z_mod[0, dim, pos_r, pos_c] = new_val

                recon_mod = model.decoder(z_mod)[0, 0].cpu().numpy()
                # Compute MSE on the affected patch (8x8 region)
                pr, pc = pos_r * 8, pos_c * 8
                patch_orig = recon_orig[pr:pr+8, pc:pc+8]
                patch_mod = recon_mod[pr:pr+8, pc:pc+8]
                patch_mse = ((patch_orig - patch_mod) ** 2).mean()
                # Full frame MSE
                full_mse = ((recon_orig - recon_mod) ** 2).mean()

                axes[row, col].imshow(recon_mod, cmap="gray")
                rect_n = Rectangle((pos_c * 8, pos_r * 8), 8, 8,
                                   linewidth=1, edgecolor='r', facecolor='none')
                axes[row, col].add_patch(rect_n)

                if row == 0:
                    axes[row, col].set_title(f"{label}", fontsize=6)
                axes[row, col].set_xlabel(f"p:{patch_mse:.4f}\nf:{full_mse:.5f}",
                                          fontsize=5)

            for ax_col in range(n_cols):
                axes[row, ax_col].set_xticks([])
                axes[row, ax_col].set_yticks([])

    fig.suptitle("FSQ Distance vs Visual Difference (p=patch MSE, f=frame MSE)",
                 fontsize=9)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
