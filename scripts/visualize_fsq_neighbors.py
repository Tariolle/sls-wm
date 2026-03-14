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

    # 6 perturbations in a 3x2 grid: original, -1 d0, +1 d0, +1 d1, +1 d2, +1 d3
    perturbations = [
        ("Original", None, None),
        ("$-1$ dim 0", 0, -1),
        ("$+1$ dim 0", 0, +1),
        ("$+1$ dim 1", 1, +1),
        ("$+1$ dim 2", 2, +1),
        ("$+1$ dim 3", 3, +1),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(6, 9), squeeze=False)

    with torch.no_grad():
        frame = x[0:1]
        z_e = model.encoder(frame)
        z_q, indices = fsq(z_e)

        pos_r = rng.integers(1, 7)
        pos_c = rng.integers(1, 7)

        recon_orig = model.decoder(z_q)[0, 0].cpu().numpy()

        for i, (label, dim, delta) in enumerate(perturbations):
            row, col = i // 2, i % 2
            ax = axes[row, col]

            if dim is None:
                img = recon_orig
            else:
                z_mod = z_q.clone()
                new_val = z_mod[0, dim, pos_r, pos_c].item() + delta
                half = levels[dim] // 2
                max_val = half if levels[dim] % 2 == 1 else half - 1
                new_val = max(-half, min(max_val, new_val))
                z_mod[0, dim, pos_r, pos_c] = new_val
                img = model.decoder(z_mod)[0, 0].cpu().numpy()

            ax.imshow(img, cmap="gray")
            rect = Rectangle((pos_c * 8, pos_r * 8), 8, 8,
                             linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.set_title(label, fontsize=24)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
