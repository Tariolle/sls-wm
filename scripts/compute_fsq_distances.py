"""Compute average semantic impact of FSQ perturbations across many frames.

For each perturbation type (±1 per dim, ±2 per dim, cross-dim combos),
computes mean patch MSE over a large sample of real frames and token
positions. Produces a summary table for supporting structured label
smoothing design decisions.

Usage:
    python scripts/compute_fsq_distances.py
    python scripts/compute_fsq_distances.py --n-frames 500 --n-positions 8
"""

import argparse
import sys
from pathlib import Path
from itertools import combinations

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepdash.fsq import FSQVAE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/fsq_best.pt")
    parser.add_argument("--episodes-dir", default="data/episodes")
    parser.add_argument("--levels", type=int, nargs="+", default=[8, 5, 5, 5])
    parser.add_argument("--n-frames", type=int, default=200,
                        help="Number of random frames to sample")
    parser.add_argument("--n-positions", type=int, default=4,
                        help="Random token positions per frame")
    parser.add_argument("--seed", type=int, default=42)
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

    # Load random frames
    frames = []
    for _ in range(args.n_frames):
        ep_frames = np.load(rng.choice(all_frames_files))
        frames.append(ep_frames[rng.integers(0, len(ep_frames))])
    frames = np.stack(frames)

    x = torch.from_numpy(frames).float().unsqueeze(1).to(device) / 255.0

    # Define perturbation types
    perturbations = {}

    # Single dim ±1 and ±2
    for d in range(n_dims):
        perturbations[f"+1 d{d}"] = [(d, +1)]
        perturbations[f"-1 d{d}"] = [(d, -1)]
        perturbations[f"+2 d{d}"] = [(d, +2)]

    # Two dims ±1
    for d1, d2 in combinations(range(n_dims), 2):
        perturbations[f"+1 d{d1},d{d2}"] = [(d1, +1), (d2, +1)]

    # Three dims ±1
    for d1, d2, d3 in combinations(range(n_dims), 3):
        perturbations[f"+1 d{d1},d{d2},d{d3}"] = [(d1, +1), (d2, +1), (d3, +1)]

    # All dims ±1
    perturbations["+1 all"] = [(d, +1) for d in range(n_dims)]

    # +2 in two dims
    for d1, d2 in combinations(range(n_dims), 2):
        perturbations[f"+2 d{d1},d{d2}"] = [(d1, +2), (d2, +2)]

    # Collect results
    results = {name: [] for name in perturbations}

    with torch.no_grad():
        for i in range(args.n_frames):
            frame = x[i:i + 1]
            z_e = model.encoder(frame)
            z_q, indices = fsq(z_e)
            recon_orig = model.decoder(z_q)[0, 0].cpu().numpy()

            for _ in range(args.n_positions):
                pos_r = rng.integers(1, 7)
                pos_c = rng.integers(1, 7)
                orig_coords = z_q[0, :, pos_r, pos_c].clone()

                for name, deltas in perturbations.items():
                    z_mod = z_q.clone()
                    clamped = False

                    for dim, delta in deltas:
                        new_val = orig_coords[dim].item() + delta
                        half = levels[dim] // 2
                        # Match FSQ valid range: odd L → [-half, half], even L → [-half, half-1]
                        max_val = half if levels[dim] % 2 == 1 else half - 1
                        min_val = -half
                        clamped_val = max(min_val, min(max_val, new_val))
                        if clamped_val != new_val:
                            clamped = True
                        z_mod[0, dim, pos_r, pos_c] = clamped_val

                    # Skip samples where any dimension was clamped (no-op)
                    if clamped:
                        continue

                    recon_mod = model.decoder(z_mod)[0, 0].cpu().numpy()
                    pr, pc = pos_r * 8, pos_c * 8
                    patch_mse = ((recon_orig[pr:pr+8, pc:pc+8] -
                                  recon_mod[pr:pr+8, pc:pc+8]) ** 2).mean()
                    results[name].append(patch_mse)

            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{args.n_frames} frames...")

    # Print summary table
    print(f"\n{'Perturbation':<20} {'Mean MSE':>10} {'Std MSE':>10} {'N':>6}")
    print("-" * 50)

    # Group and sort by category
    categories = [
        ("Single dim ±1", [k for k in results if k.startswith(("+1 d", "-1 d")) and "," not in k]),
        ("Single dim ±2", [k for k in results if k.startswith("+2 d") and "," not in k]),
        ("Two dims ±1", [k for k in results if k.startswith("+1 d") and k.count(",") == 1]),
        ("Two dims ±2", [k for k in results if k.startswith("+2 d") and "," in k]),
        ("Three dims ±1", [k for k in results if k.startswith("+1 d") and k.count(",") == 2]),
        ("All dims ±1", ["+1 all"]),
    ]

    for cat_name, keys in categories:
        print(f"\n  {cat_name}:")
        cat_values = []
        for k in sorted(keys):
            vals = np.array(results[k])
            cat_values.extend(vals)
            print(f"    {k:<18} {vals.mean():10.6f} {vals.std():10.6f} {len(vals):6d}")
        if len(keys) > 1:
            cat_arr = np.array(cat_values)
            print(f"    {'(avg)':<18} {cat_arr.mean():10.6f} {cat_arr.std():10.6f} {len(cat_arr):6d}")

    # Key comparison
    print("\n" + "=" * 50)
    print("Key comparisons:")

    single_1 = np.concatenate([np.array(results[k]) for k in results
                                if (k.startswith("+1 d") or k.startswith("-1 d")) and "," not in k])
    single_2 = np.concatenate([np.array(results[k]) for k in results
                                if k.startswith("+2 d") and "," not in k])
    two_dims_1 = np.concatenate([np.array(results[k]) for k in results
                                  if k.startswith("+1 d") and k.count(",") == 1])

    print(f"  ±1 single dim (avg):  {single_1.mean():.6f}")
    print(f"  +2 single dim (avg):  {single_2.mean():.6f}")
    print(f"  +1 two dims (avg):    {two_dims_1.mean():.6f}")
    print(f"  +2/+1 ratio:          {single_2.mean() / single_1.mean():.2f}x")
    print(f"  two_dims/single ratio: {two_dims_1.mean() / single_1.mean():.2f}x")
    print(f"  +2 vs +1×2 dims:     +2 is {single_2.mean() / two_dims_1.mean():.2f}x worse")


if __name__ == "__main__":
    main()
