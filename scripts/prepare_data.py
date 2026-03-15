"""Prepare training data: concatenate episodes, shuffle, split into train/val.

Reads all frames.npy from data/death_episodes/*, concatenates into a single array,
shuffles, and splits into data/train.npy and data/val.npy.

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --episodes-dir data/death_episodes --train-ratio 0.9
"""

import argparse
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Prepare train/val splits from episodes")
    parser.add_argument("--episodes-dir", default="data/death_episodes",
                        help="Directory containing ep_* folders")
    parser.add_argument("--output-dir", default="data",
                        help="Output directory for train.npy and val.npy")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    episodes_dir = Path(args.episodes_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all episode frames
    all_frames = []
    for ep in sorted(episodes_dir.glob("*")):
        fp = ep / "frames.npy"
        if not fp.exists():
            continue
        frames = np.load(fp)
        all_frames.append(frames)
        print(f"  {ep.name}: {frames.shape[0]} frames")

    if not all_frames:
        print("No episodes found!")
        return

    all_frames = np.concatenate(all_frames, axis=0)
    print(f"\nTotal: {all_frames.shape[0]} frames, shape {all_frames.shape}")

    # Shuffle
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(all_frames))
    all_frames = all_frames[indices]

    # Split
    split_idx = int(len(all_frames) * args.train_ratio)
    train = all_frames[:split_idx]
    val = all_frames[split_idx:]

    np.save(output_dir / "train.npy", train)
    np.save(output_dir / "val.npy", val)
    print(f"Train: {len(train)} frames -> {output_dir / 'train.npy'}")
    print(f"Val:   {len(val)} frames -> {output_dir / 'val.npy'}")


if __name__ == "__main__":
    main()
