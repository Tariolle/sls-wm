"""Split extracted frames into train/val sets."""

import os
import random
import shutil
import argparse
from pathlib import Path


def split_dataset(frames_dir: str, train_ratio: float = 0.9, seed: int = 42):
    frames_dir = Path(frames_dir)
    train_dir = frames_dir.parent / "train"
    val_dir = frames_dir.parent / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(frames_dir.glob("*.png"))
    random.seed(seed)
    random.shuffle(frames)

    split_idx = int(len(frames) * train_ratio)
    train_frames = frames[:split_idx]
    val_frames = frames[split_idx:]

    for f in train_frames:
        shutil.move(str(f), str(train_dir / f.name))
    for f in val_frames:
        shutil.move(str(f), str(val_dir / f.name))

    print(f"Train: {len(train_frames)} frames")
    print(f"Val:   {len(val_frames)} frames")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split frames into train/val")
    parser.add_argument("--frames-dir", default="data/frames", help="Directory with extracted frames")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Train split ratio (default: 0.9)")
    args = parser.parse_args()

    split_dataset(args.frames_dir, train_ratio=args.train_ratio)
