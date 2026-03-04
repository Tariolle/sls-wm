"""Extract, crop, and resize frames from gameplay videos for VAE training."""

import cv2
import os
import argparse
import numpy as np
from pathlib import Path


def extract_frames(video_dir: str, output_dir: str, every_n: int = 5, crop_top: int = 18, target_size: tuple = (176, 96)):
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(video_dir.glob("*.mp4"))
    total_saved = 0

    for video_path in videos:
        level_name = video_path.stem
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"{level_name}: {frame_count} frames @ {fps:.0f} FPS")

        frame_idx = 0
        saved = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % every_n == 0:
                # Crop top UI bar
                cropped = frame[crop_top:, :, :]
                # Resize to target
                resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
                # Save as PNG
                filename = f"{level_name}_{frame_idx:06d}.png"
                cv2.imwrite(str(output_dir / filename), resized)
                saved += 1
            frame_idx += 1

        cap.release()
        total_saved += saved
        print(f"  saved {saved} frames")

    print(f"\nTotal: {total_saved} frames saved to {output_dir}")
    return total_saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from gameplay videos")
    parser.add_argument("--video-dir", default="data/videos", help="Directory with .mp4 files")
    parser.add_argument("--output-dir", default="data/frames", help="Output directory for frames")
    parser.add_argument("--every-n", type=int, default=5, help="Sample every Nth frame (default: 5)")
    args = parser.parse_args()

    extract_frames(args.video_dir, args.output_dir, every_n=args.every_n)
