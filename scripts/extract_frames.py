"""Extract, crop, and resize frames from gameplay videos for VAE training."""

import cv2
import os
import argparse
from pathlib import Path


def extract_frames(video_dir: str, output_dir: str, every_n: int = 5,
                    crop_x: int = 220, crop_y: int = 16, crop_size: int = 344,
                    target_size: int = 64, levels: list = None):
    """Extract square crops from 640x360 gameplay videos.

    Default crop: 344x344 square at (220, 16) — bottom-aligned (skips progress bar),
    player flush to the left edge, maximizing forward visibility.
    Downscaled to 64x64 RGB with area interpolation.
    levels: if provided, only extract from these level numbers (e.g. [1, 2, 3]).
    """
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(video_dir.glob("*.mp4"))
    if levels:
        level_names = {f"level_{l}" for l in levels}
        videos = [v for v in videos if v.stem.split("_part")[0] in level_names]
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
                # Crop 344x344 square (player at left edge, no progress bar)
                cropped = frame[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size]
                # Downscale to 64x64 with area interpolation (preserves thin structures)
                resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_AREA)
                # Convert to grayscale + Sobel edge detection (ksize=3)
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                resized = cv2.convertScaleAbs(cv2.magnitude(sobel_x, sobel_y))
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
    parser.add_argument("--video-dir", default="data/videos/Standard", help="Directory with .mp4 files")
    parser.add_argument("--output-dir", default="data/frames", help="Output directory for frames")
    parser.add_argument("--every-n", type=int, default=5, help="Sample every Nth frame (default: 5)")
    parser.add_argument("--levels", type=int, nargs="+", default=None, help="Level numbers to extract (e.g. --levels 1 2 3)")
    args = parser.parse_args()

    extract_frames(args.video_dir, args.output_dir, every_n=args.every_n, levels=args.levels)
