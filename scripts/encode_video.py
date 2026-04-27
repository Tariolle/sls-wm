"""Encode a video through the FSQ-VAE pipeline.

Reads a video at its native frame rate, samples it down to 30 Hz on the
timeline (each output frame is the input frame current at i/30 seconds,
matching deploy's encoder rate), runs the same preprocessing as deploy
(crop -> grayscale -> Sobel -> 64x64 resize), passes through FSQ encode
+ decode, and writes a 30 FPS video whose total duration matches the
input.

Usage:
    python scripts/encode_video.py "D:/path/in.mp4" --checkpoint checkpoints_v7/fsq_best.pt
    python scripts/encode_video.py in.mp4 --output out.mp4 --output-size 512
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def preprocess_frame(frame, crop_x, crop_y, crop_size, target_size):
    """Crop -> grayscale -> Sobel -> resize. Same as visualize_fsq / deploy."""
    cropped = frame[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size]
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    edges = cv2.convertScaleAbs(cv2.magnitude(
        sobel_x.astype(np.float32), sobel_y.astype(np.float32)))
    return cv2.resize(edges, (target_size, target_size),
                      interpolation=cv2.INTER_AREA)


def encode_batch(model, frames_uint8, device):
    """frames_uint8: (B, target, target) -> reconstruction (B, target, target) uint8."""
    arr = np.stack(frames_uint8).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).unsqueeze(1).to(device, non_blocking=True)
    with torch.no_grad():
        recon, _, _ = model(t)
    return (recon.squeeze(1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(
        description="Encode a video through the FSQ-VAE pipeline at 30 FPS")
    parser.add_argument("video", help="Path to input video")
    parser.add_argument("--output", default=None,
                        help="Output path. Default: <video>.encoded.mp4")
    parser.add_argument("--checkpoint", default="checkpoints_v7/fsq_best.pt")
    parser.add_argument("--levels", type=int, nargs="+", default=[8, 5, 5, 5])
    parser.add_argument("--crop-x", type=int, default=660)
    parser.add_argument("--crop-y", type=int, default=48)
    parser.add_argument("--crop-size", type=int, default=1032)
    parser.add_argument("--target-size", type=int, default=64,
                        help="FSQ input resolution.")
    parser.add_argument("--output-size", type=int, default=512,
                        help="Output frame resolution (NEAREST upscale "
                             "preserves the FSQ pixel look). Set equal to "
                             "--target-size for raw 64x64 output.")
    parser.add_argument("--output-fps", type=float, default=30.0,
                        help="Encoder rate. Output total duration matches "
                             "the input regardless.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--compile-mode", default="reduce-overhead",
                        choices=["none", "default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--codec", default="mp4v",
                        help="VideoWriter fourcc (mp4v, avc1, ...)")
    args = parser.parse_args()

    in_path = Path(args.video)
    if not in_path.exists():
        print(f"Input not found: {in_path}", file=sys.stderr)
        sys.exit(1)
    out_path = Path(args.output) if args.output else \
        in_path.with_suffix(".encoded" + in_path.suffix)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from deepdash.fsq import FSQVAE
    model = FSQVAE(levels=args.levels).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded FSQ-VAE from {args.checkpoint} on {device}")

    if args.compile_mode != "none" and device.type == "cuda":
        print(f"Compiling FSQ-VAE (mode={args.compile_mode}) ...")
        model = torch.compile(model, mode=args.compile_mode)
        warmup = [np.zeros((args.target_size, args.target_size), dtype=np.uint8)
                  for _ in range(args.batch_size)]
        for _ in range(2):
            encode_batch(model, warmup, device)
        # Also warmup the smaller-batch path for the final partial batch.
        encode_batch(model, warmup[:1], device)
        print("Compile warmup done.")

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        print(f"Could not open {in_path}", file=sys.stderr)
        sys.exit(1)

    fps_in = cap.get(cv2.CAP_PROP_FPS)
    total_in = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps_in <= 0 or total_in <= 0:
        print(f"Bad input metadata: fps={fps_in} frames={total_in}", file=sys.stderr)
        sys.exit(1)
    duration = total_in / fps_in
    n_out = int(round(duration * args.output_fps))
    print(f"Input:  {total_in} frames @ {fps_in:.3f} fps ({duration:.2f}s)")
    print(f"Output: {n_out} frames @ {args.output_fps:.1f} fps "
          f"-> {n_out / args.output_fps:.2f}s "
          f"(write target: {out_path})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    writer = cv2.VideoWriter(str(out_path), fourcc, args.output_fps,
                             (args.output_size, args.output_size), isColor=True)
    if not writer.isOpened():
        print(f"VideoWriter failed to open ({args.codec}). "
              f"Try --codec avc1 or .avi extension.", file=sys.stderr)
        sys.exit(1)

    upscale_size = (args.output_size, args.output_size)

    buffered_inputs = []
    read_count = 0
    last_frame = None
    written = 0
    t_start = time.perf_counter()

    def flush():
        nonlocal written
        if not buffered_inputs:
            return
        recon = encode_batch(model, buffered_inputs, device)
        for img in recon:
            up = cv2.resize(img, upscale_size, interpolation=cv2.INTER_NEAREST)
            bgr = cv2.cvtColor(up, cv2.COLOR_GRAY2BGR)
            writer.write(bgr)
            written += 1
        buffered_inputs.clear()

    for i in range(n_out):
        target_idx = int(round(i * fps_in / args.output_fps))
        # read input frames until we have the one current at this timestamp
        while read_count <= target_idx:
            ret, frame = cap.read()
            if not ret:
                break
            last_frame = frame
            read_count += 1
        if last_frame is None:
            break
        edge = preprocess_frame(last_frame, args.crop_x, args.crop_y,
                                args.crop_size, args.target_size)
        buffered_inputs.append(edge)
        if len(buffered_inputs) >= args.batch_size:
            flush()
            elapsed = time.perf_counter() - t_start
            rate = written / max(elapsed, 1e-6)
            eta = (n_out - written) / max(rate, 1e-6)
            print(f"  {written}/{n_out} frames "
                  f"({100 * written / n_out:.1f}%, {rate:.0f} fps, "
                  f"ETA {eta:.0f}s)")
    flush()

    cap.release()
    writer.release()
    elapsed = time.perf_counter() - t_start
    print(f"\nDone. Wrote {written} frames in {elapsed:.1f}s "
          f"({written / max(elapsed, 1e-6):.0f} fps avg) -> {out_path}")


if __name__ == "__main__":
    main()
