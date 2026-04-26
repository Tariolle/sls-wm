"""GUI visualizer: original video | Sobel input | FSQ reconstruction.

The original video runs at its native frame rate. Sobel and FSQ panels
update at 30 FPS to emulate real-time deployment behavior.

Controls: SPACE=pause/resume, Q/ESC=quit
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
    """Crop, Sobel, downscale."""
    cropped = frame[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size]
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    edges = cv2.convertScaleAbs(cv2.magnitude(
        sobel_x.astype(np.float32), sobel_y.astype(np.float32)))
    return cv2.resize(edges, (target_size, target_size),
                      interpolation=cv2.INTER_AREA)


def run_model(model, frame_gray_64, device):
    """Run FSQ-VAE encode+decode on a 64x64 grayscale frame."""
    tensor = torch.from_numpy(frame_gray_64).float().unsqueeze(0).unsqueeze(0) / 255.0
    tensor = tensor.to(device, non_blocking=True)
    with torch.no_grad():
        recon, _, _ = model(tensor)
    return (recon.squeeze().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize FSQ-VAE input/output alongside original video")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--checkpoint", default="checkpoints/fsq_best.pt")
    parser.add_argument("--levels", type=int, nargs="+", default=[8, 5, 5, 5])
    parser.add_argument("--crop-x", type=int, default=660)
    parser.add_argument("--crop-y", type=int, default=48)
    parser.add_argument("--crop-size", type=int, default=1032)
    parser.add_argument("--target-size", type=int, default=64)
    parser.add_argument("--display-height", type=int, default=400)
    parser.add_argument("--compile-mode", default="reduce-overhead",
                        choices=["none", "default", "reduce-overhead", "max-autotune"])
    args = parser.parse_args()

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
        warmup = np.zeros((args.target_size, args.target_size), dtype=np.uint8)
        for _ in range(3):
            run_model(model, warmup, device)
        print("Compile warmup done.")

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"{args.video}: {total} frames, {fps:.1f} FPS, {w}x{h}")
    print("Controls: SPACE=pause/resume, Q/ESC=quit")

    frame_time = 1.0 / fps if fps > 0 else 1.0 / 30
    dh = args.display_height if args.display_height > 0 else h
    scale = dh / h
    dw = int(w * scale)
    panel_size = dh

    paused = False
    model_interval = 1.0 / 30
    model_input_display = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)
    model_output_display = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)
    x, y, s = args.crop_x, args.crop_y, args.crop_size

    playback_start = time.perf_counter()
    last_model_time = playback_start - model_interval
    frame_idx = 0
    pause_started = 0.0
    frame = None

    while True:
        if not paused:
            now = time.perf_counter()
            target_idx = int((now - playback_start) * fps) if fps > 0 else frame_idx + 1

            new_frame = None
            while frame_idx <= target_idx:
                ret, f = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    playback_start = time.perf_counter()
                    last_model_time = playback_start - model_interval
                    frame_idx = 0
                    new_frame = None
                    break
                new_frame = f
                frame_idx += 1
            if new_frame is not None:
                frame = new_frame

            if frame is not None:
                display_frame = frame.copy()
                cv2.rectangle(display_frame, (x, y), (x + s, y + s), (0, 255, 0), 2)
                display_frame = cv2.resize(display_frame, (dw, dh))

                now = time.perf_counter()
                if now - last_model_time >= model_interval:
                    last_model_time = now
                    model_input = preprocess_frame(frame, x, y, s, args.target_size)
                    model_output = run_model(model, model_input, device)

                    input_up = cv2.resize(model_input, (panel_size, panel_size),
                                          interpolation=cv2.INTER_NEAREST)
                    output_up = cv2.resize(model_output, (panel_size, panel_size),
                                           interpolation=cv2.INTER_NEAREST)
                    model_input_display = cv2.cvtColor(input_up, cv2.COLOR_GRAY2BGR)
                    model_output_display = cv2.cvtColor(output_up, cv2.COLOR_GRAY2BGR)

                combined = np.hstack([display_frame, model_input_display,
                                      model_output_display])
                cv2.putText(combined, f"Original ({frame_idx}/{total})", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.putText(combined, "Sobel 64x64 (30 FPS)", (dw + 10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.putText(combined, "FSQ Reconstruction (30 FPS)",
                            (dw + panel_size + 10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.imshow("DeepDash FSQ Visualizer", combined)

            next_target = playback_start + (frame_idx + 1) * frame_time
            wait = max(1, int((next_target - time.perf_counter()) * 1000))
        else:
            wait = 30

        key = cv2.waitKey(wait) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):
            if not paused:
                pause_started = time.perf_counter()
                paused = True
            else:
                pause_dur = time.perf_counter() - pause_started
                playback_start += pause_dur
                last_model_time += pause_dur
                paused = False

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
