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
    tensor = tensor.to(device)
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
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from deepdash.fsq import FSQVAE
    model = FSQVAE(levels=args.levels).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded FSQ-VAE from {args.checkpoint} on {device}")

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

    frame_idx = 0
    paused = False
    model_interval = 1.0 / 30
    last_model_time = 0
    model_input_display = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)
    model_output_display = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)

    while True:
        if not paused:
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                last_model_time = 0
                continue

            display_frame = frame.copy()
            x, y, s = args.crop_x, args.crop_y, args.crop_size
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
            frame_idx += 1

            elapsed = time.perf_counter() - t0
            wait = max(1, int((frame_time - elapsed) * 1000))
        else:
            wait = 30

        key = cv2.waitKey(wait) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
