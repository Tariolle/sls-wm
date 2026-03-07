"""GUI visualizer: original video | model input | decoder output."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepdash.vqvae import VQVAE


def preprocess_frame(frame, crop_x, crop_y, crop_size, target_size):
    """Crop, Sobel, downscale — same pipeline as extract_frames.py."""
    cropped = frame[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size]
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges = cv2.convertScaleAbs(cv2.magnitude(sobel_x, sobel_y))
    return cv2.resize(edges, (target_size, target_size), interpolation=cv2.INTER_AREA)


def run_model(model, frame_gray_64, device):
    """Run VQ-VAE encode+decode on a 64x64 grayscale frame."""
    tensor = torch.from_numpy(frame_gray_64).float().unsqueeze(0).unsqueeze(0) / 255.0
    tensor = tensor.to(device)
    with torch.no_grad():
        recon, _, _ = model(tensor)
    return (recon.squeeze().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Visualize VQ-VAE input/output alongside original video")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--checkpoint", default="checkpoints/vqvae_best.pt", help="VQ-VAE checkpoint")
    parser.add_argument("--every-n", type=int, default=5, help="Frame sampling rate (default: 5)")
    parser.add_argument("--crop-x", type=int, default=220)
    parser.add_argument("--crop-y", type=int, default=16)
    parser.add_argument("--crop-size", type=int, default=344)
    parser.add_argument("--target-size", type=int, default=64)
    args = parser.parse_args()

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VQVAE().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded model from {args.checkpoint} on {device}")

    # Open video
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"{args.video}: {total} frames, {fps:.1f} FPS, {w}x{h}")
    print("Controls: SPACE=pause/resume, Q/ESC=quit")

    delay_ms = max(1, int(1000 / fps)) if fps > 0 else 33
    panel_h = h  # side panels match video height (square)

    # Black placeholder for model panels before first sample
    model_input_display = np.zeros((panel_h, panel_h, 3), dtype=np.uint8)
    model_output_display = np.zeros((panel_h, panel_h, 3), dtype=np.uint8)

    frame_idx = 0
    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                continue

            # Green crop rectangle on original
            display_frame = frame.copy()
            x, y, s = args.crop_x, args.crop_y, args.crop_size
            cv2.rectangle(display_frame, (x, y), (x + s, y + s), (0, 255, 0), 2)

            # Update model panels on sampled frames
            if frame_idx % args.every_n == 0:
                model_input = preprocess_frame(frame, x, y, s, args.target_size)
                model_output = run_model(model, model_input, device)

                # Upscale with nearest neighbor for crisp pixels
                input_up = cv2.resize(model_input, (panel_h, panel_h), interpolation=cv2.INTER_NEAREST)
                output_up = cv2.resize(model_output, (panel_h, panel_h), interpolation=cv2.INTER_NEAREST)

                model_input_display = cv2.cvtColor(input_up, cv2.COLOR_GRAY2BGR)
                model_output_display = cv2.cvtColor(output_up, cv2.COLOR_GRAY2BGR)

            # Compose and label
            combined = np.hstack([display_frame, model_input_display, model_output_display])
            cv2.putText(combined, f"Original ({frame_idx}/{total})", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(combined, "Model Input (Sobel 64x64)", (w + 10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(combined, "Decoder Output", (w + panel_h + 10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

            cv2.imshow("DeepDash Visualizer", combined)
            frame_idx += 1

        key = cv2.waitKey(delay_ms) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
