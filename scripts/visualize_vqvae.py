"""GUI visualizer: original video | model input | decoder output."""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


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
    parser = argparse.ArgumentParser(description="Visualize tokenizer input/output alongside original video")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--model", choices=["vqvae", "fsq"], default="fsq")
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint path (default: checkpoints/{model}_best.pt)")
    parser.add_argument("--levels", type=int, nargs="+", default=[8, 5, 5, 5])

    parser.add_argument("--crop-x", type=int, default=660)
    parser.add_argument("--crop-y", type=int, default=48)
    parser.add_argument("--crop-size", type=int, default=1032)
    parser.add_argument("--target-size", type=int, default=64)
    parser.add_argument("--display-height", type=int, default=400,
                        help="Height to scale the video panel to (0 = original size)")
    args = parser.parse_args()

    if args.checkpoint is None:
        args.checkpoint = f"checkpoints/{args.model}_best.pt"

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model == "fsq":
        from deepdash.fsq import FSQVAE
        model = FSQVAE(levels=args.levels).to(device)
    else:
        from deepdash.vqvae import VQVAE
        model = VQVAE().to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded {args.model.upper()} from {args.checkpoint} on {device}")

    # Open video
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"{args.video}: {total} frames, {fps:.1f} FPS, {w}x{h}")
    print("Controls: SPACE=pause/resume, Q/ESC=quit")

    frame_time = 1.0 / fps if fps > 0 else 1.0 / 30

    # Scale display to fit screen
    dh = args.display_height if args.display_height > 0 else h
    scale = dh / h
    dw = int(w * scale)
    panel_size = dh  # square panels for model input/output

    # Black placeholder for model panels before first sample
    model_input_display = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)
    model_output_display = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)

    frame_idx = 0
    paused = False

    while True:
        if not paused:
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                continue

            # Green crop rectangle on original (at original resolution)
            display_frame = frame.copy()
            x, y, s = args.crop_x, args.crop_y, args.crop_size
            cv2.rectangle(display_frame, (x, y), (x + s, y + s), (0, 255, 0), 2)

            # Scale down for display
            display_frame = cv2.resize(display_frame, (dw, dh))

            # Process every frame (matches inference behavior)
            model_input = preprocess_frame(frame, x, y, s, args.target_size)
            model_output = run_model(model, model_input, device)

            # Upscale with nearest neighbor for crisp pixels
            input_up = cv2.resize(model_input, (panel_size, panel_size), interpolation=cv2.INTER_NEAREST)
            output_up = cv2.resize(model_output, (panel_size, panel_size), interpolation=cv2.INTER_NEAREST)

            model_input_display = cv2.cvtColor(input_up, cv2.COLOR_GRAY2BGR)
            model_output_display = cv2.cvtColor(output_up, cv2.COLOR_GRAY2BGR)

            # Compose and label
            combined = np.hstack([display_frame, model_input_display, model_output_display])
            cv2.putText(combined, f"Original ({frame_idx}/{total})", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(combined, "Sobel 64x64", (dw + 10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(combined, "Decoder Output", (dw + panel_size + 10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

            cv2.imshow("DeepDash Visualizer", combined)
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
