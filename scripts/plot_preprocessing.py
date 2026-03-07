"""Plot the preprocessing pipeline stages for README images."""

import cv2
import argparse
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="data/videos/Standard/level_2.mp4")
    parser.add_argument("--frame", type=int, default=2104, help="Frame index to extract")
    parser.add_argument("--crop-x", type=int, default=220)
    parser.add_argument("--crop-y", type=int, default=16)
    parser.add_argument("--crop-size", type=int, default=344)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"Failed to read frame {args.frame}")
        return

    # Stage 1: Raw frame (BGR -> RGB for matplotlib)
    raw_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Stage 2: Square crop
    cropped = frame[args.crop_y:args.crop_y + args.crop_size,
                     args.crop_x:args.crop_x + args.crop_size]
    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

    # Stage 3: Sobel edge detection at full crop resolution
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges = cv2.convertScaleAbs(cv2.magnitude(sobel_x, sobel_y))

    # Stage 4: Downscale to 64x64
    downscaled = cv2.resize(edges, (64, 64), interpolation=cv2.INTER_AREA)

    # Plot all stages
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    axes[0].imshow(raw_rgb)
    axes[0].set_title(f"Raw Frame (640x360)", fontsize=13)
    axes[0].axis("off")

    axes[1].imshow(cropped_rgb)
    axes[1].set_title(f"Square Crop (344x344)", fontsize=13)
    axes[1].axis("off")

    axes[2].imshow(edges, cmap="gray")
    axes[2].set_title(f"Sobel Edges (344x344)", fontsize=13)
    axes[2].axis("off")

    axes[3].imshow(downscaled, cmap="gray", interpolation="nearest")
    axes[3].set_title(f"Final Input (64x64)", fontsize=13)
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()

    # Save individual images
    cv2.imwrite("docs/preprocessing_1_original.png", frame)
    cv2.imwrite("docs/preprocessing_2_cropped.png", cropped)
    cv2.imwrite("docs/preprocessing_3_sobel.png", edges)
    cv2.imwrite("docs/preprocessing_4_final.png", downscaled)
    print("Saved 4 images to docs/")


if __name__ == "__main__":
    main()
