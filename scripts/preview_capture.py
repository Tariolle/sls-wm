"""Live preview of the GD capture crop — verify alignment before recording.

Usage:
    python scripts/preview_capture.py

Shows two windows:
    - Full capture with crop rectangle overlay
    - Cropped + Sobel result (what the model sees, upscaled)

Press Q or ESC to quit.
"""

import cv2
import dxcam
import numpy as np

# Same crop constants as record_gameplay.py (1920x1080 fullscreen)
REGION = (0, 0, 1920, 1080)
CROP_X, CROP_Y, CROP_SIZE = 660, 48, 1032


def main():
    cam = dxcam.create()
    print(f"Capture: {REGION}, crop: {CROP_SIZE}x{CROP_SIZE} at ({CROP_X}, {CROP_Y})")
    print("Press Q or ESC to quit.")

    while True:
        img = cam.grab(region=REGION)
        if img is None:
            continue

        # Draw crop rectangle on full capture (scaled down for display)
        overlay = img.copy()
        cv2.rectangle(overlay,
                      (CROP_X, CROP_Y),
                      (CROP_X + CROP_SIZE, CROP_Y + CROP_SIZE),
                      (0, 255, 0), 2)
        overview = cv2.resize(overlay, (960, 540))
        cv2.imshow("Full capture (green = crop)", cv2.cvtColor(overview, cv2.COLOR_RGB2BGR))

        # Cropped + Sobel preview
        cropped = img[CROP_Y:CROP_Y + CROP_SIZE, CROP_X:CROP_X + CROP_SIZE]
        gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
        sx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
        edges = cv2.convertScaleAbs(cv2.magnitude(sx.astype(np.float32), sy.astype(np.float32)))
        small = cv2.resize(edges, (64, 64), interpolation=cv2.INTER_AREA)
        preview = cv2.resize(small, (384, 384), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Model input (64x64 Sobel)", preview)

        key = cv2.waitKey(16) & 0xFF
        if key in (ord('q'), 27):
            break

    del cam
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
