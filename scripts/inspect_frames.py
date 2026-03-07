"""Inspect individual frames from gameplay videos after cropping."""

import argparse
import cv2


def main():
    parser = argparse.ArgumentParser(description="Inspect cropped frames from a video")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("frame", nargs="?", type=int, default=None, help="Frame number to show")
    parser.add_argument("--crop-x", type=int, default=220)
    parser.add_argument("--crop-y", type=int, default=16)
    parser.add_argument("--crop-size", type=int, default=344)
    args = parser.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"{args.video}: {total} frames, {fps:.1f} FPS, {total/fps:.1f}s")

    if args.frame is None:
        cap.release()
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"Could not read frame {args.frame}")
        return

    x, y, s = args.crop_x, args.crop_y, args.crop_size
    crop = frame[y:y+s, x:x+s]

    cv2.imshow(f"Frame {args.frame}/{total}", crop)
    print(f"Showing frame {args.frame}/{total}. Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
