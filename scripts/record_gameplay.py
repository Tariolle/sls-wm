"""Record Geometry Dash gameplay: screen capture + keyboard input + death detection.

Fully automated recording — press F5 once, then just play. Episodes are saved
on each death (respawn noise skipped). Auto-resumes after respawn.
Press F5 again to stop (trims last --trim-end seconds to remove win animation
or other end noise).

Usage:
    python scripts/record_gameplay.py --custom

Controls:
    F5  — start/stop auto-recording (stop trims last --trim-end seconds)
    F6  — manual episode split (saves current, starts fresh)
    F10 — quit (saves current episode if recording)

Output:
    data/death_episodes/ep_NNNN/
        frames.npy     (T, 64, 64) uint8 — Sobel edge maps
        actions.npy    (T,) uint8 — 0=idle, 1=jump
        metadata.json  {fps_target, fps_actual, timestamp, num_frames}
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import ctypes
import ctypes.wintypes as wt
import cv2
import dxcam
import keyboard
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from deepdash.gd_mem import GDReader


# Win32 helpers
_user32 = ctypes.windll.user32
_user32.FindWindowW.argtypes = [wt.LPCWSTR, wt.LPCWSTR]
_user32.FindWindowW.restype = wt.HWND
_user32.SetWindowPos.argtypes = [
    wt.HWND, wt.HWND, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_uint,
]
_user32.SetWindowPos.restype = wt.BOOL

_HWND_TOPMOST = wt.HWND(-1)
_SWP_NOACTIVATE = 0x0010


def _force_topmost(window_title: str):
    """Force a window to stay on top using Win32 SetWindowPos (every frame)."""
    hwnd = _user32.FindWindowW(None, window_title)
    if hwnd:
        _user32.SetWindowPos(
            hwnd, _HWND_TOPMOST, 0, 0, 0, 0,
            0x0002 | 0x0001 | _SWP_NOACTIVATE)  # NOMOVE | NOSIZE


def preprocess_frame(rgb: np.ndarray, crop_x: int, crop_y: int,
                     crop_size: int, target_size: int) -> np.ndarray:
    """RGB screenshot -> 64x64 Sobel edge map (uint8).

    1. Crop crop_size x crop_size at (crop_x, crop_y)
    2. Sobel edge detection at full resolution (1032x1032 for 1080p)
    3. Resize to target_size x target_size with INTER_AREA
    """
    cropped = rgb[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size]
    gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    edges = cv2.convertScaleAbs(cv2.magnitude(
        sobel_x.astype(np.float32), sobel_y.astype(np.float32)))
    resized = cv2.resize(edges, (target_size, target_size),
                         interpolation=cv2.INTER_AREA)
    return resized


def save_episode(episode_dir: Path, frames: list, actions: list,
                 fps_target: int, t_start: float, t_end: float,
                 level: int = 0):
    """Save a single episode to disk."""
    if not frames:
        return
    episode_dir.mkdir(parents=True, exist_ok=True)
    np.save(episode_dir / "frames.npy", np.array(frames, dtype=np.uint8))
    np.save(episode_dir / "actions.npy", np.array(actions, dtype=np.uint8))
    num_frames = len(frames)
    duration = t_end - t_start
    fps_actual = num_frames / duration if duration > 0 else 0
    metadata = {
        "level": level,
        "fps_target": fps_target,
        "fps_actual": round(fps_actual, 2),
        "num_frames": num_frames,
        "duration_s": round(duration, 2),
        "timestamp": datetime.now().isoformat(),
    }
    with open(episode_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved {episode_dir.name}: {num_frames} frames, "
          f"{duration:.1f}s, {fps_actual:.1f} FPS actual")


def next_episode_id(episodes_dir: Path, user_id: int = 0) -> int:
    """Find the next available episode number within a user's range.

    User 0: ep_0001–ep_9999, User 1: ep_10001–ep_19999, etc.
    """
    range_start = user_id * 10000 + 1
    range_end = range_start + 9999
    max_id = range_start - 1
    for ep in episodes_dir.glob("ep_*"):
        ep_id = int(ep.name.split("_")[1])
        if range_start <= ep_id <= range_end and ep_id > max_id:
            max_id = ep_id
    return max_id + 1


# Auto-record states
STATE_IDLE = "IDLE"           # F5 not pressed yet
STATE_RECORDING = "REC"       # Actively capturing frames
STATE_WAIT_RESPAWN = "WAIT"   # Dead, waiting for respawn
STATE_WAIT_ATTEMPT = "DELAY"  # Respawned, waiting for "ATTEMPT" to clear


def main():
    parser = argparse.ArgumentParser(
        description="Record Geometry Dash gameplay for training data")
    parser.add_argument("--user", type=int, default=0,
                        help="User ID for episode numbering offset "
                             "(0 = ep_0001-9999, 1 = ep_10001-19999, ...)")
    parser.add_argument("--target-size", type=int, default=64,
                        help="Output frame size (default: 64)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Target capture FPS (default: 30)")
    parser.add_argument("--jump-key", default="space",
                        help="Key used for jumping (default: space)")
    level_group = parser.add_mutually_exclusive_group(required=True)
    level_group.add_argument("--level", type=int,
                             help="Level number being recorded (saved in metadata)")
    level_group.add_argument("--custom", action="store_true",
                             help="Auto-assign level = max existing level ID + 1 "
                                  "(for custom/community levels)")
    level_group.add_argument("--perfect-run", type=int, metavar="LEVEL_ID",
                             help="Record a perfect run (no deaths) for given level. "
                                  "Auto-deletes on death. Saved to data/expert_episodes/")
    parser.add_argument("--output-dir", default="data/death_episodes",
                        help="Output directory for episodes")
    parser.add_argument("--respawn-delay", type=float, default=0.75,
                        help="Seconds to wait after respawn before recording "
                             "(skip ATTEMPT overlay, default: 0.7)")
    parser.add_argument("--trim-end", type=float, default=0.5,
                        help="Seconds to trim from end when stopping with F5 "
                             "(remove win animation noise, default: 0.5)")
    args = parser.parse_args()

    # Resolve level ID and output directory
    perfect_run_mode = args.perfect_run is not None
    if perfect_run_mode:
        args.level = args.perfect_run
        args.output_dir = "data/expert_episodes"
        print(f"Perfect run mode: level {args.perfect_run} "
              f"(recording deleted on death)")
    elif args.custom:
        args.level = 0
        print("Custom level mode: using level ID 0")

    # Connect to GD process for death detection
    print("Connecting to Geometry Dash process...")
    try:
        gd = GDReader()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        print("Make sure Geometry Dash is running.")
        return

    # Capture full screen (GD should be in fullscreen mode)
    region = (0, 0, 1920, 1080)
    crop_x, crop_y, crop_size = 660, 48, 1032
    print(f"  Capture: {region}, crop: {crop_size}x{crop_size} at ({crop_x}, {crop_y})")

    episodes_dir = Path(args.output_dir)
    episodes_dir.mkdir(parents=True, exist_ok=True)
    if perfect_run_mode:
        episode_id = args.perfect_run
        part_counter = 1
        ep_name = lambda: f"perfect_run_{episode_id}_{part_counter}"
    else:
        episode_id = next_episode_id(episodes_dir, args.user)
        ep_name = lambda: f"ep_{episode_id:04d}"

    # DXGI hardware-accelerated capture
    cam = dxcam.create()

    frame_interval = 1.0 / args.fps
    state = STATE_IDLE
    auto_mode = False  # True after F5 pressed
    frames = []
    actions = []
    t_episode_start = 0.0
    t_respawn = 0.0
    preview_topmost_set = False

    print(f"Target FPS: {args.fps}")
    print(f"Jump key: {args.jump_key}")
    print(f"Episodes dir: {episodes_dir}")
    print(f"Respawn delay: {args.respawn_delay}s")
    print(f"F5 stop trim: {args.trim_end}s")
    print()
    print("Controls:")
    print("  F5  — start/stop auto-recording (stop trims last "
          f"{args.trim_end}s)")
    print("  F6  — manual episode split")
    print("  F10 — quit")
    print()
    print("Auto-record: death splits + respawn skip")
    print("Preview window open — verify capture region is correct.")

    while True:
        t_loop_start = time.perf_counter()

        # Read game state every frame (cheap memory read)
        gd_state = gd.get_state()
        in_level = gd_state["in_level"]
        is_dead = gd_state["is_dead"]

        # --- State machine transitions ---
        if auto_mode:
            if state == STATE_RECORDING:
                if is_dead:
                    if perfect_run_mode:
                        # Perfect run — trim last 30 frames (death region), then save
                        trim = min(30, len(frames))
                        if trim:
                            frames = frames[:-trim]
                            actions = actions[:-trim]
                        t_end = time.perf_counter()
                        if frames:
                            ep_dir = episodes_dir / ep_name()
                            save_episode(ep_dir, frames, actions,
                                         args.fps, t_episode_start, t_end,
                                         args.level)
                            print(f">> Death detected. Trimmed {trim} frames, "
                                  f"saved {len(frames)}.")
                            part_counter += 1
                        else:
                            print(f">> Death detected. Nothing to save after trim.")
                        frames = []
                        actions = []
                        state = STATE_WAIT_RESPAWN
                    else:
                        # Death detected — save episode, wait for respawn
                        t_end = time.perf_counter()
                        ep_dir = episodes_dir / ep_name()
                        save_episode(ep_dir, frames, actions,
                                     args.fps, t_episode_start, t_end,
                                     args.level)
                        episode_id += 1
                        frames = []
                        actions = []
                        state = STATE_WAIT_RESPAWN
                        print(f">> Death detected. Waiting for respawn...")

            elif state == STATE_WAIT_RESPAWN:
                if not is_dead and in_level:
                    # Player respawned — start delay timer
                    t_respawn = time.perf_counter()
                    state = STATE_WAIT_ATTEMPT
                    print(f">> Respawned. Skipping ATTEMPT overlay "
                          f"({args.respawn_delay}s)...")

            elif state == STATE_WAIT_ATTEMPT:
                if is_dead:
                    # Died again during delay — restart wait
                    state = STATE_WAIT_RESPAWN
                elif time.perf_counter() - t_respawn >= args.respawn_delay:
                    # Delay done — start recording
                    frames = []
                    actions = []
                    t_episode_start = time.perf_counter()
                    state = STATE_RECORDING
                    print(f">> Recording {ep_name()}...")

        # --- Capture + preprocess ---
        img = cam.grab(region=region)
        if img is not None:
            edge_frame = preprocess_frame(
                img, crop_x, crop_y,
                crop_size, args.target_size)

            # --- Record frame + action if recording ---
            if state == STATE_RECORDING:
                jump = keyboard.is_pressed(args.jump_key)
                frames.append(edge_frame)
                actions.append(1 if jump else 0)

            # --- Show preview ---
            preview = cv2.resize(edge_frame, (256, 256),
                                 interpolation=cv2.INTER_NEAREST)
            color_map = {
                STATE_IDLE: (128, 128, 128),
                STATE_RECORDING: (0, 0, 255),
                STATE_WAIT_RESPAWN: (0, 165, 255),
                STATE_WAIT_ATTEMPT: (0, 255, 255),
            }
            color = color_map.get(state, (128, 128, 128))
            preview_bgr = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
            cv2.putText(preview_bgr, state, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            if state == STATE_RECORDING:
                cv2.putText(preview_bgr,
                            f"{ep_name()}  {len(frames)} frames",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 1)
            cv2.imshow("DeepDash Recorder", preview_bgr)
            _force_topmost("DeepDash Recorder")

        # --- Handle OpenCV key events ---
        cv2.waitKey(1)

        # --- Hotkeys ---
        if keyboard.is_pressed("f5"):
            if not auto_mode:
                auto_mode = True
                if in_level and not is_dead:
                    frames = []
                    actions = []
                    t_episode_start = time.perf_counter()
                    state = STATE_RECORDING
                    print(f"\n>> Auto-record ON. Recording "
                          f"{ep_name()}...")
                elif in_level and is_dead:
                    state = STATE_WAIT_RESPAWN
                    print(f"\n>> Auto-record ON. Waiting for respawn...")
                else:
                    state = STATE_WAIT_RESPAWN
                    print(f"\n>> Auto-record ON. Waiting for level...")
            else:
                # Stop — trim end and save current episode
                auto_mode = False
                if state == STATE_RECORDING and frames:
                    trim_frames = int(args.trim_end * args.fps)
                    if trim_frames > 0 and len(frames) > trim_frames:
                        frames = frames[:-trim_frames]
                        actions = actions[:-trim_frames]
                    t_end = time.perf_counter() - args.trim_end
                    ep_dir = episodes_dir / ep_name()
                    save_episode(ep_dir, frames, actions,
                                 args.fps, t_episode_start, t_end,
                                 args.level)
                    if perfect_run_mode:
                        part_counter += 1
                    else:
                        episode_id += 1
                    print(f">> Stopped. Trimmed last {args.trim_end}s.")
                else:
                    print(">> Stopped.")
                frames = []
                actions = []
                state = STATE_IDLE
            while keyboard.is_pressed("f5"):
                time.sleep(0.01)

        if keyboard.is_pressed("f6"):
            if state == STATE_RECORDING and frames:
                t_end = time.perf_counter()
                ep_dir = episodes_dir / ep_name()
                save_episode(ep_dir, frames, actions,
                             args.fps, t_episode_start, t_end)
                if not perfect_run_mode:
                    episode_id += 1
                frames = []
                actions = []
                t_episode_start = time.perf_counter()
                print(f"\n>> Manual split. Recording "
                      f"{ep_name()}...")
            while keyboard.is_pressed("f6"):
                time.sleep(0.01)

        if keyboard.is_pressed("f10"):
            if state == STATE_RECORDING and frames:
                t_end = time.perf_counter()
                ep_dir = episodes_dir / ep_name()
                save_episode(ep_dir, frames, actions,
                             args.fps, t_episode_start, t_end)
                print(">> Saved final episode on exit.")
            break

        # --- Frame rate limiting ---
        elapsed = time.perf_counter() - t_loop_start
        sleep_time = frame_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    del cam
    gd.close()
    cv2.destroyAllWindows()
    print("\nDone.")


if __name__ == "__main__":
    main()
