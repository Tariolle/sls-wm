"""Automated real-game evaluation: run N episodes and collect statistics.

Runs the full deploy pipeline (screen capture -> FSQ -> Transformer -> Controller)
on the real game, detects death via memory reading, and logs survival stats.

The game must be running and the player must be in a level.
After each death, the game auto-respawns -- the script waits for respawn
and starts the next episode automatically.

Usage:
    python scripts/eval_real_game.py --n-runs 100
    python scripts/eval_real_game.py --n-runs 50 --output eval_results.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import dxcam
import keyboard
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepdash.fsq import FSQVAE
from deepdash.world_model import WorldModel
from deepdash.controller import MLPPolicy
from deepdash.gd_mem import GDMemReader


def preprocess_frame(rgb, crop_x, crop_y, crop_size, target_size, device=None):
    """RGB screenshot -> 64x64 Sobel edge map (uint8). GPU Sobel + CPU resize."""
    cropped = rgb[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size]
    gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)

    if device is not None and device.type == "cuda":
        gray_t = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0).to(device)
        padded = torch.nn.functional.pad(gray_t, (1, 1, 1, 1), mode='reflect')
        sx_k = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                            dtype=torch.float32, device=device).reshape(1, 1, 3, 3)
        sy_k = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                            dtype=torch.float32, device=device).reshape(1, 1, 3, 3)
        sx = torch.nn.functional.conv2d(padded, sx_k)
        sy = torch.nn.functional.conv2d(padded, sy_k)
        mag = torch.sqrt(sx ** 2 + sy ** 2)
        edges = torch.clamp(torch.round(mag), 0, 255).to(torch.uint8)
        edges = edges.squeeze().cpu().numpy()
    else:
        sobel_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
        edges = cv2.convertScaleAbs(cv2.magnitude(
            sobel_x.astype(np.float32), sobel_y.astype(np.float32)))

    return cv2.resize(edges, (target_size, target_size),
                      interpolation=cv2.INTER_AREA)


def main():
    parser = argparse.ArgumentParser(description="Automated real-game evaluation")
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--vae-checkpoint", default="checkpoints/fsq_best.pt")
    parser.add_argument("--transformer-checkpoint",
                        default="checkpoints/transformer_best.pt")
    parser.add_argument("--controller-checkpoint",
                        default="checkpoints/controller_ppo_best.pt")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--jump-threshold", type=float, default=0.5)
    parser.add_argument("--config", default=None)
    parser.add_argument("--levels", type=int, nargs="+", default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--tokens-per-frame", type=int, default=None)
    parser.add_argument("--context-frames", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--output", default="eval_results.json")
    args = parser.parse_args()

    from deepdash.config import apply_config
    apply_config(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    K = args.context_frames

    # Load models
    print("Loading models...")
    vae = FSQVAE(levels=args.levels).to(device)
    state = torch.load(args.vae_checkpoint, map_location=device, weights_only=True)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    vae.load_state_dict(state)
    vae.eval()

    wm = WorldModel(
        vocab_size=args.vocab_size, embed_dim=args.embed_dim,
        n_heads=args.n_heads, n_layers=args.n_layers,
        context_frames=args.context_frames, dropout=args.dropout,
        tokens_per_frame=args.tokens_per_frame,
        adaln=getattr(args, 'adaln', False),
        fsq_dim=len(args.levels) if getattr(args, 'levels', None) else None,
    ).to(device)
    state = torch.load(args.transformer_checkpoint, map_location=device,
                       weights_only=True)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    wm.load_state_dict(state, strict=False)
    wm.eval()

    controller = MLPPolicy(h_dim=args.embed_dim).to(device)
    state = torch.load(args.controller_checkpoint, map_location=device,
                       weights_only=True)
    controller.load_state_dict(state)
    controller.eval()
    print("All models loaded.")

    # Screen capture setup
    region = (0, 0, 1920, 1080)
    crop_x, crop_y, crop_size = 660, 48, 1032
    cam = dxcam.create()
    frame_interval = 1.0 / args.fps

    # Memory reader
    gd = GDMemReader()
    print(f"GD memory reader connected (PID: {gd.pid})")

    results = []
    print(f"\nRunning {args.n_runs} evaluation episodes...")
    print("Press F10 to abort.\n")

    run_idx = 0
    while run_idx < args.n_runs:
        if keyboard.is_pressed("f10"):
            print("Aborted by user.")
            break

        # Wait for player to be alive in level
        while True:
            state_dict = gd.get_state()
            if state_dict["in_level"] and not state_dict["is_dead"]:
                break
            time.sleep(0.05)

        # Run one episode
        ctx_tokens = []
        ctx_actions = []
        jumping = False
        frames_survived = 0
        t_start = time.perf_counter()

        while True:
            t0 = time.perf_counter()

            if keyboard.is_pressed("f10"):
                break

            # Check death
            if gd.is_dead():
                if jumping:
                    keyboard.release("space")
                    jumping = False
                break

            # Capture
            img = cam.grab(region=region)
            if img is None:
                time.sleep(0.001)
                continue

            # Preprocess
            edge_frame = preprocess_frame(img, crop_x, crop_y, crop_size, 64, device)

            # FSQ encode
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                frame_t = torch.from_numpy(edge_frame.astype(np.float32) / 255.0)
                frame_t = frame_t.unsqueeze(0).unsqueeze(0).to(device)
                tokens = vae.encode(frame_t)
                tokens_flat = tokens.reshape(-1).cpu().numpy().astype(np.int64)

            ctx_tokens.append(tokens_flat)
            ctx_actions.append(1 if jumping else 0)

            if len(ctx_tokens) > K:
                ctx_tokens = ctx_tokens[-K:]
                ctx_actions = ctx_actions[-K:]

            # Need K frames before acting
            if len(ctx_tokens) < K:
                elapsed = time.perf_counter() - t0
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
                continue

            # Transformer + Controller
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                ctx_tok_np = np.array(ctx_tokens)
                ctx_act_np = np.array(ctx_actions)
                status = np.full((K, 1), wm.ALIVE_TOKEN, dtype=np.int64)
                ctx_with_status = np.concatenate([ctx_tok_np, status], axis=1)
                ctx_t = torch.from_numpy(ctx_with_status[None]).to(device)
                ctx_a = torch.from_numpy(ctx_act_np[None]).to(device)
                h_t = wm.encode_context(ctx_t, ctx_a)

            with torch.no_grad():
                prob, _ = controller(h_t.float())
                jump = prob[0].item() > args.jump_threshold

            if jump and not jumping:
                keyboard.press("space")
                jumping = True
            elif not jump and jumping:
                keyboard.release("space")
                jumping = False

            frames_survived += 1

            elapsed = time.perf_counter() - t0
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

        episode_time = time.perf_counter() - t_start
        results.append({
            "run": run_idx + 1,
            "frames_survived": frames_survived,
            "time_s": round(episode_time, 2),
        })

        run_idx += 1
        print(f"  Run {run_idx:3d}/{args.n_runs}: "
              f"{frames_survived} frames ({episode_time:.1f}s)")

        # Wait for respawn (death animation + respawn)
        time.sleep(1.0)

    # Summary
    if results:
        survivals = [r["frames_survived"] for r in results]
        print(f"\n{'='*50}")
        print(f"Results ({len(results)} runs):")
        print(f"  Mean survival: {np.mean(survivals):.1f} frames "
              f"({np.mean(survivals)/args.fps:.1f}s)")
        print(f"  Std:  {np.std(survivals):.1f}")
        print(f"  Min:  {np.min(survivals)} | Max: {np.max(survivals)}")
        print(f"  Median: {np.median(survivals):.0f}")
        p25, p75 = np.percentile(survivals, [25, 75])
        print(f"  P25: {p25:.0f} | P75: {p75:.0f}")

        summary = {
            "n_runs": len(results),
            "mean_frames": round(float(np.mean(survivals)), 1),
            "std_frames": round(float(np.std(survivals)), 1),
            "min_frames": int(np.min(survivals)),
            "max_frames": int(np.max(survivals)),
            "median_frames": round(float(np.median(survivals)), 0),
            "p25_frames": round(float(p25), 0),
            "p75_frames": round(float(p75), 0),
            "mean_time_s": round(float(np.mean(survivals)) / args.fps, 2),
            "checkpoints": {
                "vae": args.vae_checkpoint,
                "transformer": args.transformer_checkpoint,
                "controller": args.controller_checkpoint,
            },
            "runs": results,
        }

        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
