"""Deploy the World Models agent on real Geometry Dash.

Captures screen at 30 FPS, runs FSQ + Transformer + Controller,
and simulates keyboard input to play the game.

Controls:
    F5  -- toggle agent on/off
    F10 -- quit

HUD: colored dot in top-left corner
    Black = standby, Red = idle, Green = jump

Usage:
    python scripts/deploy.py
    python scripts/deploy.py --controller-checkpoint checkpoints/controller_ppo_best.pt
"""

import argparse
import ctypes
import ctypes.wintypes as wt
import os
import sys
import time

import cv2
import dxcam
import keyboard
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from deepdash.fsq import FSQVAE
from deepdash.world_model import WorldModel
from deepdash.controller import CNNPolicy, V3CNNPolicy


# Win32 helpers for topmost window
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


def _force_topmost(window_title):
    hwnd = _user32.FindWindowW(None, window_title)
    if hwnd:
        _user32.SetWindowPos(
            hwnd, _HWND_TOPMOST, 0, 0, 0, 0,
            0x0002 | 0x0001 | _SWP_NOACTIVATE)


_SOBEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                        dtype=torch.float32).reshape(1, 1, 3, 3)
_SOBEL_Y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                        dtype=torch.float32).reshape(1, 1, 3, 3)
_sobel_device = None


def _init_sobel_kernels(device):
    global _SOBEL_X, _SOBEL_Y, _sobel_device
    if _sobel_device != device:
        _SOBEL_X = _SOBEL_X.to(device)
        _SOBEL_Y = _SOBEL_Y.to(device)
        _sobel_device = device


def preprocess_frame(rgb, crop_x, crop_y, crop_size, target_size, device=None):
    """RGB screenshot -> 64x64 Sobel edge map (uint8).

    GPU Sobel (numerically identical to cv2.Sobel) + CPU INTER_AREA resize.
    """
    cropped = rgb[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size]
    gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)

    if device is not None and device.type == "cuda":
        _init_sobel_kernels(device)
        gray_t = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0).to(device)
        padded = torch.nn.functional.pad(gray_t, (1, 1, 1, 1), mode='reflect')
        sx = torch.nn.functional.conv2d(padded, _SOBEL_X)
        sy = torch.nn.functional.conv2d(padded, _SOBEL_Y)
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
    parser = argparse.ArgumentParser(
        description="Deploy World Models agent on Geometry Dash")
    parser.add_argument("--vae-checkpoint", default="checkpoints/fsq_best.pt")
    parser.add_argument("--transformer-checkpoint",
                        default="checkpoints/transformer_best.pt")
    parser.add_argument("--controller-checkpoint",
                        default="checkpoints/controller_ppo_best.pt")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--jump-threshold", type=float, default=0.5,
                        help="Jump probability threshold (higher = less jumping)")
    # Model architecture (defaults from configs/v3.yaml)
    parser.add_argument("--config", default=None)
    parser.add_argument("--levels", type=int, nargs="+", default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--tokens-per-frame", type=int, default=None)
    parser.add_argument("--context-frames", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--policy-class", type=str, default=None,
                        choices=["cnn", "v3_cnn"],
                        help="Controller architecture. 'cnn' = E6.10-era "
                             "CNNPolicy. 'v3_cnn' = V3-deploy/V7 faithful "
                             "(direct h_t concat, ReLU+MaxPool, MTP head).")
    args = parser.parse_args()

    from deepdash.config import apply_config
    apply_config(args)
    apply_config(args, section="controller_ppo")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    K = args.context_frames

    # --- Load models ---
    print("Loading FSQ-VAE...")
    vae = FSQVAE(levels=args.levels).to(device)
    state = torch.load(args.vae_checkpoint, map_location=device,
                       weights_only=True)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    vae.load_state_dict(state)
    vae.eval()

    print("Loading Transformer...")
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

    print("Loading Controller...")
    grid_size = int(args.tokens_per_frame ** 0.5)
    policy_class = (getattr(args, "policy_class", None) or "cnn").lower()
    if policy_class == "v3_cnn":
        controller = V3CNNPolicy(
            vocab_size=args.vocab_size,
            grid_size=grid_size,
            token_embed_dim=getattr(args, 'token_embed_dim', 16),
            h_dim=args.embed_dim,
            mtp_steps=int(getattr(args, "mtp_steps", None) or 8),
        ).to(device)
        print(f"  V3CNNPolicy (mtp_steps={controller.mtp_steps})")
    else:
        controller = CNNPolicy(
            vocab_size=args.vocab_size,
            grid_size=grid_size,
            token_embed_dim=getattr(args, 'token_embed_dim', 16),
            h_dim=args.embed_dim,
            temporal_dim=getattr(args, 'temporal_dim', 32),
        ).to(device)
        print(f"  CNNPolicy (temporal_dim={getattr(args, 'temporal_dim', 32)})")
    state = torch.load(args.controller_checkpoint, map_location=device,
                       weights_only=True)
    # controller_ppo_best.pt is a raw state_dict; controller_ppo_latest.pt is
    # a full training checkpoint — unwrap it to the same shape.
    if "controller" in state and isinstance(state["controller"], dict):
        state = state["controller"]
    controller.load_state_dict(state)
    controller.eval()

    # Optimize inference
    use_cuda_graph = False
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            vae.encode = torch.compile(vae.encode)
            print("torch.compile enabled for FSQ")
        except Exception as e:
            print(f"torch.compile not available: {e}")

        # CUDA Graph for encode_context: records all kernels once,
        # replays with a single CPU call (zero per-op launch overhead).
        try:
            graph_ctx_t = torch.zeros(1, K, 65, dtype=torch.long, device=device)
            graph_ctx_a = torch.zeros(1, K, dtype=torch.long, device=device)

            # Warmup runs (CUDA needs to see the kernels before capture)
            with torch.no_grad():
                for _ in range(3):
                    wm.encode_context(graph_ctx_t, graph_ctx_a)
            torch.cuda.synchronize()

            encode_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(encode_graph):
                with torch.no_grad():
                    graph_h_t = wm.encode_context(graph_ctx_t, graph_ctx_a)

            use_cuda_graph = True
            print("CUDA Graph captured for encode_context")
        except Exception as e:
            print(f"CUDA Graph capture failed, using eager: {e}")

    print("All models loaded.\n")

    # --- Screen capture setup ---
    region = (0, 0, 1920, 1080)
    crop_x, crop_y, crop_size = 660, 48, 1032
    cam = dxcam.create()
    frame_interval = 1.0 / args.fps

    # --- State ---
    active = False
    jumping = False
    # Ring buffer on GPU: avoids GPU->CPU->GPU round-trip each frame
    ctx_tokens = torch.zeros(K, 64, dtype=torch.long, device=device)
    ctx_actions = torch.zeros(K, dtype=torch.long, device=device)
    ctx_fill = 0  # how many frames stored so far
    ctx_status = torch.full((K, 1), wm.ALIVE_TOKEN, dtype=torch.long, device=device)
    frame_count = 0


    # Pinned memory buffer for CPU->GPU frame transfer (skips implicit staging copy)
    if device.type == "cuda":
        pin_buf = torch.zeros(1, 1, 64, 64, dtype=torch.float32, pin_memory=True)
    else:
        pin_buf = None

    print("Controls:")
    print("  F5  -- toggle agent on/off")
    print("  F10 -- quit")
    print("\nWaiting for F5...")

    while True:
        t0 = time.perf_counter()

        # --- Hotkeys ---
        if keyboard.is_pressed("f5"):
            active = not active
            if active:
                ctx_tokens.zero_()
                ctx_actions.zero_()
                ctx_fill = 0
                frame_count = 0

                if jumping:
                    keyboard.release("space")
                    jumping = False
                print(">> Agent ON")
            else:
                if jumping:
                    keyboard.release("space")
                    jumping = False
                print(">> Agent OFF")
            while keyboard.is_pressed("f5"):
                time.sleep(0.01)

        if keyboard.is_pressed("f10"):
            break

        # --- Capture ---
        t1 = time.perf_counter()
        img = cam.grab(region=region)
        if img is None:
            time.sleep(0.001)
            continue
        t_capture = time.perf_counter() - t1

        # --- Sobel preprocess ---
        t1 = time.perf_counter()
        edge_frame = preprocess_frame(img, crop_x, crop_y, crop_size, 64, device)
        t_sobel = time.perf_counter() - t1

        if not active:
            elapsed = time.perf_counter() - t0
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
            continue

        # --- FSQ encode (stays on GPU) ---
        t1 = time.perf_counter()
        with torch.no_grad(), torch.no_grad():
            if pin_buf is not None:
                pin_buf[0, 0] = torch.from_numpy(edge_frame.astype(np.float32) * (1.0 / 255.0))
                frame_t = pin_buf.to(device, non_blocking=True)
            else:
                frame_t = torch.from_numpy(edge_frame.astype(np.float32) * (1.0 / 255.0))
                frame_t = frame_t.unsqueeze(0).unsqueeze(0).to(device)
            tokens = vae.encode(frame_t).reshape(64)  # (64,) on GPU
        t_fsq = time.perf_counter() - t1

        # --- Update context ring buffer (all on GPU) ---
        if ctx_fill < K:
            ctx_tokens[ctx_fill] = tokens
            ctx_actions[ctx_fill] = 1 if jumping else 0
            ctx_fill += 1
        else:
            ctx_tokens[:-1] = ctx_tokens[1:].clone()
            ctx_actions[:-1] = ctx_actions[1:].clone()
            ctx_tokens[-1] = tokens
            ctx_actions[-1] = 1 if jumping else 0
        frame_count += 1

        # --- Warmup: need K frames before acting ---
        if ctx_fill < K:
            print(f"  Warmup: {ctx_fill}/{K} frames")
            elapsed = time.perf_counter() - t0
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
            continue

        # --- Transformer: get h_t (all on GPU, no CPU round-trip) ---
        t1 = time.perf_counter()
        if use_cuda_graph:
            ctx_t = torch.cat([ctx_tokens, ctx_status], dim=1).unsqueeze(0)
            graph_ctx_t.copy_(ctx_t)
            graph_ctx_a.copy_(ctx_actions.unsqueeze(0))
            encode_graph.replay()
            h_t = graph_h_t
        else:
            with torch.no_grad(), torch.no_grad():
                ctx_t = torch.cat([ctx_tokens, ctx_status], dim=1).unsqueeze(0)
                ctx_a = ctx_actions.unsqueeze(0)
                h_t = wm.encode_context(ctx_t, ctx_a)
        t_tfm = time.perf_counter() - t1

        # --- Controller: decide action ---
        t1 = time.perf_counter()
        with torch.no_grad():
            # ctx_tokens is (K, 64); controller wants (B=1, 64) = current frame
            z_t = ctx_tokens[-1:].clone()
            prob, _ = controller(z_t, h_t.float())
            jump = prob[0].item() > args.jump_threshold
        t_ctrl = time.perf_counter() - t1

        # --- Execute action ---
        p = prob[0].item()
        if jump and not jumping:
            keyboard.press("space")
            jumping = True
        elif not jump and jumping:
            keyboard.release("space")
            jumping = False

        # Track probability distribution
        if not hasattr(main, '_probs'):
            main._probs = []
        main._probs.append(p)

        # --- Frame rate + per-stage timing ---
        elapsed = time.perf_counter() - t0
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)
        total_frame_time = time.perf_counter() - t0
        if frame_count % 30 == 0:
            real_fps = 1.0 / total_frame_time if total_frame_time > 0 else 0
            probs = main._probs
            lo = sum(1 for x in probs if x < 0.3)
            mid = sum(1 for x in probs if 0.3 <= x <= 0.7)
            hi = sum(1 for x in probs if x > 0.7)
            n = len(probs)
            print(f"  frame {frame_count}: {real_fps:.0f}fps p={p:.2f} | "
                  f"prob dist: <0.3={lo*100//n}% 0.3-0.7={mid*100//n}% >0.7={hi*100//n}%")
            main._probs = []

    # Cleanup
    if jumping:
        keyboard.release("space")
    del cam
    print("\nDone.")


if __name__ == "__main__":
    main()
