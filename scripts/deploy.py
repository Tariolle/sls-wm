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
from deepdash.controller import CNNPolicy


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


@torch.no_grad()
def encode_context_cached(wm, frame_tokens, actions, past_kvs):
    """Encode context frames with sliding-window KV cache.

    On the first call (past_kvs is None), encodes all K frames and caches
    KV for the full context.  On subsequent calls, evicts the oldest frame
    slot from the cache and runs only the newest frame + action through the
    transformer, reusing cached KV for the K-1 older frames.

    RoPE note: cached KV retains the RoPE angles from its original encoding
    positions.  After the window slides, those angles are off by one frame
    slot (66 positions).  In practice the temporal RoPE band uses theta=10000,
    so the angular error from a 1-step shift is negligible.  Use --no-kv-cache
    to verify numerically if needed.

    Args:
        wm: WorldModel instance.
        frame_tokens: (1, K, block_size) long -- K context frames with status.
        actions: (1, K) long -- actions for context frames.
        past_kvs: list[tuple(k, v)] per layer, or None on first call.
            Each k, v has shape (B, n_heads, cached_seq_len, head_dim).

    Returns:
        h_t: (1, embed_dim) -- hidden state at last context position.
        new_kvs: updated KV cache for next call.
    """
    K = wm.context_frames
    BS = wm.block_size          # 65
    slot = BS + 1               # 66 tokens per frame slot (frame block + action)

    if past_kvs is None:
        # -- Cold start: encode full context, cache everything --
        parts = []
        for i in range(K):
            parts.append(wm.token_embed(frame_tokens[:, i]))    # (1, 65, D)
            act = wm.action_embed(actions[:, i])
            parts.append(act.unsqueeze(1))                       # (1,  1, D)
        x = torch.cat(parts, dim=1)                              # (1, K*66, D)

        ctx_len = K * slot
        ctx_mask = wm.attn_mask[:ctx_len, :ctx_len]
        rope_cos = wm.rope_cos[:ctx_len]
        rope_sin = wm.rope_sin[:ctx_len]

        new_kvs = []
        for block in wm.blocks:
            x, kv = block(x, ctx_mask, rope_cos, rope_sin, use_cache=True)
            new_kvs.append(kv)
        x = wm.ln_f(x)
        return x[:, -1], new_kvs

    # -- Incremental: evict oldest slot, run newest slot with cache --
    # Trim oldest frame slot (first `slot` positions) from every layer's KV
    trimmed_kvs = []
    for (pk, pv) in past_kvs:
        trimmed_kvs.append((pk[:, :, slot:, :], pv[:, :, slot:, :]))

    # Build embeddings for the NEW frame + action only (66 tokens)
    new_frame = wm.token_embed(frame_tokens[:, -1])              # (1, 65, D)
    new_action = wm.action_embed(actions[:, -1]).unsqueeze(1)    # (1,  1, D)
    x_new = torch.cat([new_frame, new_action], dim=1)            # (1, 66, D)

    # RoPE for the last frame slot: positions (K-1)*66 .. K*66-1
    start = (K - 1) * slot
    end = K * slot
    rope_cos_new = wm.rope_cos[start:end]
    rope_sin_new = wm.rope_sin[start:end]

    # Attention mask for new tokens attending to (cached_prefix + new_tokens).
    # cached_prefix length = (K-1)*66 after trimming.
    cached_len = (K - 1) * slot
    # Block-causal: frame tokens (0..64) can't see the action token (65),
    # but the action token can see everything.
    attn_len = cached_len + slot
    sdpa_mask = torch.ones(slot, attn_len, dtype=torch.bool,
                           device=x_new.device)
    # Frame tokens in the new block must not attend to the action token
    sdpa_mask[:BS, cached_len + BS:] = False

    new_kvs = []
    for i, block in enumerate(wm.blocks):
        x_new, kv = block(x_new, ~sdpa_mask, rope_cos_new, rope_sin_new,
                          past_kv=trimmed_kvs[i], use_cache=True)
        new_kvs.append(kv)
    x_new = wm.ln_f(x_new)
    # h_t is at the action token position (last of the new tokens)
    return x_new[:, -1], new_kvs


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
    # Model architecture
    parser.add_argument("--levels", type=int, nargs="+", default=[8, 5, 5, 5])
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--embed-dim", type=int, default=384)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--tokens-per-frame", type=int, default=64)
    parser.add_argument("--context-frames", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use-kv-cache", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Sliding-window KV cache for encode_context (disabled: RoPE drift bug)")
    args = parser.parse_args()

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
    ).to(device)
    state = torch.load(args.transformer_checkpoint, map_location=device,
                       weights_only=True)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    wm.load_state_dict(state)
    wm.eval()

    print("Loading Controller...")
    controller = CNNPolicy(vocab_size=args.vocab_size).to(device)
    state = torch.load(args.controller_checkpoint, map_location=device,
                       weights_only=True)
    controller.load_state_dict(state)
    controller.eval()

    # Optimize inference
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if sys.platform != "win32":
            try:
                vae.encode = torch.compile(vae.encode)
                wm.encode_context = torch.compile(wm.encode_context)
                print("torch.compile enabled")
            except Exception as e:
                print(f"torch.compile not available: {e}")
        else:
            print("Skipping torch.compile (Windows)")

    kv_status = "ON" if args.use_kv_cache else "OFF"
    print(f"All models loaded. KV cache: {kv_status}\n")

    # --- Screen capture setup ---
    region = (0, 0, 1920, 1080)
    crop_x, crop_y, crop_size = 660, 48, 1032
    cam = dxcam.create()
    frame_interval = 1.0 / args.fps

    # --- State ---
    active = False
    jumping = False
    ctx_tokens = []  # list of (64,) int64 arrays
    ctx_actions = []  # list of int actions
    frame_count = 0
    kv_cache = None   # sliding KV cache (list of (k,v) per layer)

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
                ctx_tokens = []
                ctx_actions = []
                frame_count = 0
                kv_cache = None
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

        # --- FSQ encode ---
        t1 = time.perf_counter()
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            frame_t = torch.from_numpy(edge_frame.astype(np.float32) / 255.0)
            frame_t = frame_t.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 64, 64)
            tokens = vae.encode(frame_t)  # (1, 8, 8)
            tokens_flat = tokens.reshape(64).cpu().numpy().astype(np.int64)
        t_fsq = time.perf_counter() - t1

        # --- Update context buffer ---
        ctx_tokens.append(tokens_flat)
        ctx_actions.append(1 if jumping else 0)
        frame_count += 1

        # Keep only last K frames
        if len(ctx_tokens) > K:
            ctx_tokens = ctx_tokens[-K:]
            ctx_actions = ctx_actions[-K:]

        # --- Warmup: need K frames before acting ---
        if len(ctx_tokens) < K:
            print(f"  Warmup: {len(ctx_tokens)}/{K} frames")
            elapsed = time.perf_counter() - t0
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
            continue

        # --- Transformer: get h_t ---
        t1 = time.perf_counter()
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            ctx_tok_np = np.array(ctx_tokens)  # (K, 64)
            ctx_act_np = np.array(ctx_actions)  # (K,)
            status = np.full((K, 1), wm.ALIVE_TOKEN, dtype=np.int64)
            ctx_with_status = np.concatenate([ctx_tok_np, status], axis=1)

            ctx_t = torch.from_numpy(ctx_with_status[None]).to(device)
            ctx_a = torch.from_numpy(ctx_act_np[None]).to(device)

            if args.use_kv_cache:
                h_t, kv_cache = encode_context_cached(
                    wm, ctx_t, ctx_a, kv_cache)
            else:
                h_t = wm.encode_context(ctx_t, ctx_a)  # (1, embed_dim)
        t_tfm = time.perf_counter() - t1

        # --- Controller: decide action ---
        t1 = time.perf_counter()
        with torch.no_grad():
            current_tokens = torch.from_numpy(tokens_flat).unsqueeze(0).to(device)
            prob, _ = controller(current_tokens, h_t.float())
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
