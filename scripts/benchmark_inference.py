"""Benchmark inference latency for real-time play.

Two modes:
  - Prefill only: context → h_t (what real-time play actually needs)
  - Full predict_next_frame: prefill + MaskGIT iterative decode (dream rollouts)

Tests each mode: eager, AMP, torch.compile + AMP.
Compilation cost is paid upfront and excluded from timing.

Usage:
    python scripts/benchmark_inference.py
    python scripts/benchmark_inference.py --device cpu
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepdash.world_model import WorldModel


def prefill_only(model, ctx_s, actions):
    """Run only the prefill pass to extract h_t (no autoregressive decode)."""
    K = model.context_frames
    BS = model.block_size

    parts = []
    for i in range(K):
        parts.append(model.token_embed(ctx_s[:, i]))
        act = model.action_embed(actions[:, i])
        parts.append(act.unsqueeze(1))
    x = torch.cat(parts, dim=1)

    ctx_len = K * (BS + 1)
    ctx_mask = model.attn_mask[:ctx_len, :ctx_len]
    rope_cos = model.rope_cos[:ctx_len]
    rope_sin = model.rope_sin[:ctx_len]

    for block in model.blocks:
        x, _ = block(x, ctx_mask, rope_cos, rope_sin)
    x = model.ln_f(x)
    return x[:, -1]


def _with_amp(fn):
    with torch.autocast("cuda", dtype=torch.float16):
        return fn()


def main():
    parser = argparse.ArgumentParser(description="Benchmark inference latency")
    parser.add_argument("--checkpoint", default="checkpoints/transformer_best.pt")
    parser.add_argument("--episodes-dir", default="data/death_episodes")
    parser.add_argument("--device", default=None, help="Force device (cpu/cuda)")
    parser.add_argument("--n-runs", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    model = WorldModel(
        vocab_size=1000, embed_dim=256, n_heads=8, n_layers=8,
        context_frames=4, tokens_per_frame=64,
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    # Load one episode for context
    ep = next(ep for ep in sorted(Path(args.episodes_dir).glob("*"))
              if (ep / "tokens.npy").exists())
    t = np.load(ep / "tokens.npy").astype(np.int64)
    ctx = torch.from_numpy(t[:4]).unsqueeze(0).to(device)
    status = torch.full((1, 4, 1), model.ALIVE_TOKEN,
                        dtype=torch.long, device=device)
    ctx_s = torch.cat([ctx, status], dim=2)
    actions = torch.zeros(1, 4, dtype=torch.long, device=device)

    use_cuda = device.type == "cuda"

    def sync():
        if use_cuda:
            torch.cuda.synchronize()

    def bench(fn, label):
        with torch.no_grad():
            # Warmup (excluded from timing)
            for _ in range(args.warmup):
                fn()
            sync()
            # Timed runs
            start = time.perf_counter()
            for _ in range(args.n_runs):
                fn()
            sync()
            ms = (time.perf_counter() - start) / args.n_runs * 1000
        verdict = f"OK ({33.3 / ms:.1f}x margin)" if ms < 33.3 else \
                  f"TOO SLOW ({ms / 33.3:.1f}x over budget)"
        print(f"  {label}: {ms:.2f} ms  -> {verdict}")
        return ms

    print(f"\nBudget: 33.3 ms (30 FPS)")

    # --- Eager ---
    print(f"\n=== Eager ===")
    bench(lambda: prefill_only(model, ctx_s, actions),
          "Prefill only")
    bench(lambda: model.predict_next_frame(ctx_s, actions, return_hidden=True),
          "Full predict_next_frame")

    # --- AMP ---
    if use_cuda:
        print(f"\n=== AMP (FP16) ===")
        bench(lambda: _with_amp(lambda: prefill_only(model, ctx_s, actions)),
              "Prefill only")
        bench(lambda: _with_amp(lambda: model.predict_next_frame(
            ctx_s, actions, return_hidden=True)),
              "Full predict_next_frame")

    # --- torch.compile + AMP ---
    if sys.platform != "win32":
        print(f"\n=== torch.compile + AMP ===")
        print("  Compiling (one-time cost, excluded from benchmark)...")
        try:
            compiled_prefill = torch.compile(prefill_only)
            compiled_model = torch.compile(model)
            # Force compilation by running once
            with torch.no_grad():
                if use_cuda:
                    _with_amp(lambda: compiled_prefill(model, ctx_s, actions))
                    _with_amp(lambda: compiled_model.predict_next_frame(
                        ctx_s, actions, return_hidden=True))
                else:
                    compiled_prefill(model, ctx_s, actions)
                    compiled_model.predict_next_frame(
                        ctx_s, actions, return_hidden=True)
                sync()
            print("  Compilation done.\n")

            if use_cuda:
                bench(lambda: _with_amp(
                    lambda: compiled_prefill(model, ctx_s, actions)),
                      "Prefill only (compiled)")
                bench(lambda: _with_amp(
                    lambda: compiled_model.predict_next_frame(
                        ctx_s, actions, return_hidden=True)),
                      "Full predict_next_frame (compiled)")
            else:
                bench(lambda: compiled_prefill(model, ctx_s, actions),
                      "Prefill only (compiled)")
                bench(lambda: compiled_model.predict_next_frame(
                    ctx_s, actions, return_hidden=True),
                      "Full predict_next_frame (compiled)")
        except Exception as e:
            print(f"  torch.compile failed: {e}")


if __name__ == "__main__":
    main()
