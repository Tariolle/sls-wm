"""Benchmark real-time inference latency of the full pipeline.

Measures predict_next_frame (prefill + decode) which is the bottleneck
for real-time play. In deployment, only prefill is needed (h_t for the
controller), but we benchmark the full call for a conservative estimate.

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


def main():
    parser = argparse.ArgumentParser(description="Benchmark inference latency")
    parser.add_argument("--checkpoint", default="checkpoints/transformer_best.pt")
    parser.add_argument("--episodes-dir", default="data/episodes")
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
        vocab_size=1000, embed_dim=128, n_heads=4, n_layers=6,
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

    # Warmup
    with torch.no_grad():
        for _ in range(args.warmup):
            model.predict_next_frame(ctx_s, actions, return_hidden=True)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(args.n_runs):
            model.predict_next_frame(ctx_s, actions, return_hidden=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) / args.n_runs * 1000

    print(f"\npredict_next_frame (prefill + decode):")
    print(f"  {elapsed_ms:.2f} ms  (mean over {args.n_runs} runs)")
    print(f"  Budget 30 FPS: 33.3 ms")
    if elapsed_ms < 33.3:
        print(f"  -> OK ({33.3 / elapsed_ms:.1f}x margin)")
    else:
        print(f"  -> TOO SLOW ({elapsed_ms / 33.3:.1f}x over budget)")


if __name__ == "__main__":
    main()
