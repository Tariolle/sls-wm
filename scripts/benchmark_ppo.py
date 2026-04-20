"""Benchmark PPO dream rollout speed with different optimizations.

Tests combinations of: eager/compile, fp32/tf32/bf16/fp16, CUDA graphs.
Reports time per iteration and projected time for 10K iterations.

Usage:
    python scripts/benchmark_ppo.py
    python scripts/benchmark_ppo.py --n-episodes 64 --n-iters 3
"""

import argparse
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepdash.world_model import WorldModel
from deepdash.controller import MLPPolicy
from deepdash.config import apply_config


def run_dream_rollout(wm, ctrl, n_episodes, max_steps, device, dtype_ctx=None):
    """Simulate one PPO iteration: dream rollouts + controller forward."""
    K = wm.context_frames
    TPF = wm.tokens_per_frame

    ctx = torch.randint(0, 1000, (n_episodes, K, TPF + 1), device=device)
    acts = torch.randint(0, 2, (n_episodes, K), device=device)

    with torch.no_grad():
        ctx_mgr = torch.autocast(device.type, dtype=dtype_ctx) if dtype_ctx else nullcontext()
        with ctx_mgr:
            for step in range(max_steps):
                pred, dp, h_t = wm.predict_next_frame(ctx, acts, return_hidden=True)
                prob, val = ctrl(h_t)

                new_action = (prob > 0.5).long()
                alive = torch.full((n_episodes, 1), wm.ALIVE_TOKEN,
                                   dtype=torch.long, device=device)
                new_frame = torch.cat([pred, alive], dim=1).unsqueeze(1)
                ctx = torch.cat([ctx[:, 1:], new_frame], dim=1)
                acts = torch.cat([acts[:, 1:], new_action.unsqueeze(1)], dim=1)


def bench_config(name, wm, ctrl, args, device, dtype=None):
    """Benchmark a single config: warmup, time, report."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    try:
        for _ in range(args.warmup):
            run_dream_rollout(wm, ctrl, args.n_episodes, args.max_steps,
                              device, dtype)
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(args.n_iters):
            run_dream_rollout(wm, ctrl, args.n_episodes, args.max_steps,
                              device, dtype)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / args.n_iters

        mem = torch.cuda.max_memory_allocated() / 1e9
        hours = elapsed * 10000 / 3600
        print(f"{name:<30} {elapsed:>8.1f}s {hours:>9.1f}h {mem:>8.2f}")
        return elapsed
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"{name:<30} {'OOM':>8}")
            torch.cuda.empty_cache()
        else:
            print(f"{name:<30} ERROR: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=45)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--n-iters", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"PyTorch: {torch.__version__}")
    print(f"Episodes: {args.n_episodes}, Steps: {args.max_steps}")
    print()

    ns = argparse.Namespace(config=None)
    apply_config(ns)

    def make_models():
        wm = WorldModel(
            vocab_size=ns.vocab_size, embed_dim=ns.embed_dim,
            n_heads=ns.n_heads, n_layers=ns.n_layers,
            context_frames=ns.context_frames, dropout=ns.dropout,
            tokens_per_frame=ns.tokens_per_frame,
            adaln=getattr(ns, 'adaln', False),
            fsq_dim=len(ns.levels) if getattr(ns, 'levels', None) else None,
        ).to(device).eval()
        ctrl = MLPPolicy(h_dim=ns.embed_dim).to(device).eval()
        return wm, ctrl

    # Check torch.compile availability
    compile_works = False
    try:
        test = torch.compile(torch.nn.Linear(10, 10).to(device))
        test(torch.randn(2, 10, device=device))
        compile_works = True
    except Exception:
        pass

    print(f"torch.compile: {'available' if compile_works else 'not available'}")
    print()
    print(f"{'Config':<30} {'Iter':>8} {'10K iters':>9} {'VRAM':>8}")
    print("-" * 58)

    results = {}

    # === 1. Baseline: eager fp32 ===
    wm, ctrl = make_models()
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('medium')
    results['eager fp32'] = bench_config("eager fp32", wm, ctrl, args, device)

    # === 2. cuDNN benchmark (autotuning for convs) ===
    torch.backends.cudnn.benchmark = True
    results['+ cudnn.benchmark'] = bench_config("+ cudnn.benchmark", wm, ctrl, args, device)

    # === 3. TF32 matmul (Turing+: ~2x matmul speed) ===
    torch.set_float32_matmul_precision('high')
    results['+ tf32 matmul'] = bench_config("+ tf32 matmul", wm, ctrl, args, device)

    # === 4. BF16 autocast ===
    results['+ bf16 autocast'] = bench_config("+ bf16 autocast", wm, ctrl, args, device, torch.bfloat16)

    # === 5. FP16 autocast ===
    results['+ fp16 autocast'] = bench_config("+ fp16 autocast", wm, ctrl, args, device, torch.float16)

    # === 6. torch.compile (if available) ===
    if compile_works:
        wm_c, ctrl_c = make_models()
        wm_c = torch.compile(wm_c)
        results['+ compile'] = bench_config("+ compile (fp32+tf32)", wm_c, ctrl_c, args, device)
        results['+ compile + bf16'] = bench_config("+ compile + bf16", wm_c, ctrl_c, args, device, torch.bfloat16)

    # === Summary ===
    print()
    baseline = results.get('eager fp32')
    if baseline:
        print("Speedup vs eager fp32:")
        for name, t in results.items():
            if t and name != 'eager fp32':
                print(f"  {name:<28} {baseline/t:.2f}x")

    best_name = min((k for k, v in results.items() if v), key=lambda k: results[k])
    print(f"\nFastest: {best_name} ({results[best_name]:.1f}s/iter)")


if __name__ == "__main__":
    main()
