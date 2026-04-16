"""Benchmark training throughput under various optimization combos.

Tests: eager vs compile, fp32 vs fp16 vs bf16, compile modes.
Reports time/step and throughput for each configuration.

Each config runs in a subprocess for clean CUDA memory state,
preventing fragmentation from causing OOM in later configs.

Usage:
    python scripts/benchmark_training.py
    python scripts/benchmark_training.py --config configs/v5.yaml --warmup 3 --steps 10
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch


# Config tuples: (name, amp_dtype, use_scaler, use_compile, compile_mode, tf32, fused_optim)
CONFIGS = [
    ("eager + fp32",                       "none",     False, False, "none",             False, False),
    ("eager + fp32 + tf32",                "none",     False, False, "none",             True,  False),
    ("eager + fp16 + GradScaler",          "float16",  True,  False, "none",             False, False),
    ("eager + fp16 + GradScaler + tf32",   "float16",  True,  False, "none",             True,  False),
    ("eager + bf16",                       "bfloat16", False, False, "none",             False, False),
    ("eager + bf16 + tf32",                "bfloat16", False, False, "none",             True,  False),
    ("eager + bf16 + tf32 + fused optim",  "bfloat16", False, False, "none",             True,  True),
    ("compile + fp16 + tf32",              "float16",  True,  True,  "default",          True,  True),
    ("compile + bf16 + tf32",              "bfloat16", False, True,  "default",          True,  True),
    ("compile(reduce-overhead) + bf16 + tf32", "bfloat16", False, True, "reduce-overhead", True, True),
    ("compile(max-autotune) + bf16 + tf32",    "bfloat16", False, True, "max-autotune",    True, True),
]


def run_single_config(config_json, model_json, batch_size, warmup, steps):
    """Run a single benchmark config (called in subprocess)."""
    import torch.nn.functional as F

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from deepdash.world_model import WorldModel

    cfg = json.loads(config_json)
    mp = json.loads(model_json)
    name = cfg["name"]
    amp_dtype_str = cfg["amp_dtype"]
    use_scaler = cfg["use_scaler"]
    use_compile = cfg["use_compile"]
    compile_mode = cfg["compile_mode"]
    tf32 = cfg["tf32"]
    fused_optim = cfg["fused_optim"]

    amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "none": None}[amp_dtype_str]

    device = torch.device("cuda")

    torch.set_float32_matmul_precision("high" if tf32 else "highest")
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32

    # Synthetic data
    tpf = int(mp["tokens_per_frame"])
    block_size = tpf + 1
    B, K, BS = batch_size, int(mp["context_frames"]), block_size
    tokens = torch.randint(0, int(mp["vocab_size"]), (B, K + 1, tpf), device=device)
    status = torch.zeros(B, K + 1, 1, dtype=torch.long, device=device)
    frames = torch.cat([tokens, status], dim=2)
    actions = torch.randint(0, 2, (B, K), device=device)
    target = torch.randint(0, int(mp["vocab_size"]), (B, tpf), device=device)

    model = WorldModel(
        vocab_size=int(mp["vocab_size"]), embed_dim=int(mp["embed_dim"]),
        n_heads=int(mp["n_heads"]), n_layers=int(mp["n_layers"]),
        context_frames=int(mp["context_frames"]), dropout=float(mp["dropout"]),
        tokens_per_frame=int(mp["tokens_per_frame"]),
        adaln=bool(mp.get("adaln", False)),
    ).to(device)

    if use_compile:
        model = torch.compile(model, mode=compile_mode)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            _ = model(frames, actions)
        torch.cuda.synchronize()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=fused_optim)
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None

    times = []
    for i in range(warmup + steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            logits, cpc_loss = model(frames, actions)
            pred = logits[:, :target.shape[1]]
            loss = F.cross_entropy(pred.reshape(-1, pred.size(-1)), target.reshape(-1))
            total = loss + cpc_loss

        optimizer.zero_grad()
        if scaler:
            scaler.scale(total).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total.backward()
            optimizer.step()

        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000

        if i >= warmup:
            times.append(dt)

    med = sorted(times)[len(times) // 2]
    mean = sum(times) / len(times)
    # Output JSON result for parent process to parse
    print(json.dumps({"name": name, "median": med, "mean": mean,
                       "throughput": batch_size / (med / 1000)}))


def main():
    parser = argparse.ArgumentParser(description="Benchmark training optimizations")
    parser.add_argument("--config", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--_run-single", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_config-json", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_model-json", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    # Subprocess mode: run a single config and exit
    if args._run_single:
        run_single_config(args._config_json, args._model_json,
                          args.batch_size, args.warmup, args.steps)
        return

    # Parent mode: parse model config, then spawn subprocesses
    from deepdash.config import apply_config
    parser2 = argparse.ArgumentParser()
    for k in ["vocab_size", "embed_dim", "n_heads", "n_layers",
              "tokens_per_frame", "context_frames", "dropout"]:
        parser2.add_argument(f"--{k.replace('_', '-')}", default=None)
    parser2.add_argument("--config", default=args.config)
    parser2.add_argument("--adaln", default=None)
    margs = parser2.parse_args([])
    if args.config:
        margs.config = args.config
    apply_config(margs, section="transformer")

    model_params = {
        "vocab_size": margs.vocab_size, "embed_dim": margs.embed_dim,
        "n_heads": margs.n_heads, "n_layers": margs.n_layers,
        "context_frames": margs.context_frames, "dropout": margs.dropout,
        "tokens_per_frame": margs.tokens_per_frame,
        "adaln": getattr(margs, "adaln", False),
    }
    model_json = json.dumps(model_params)

    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Model: {margs.embed_dim}d / {margs.n_layers}L / {margs.n_heads}H")
    print(f"Batch size: {args.batch_size}, Warmup: {args.warmup}, Steps: {args.steps}")
    print(f"Flash Attention (SDPA): {torch.backends.cuda.flash_sdp_enabled()}")
    print()

    results = []
    for name, amp_dtype, use_scaler, use_compile, compile_mode, tf32, fused_optim in CONFIGS:
        cfg = json.dumps({
            "name": name, "amp_dtype": amp_dtype, "use_scaler": use_scaler,
            "use_compile": use_compile, "compile_mode": compile_mode,
            "tf32": tf32, "fused_optim": fused_optim,
        })

        cmd = [
            sys.executable, __file__,
            "--_run-single",
            "--_config-json", cfg,
            "--_model-json", model_json,
            "--batch-size", str(args.batch_size),
            "--warmup", str(args.warmup),
            "--steps", str(args.steps),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                # Extract first meaningful line from stderr
                err = result.stderr.strip().split("\n")[-1][:120]
                print(f"  {name:45s}  FAILED: {err}")
                continue

            # Parse JSON from last line of stdout
            output_lines = result.stdout.strip().split("\n")
            data = json.loads(output_lines[-1])
            med = data["median"]
            mean = data["mean"]
            throughput = data["throughput"]
            print(f"  {name:45s}  median={med:7.1f}ms  mean={mean:7.1f}ms  "
                  f"throughput={throughput:6.0f} samples/s")
            results.append((name, med))

        except subprocess.TimeoutExpired:
            print(f"  {name:45s}  TIMEOUT (>300s)")
        except Exception as e:
            print(f"  {name:45s}  FAILED: {e}")

    print("\n=== Summary (sorted by speed) ===")
    baseline = next((m for n, m in results if "eager + fp32" in n and "tf32" not in n), results[0][1])
    for name, med in sorted(results, key=lambda x: x[1]):
        speedup = baseline / med
        print(f"  {name:45s}  {med:7.1f}ms  {speedup:.2f}x vs eager fp32")


if __name__ == "__main__":
    main()
