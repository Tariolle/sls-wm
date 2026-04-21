"""Smoke-test JointStep compile under fullgraph=True.

Runs one forward + backward on tiny synthetic data. If fullgraph=True
surfaces a graph break, this script fails with a specific error pointing
at the offending op, so you can fix before the next real run.

Usage: python scripts/test_compile_fullgraph.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from deepdash.fsq import FSQVAE
from deepdash.world_model import WorldModel
from scripts.train_world_model import (
    JointStep, build_fsq_neighbor_table, build_structured_smooth_targets,
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compile_mode = "reduce-overhead" if device.type == "cuda" else "default"
    print(f"device={device} compile_mode={compile_mode}")

    torch.manual_seed(0)
    levels = [5, 5, 5, 5]
    K = 4
    B = 2
    vocab = 625

    fsq = FSQVAE(levels=levels).to(device)
    wm = WorldModel(
        vocab_size=vocab, n_actions=2, embed_dim=64, n_heads=4, n_layers=2,
        context_frames=K, dropout=0.0, tokens_per_frame=64, adaln=True,
        fsq_dim=len(levels), use_cpc=False,
    ).to(device)

    nt, nc = build_fsq_neighbor_table(levels)
    nt, nc = nt.to(device), nc.to(device)
    stm = build_structured_smooth_targets(
        levels, wm.full_vocab_size, sigma=1.0, smoothing=0.1,
        kernel="laplace",
    ).to(device)

    js = JointStep(
        fsq=fsq, wm=wm,
        alpha_uniform=0.0, cpc_weight=1.0,
        label_smoothing=0.1, focal_gamma=2.0,
        token_noise=0.05, fsq_noise=0.05,
        shift_max=4, use_recon=True,
        neighbor_table=nt, neighbor_counts=nc,
        soft_target_matrix=stm,
        fsq_levels=levels, fsq_sigma=1.0,
    ).to(device)

    print("compiling JointStep with fullgraph=True ...")
    t0 = time.perf_counter()
    try:
        js_c = torch.compile(js, mode=compile_mode, fullgraph=True)
    except Exception as e:
        print(f"COMPILE FAILED: {type(e).__name__}: {e}")
        return 1

    raw = torch.randint(0, 256, (B, K + 1, 64, 64), dtype=torch.uint8, device=device)
    act = torch.randint(0, 2, (B, K), dtype=torch.long, device=device)
    isd = torch.zeros(B, dtype=torch.bool, device=device)

    print("first forward (traces the graph) ...")
    amp = torch.bfloat16 if device.type == "cuda" else torch.float32
    try:
        with torch.autocast(device.type, dtype=amp, enabled=device.type == "cuda"):
            loss, metrics = js_c(raw, act, isd)
        print(f"  loss={loss.item():.4f}")
    except Exception as e:
        print(f"FORWARD FAILED: {type(e).__name__}: {e}")
        return 1

    print("backward ...")
    try:
        loss.backward()
    except Exception as e:
        print(f"BACKWARD FAILED: {type(e).__name__}: {e}")
        return 1

    dt = time.perf_counter() - t0
    print(f"OK — compile + fwd + bwd completed in {dt:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
