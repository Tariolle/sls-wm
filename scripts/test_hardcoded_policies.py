"""Test hardcoded policies (always jump, always idle, random) in dream rollouts.

Quick sanity check: if always-jump doesn't beat always-idle, no controller can learn.

Usage:
    python scripts/test_hardcoded_policies.py
"""

import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepdash.world_model import WorldModel


def load_episodes(d, K):
    shift_re = re.compile(r"_s[+-]\d+_[+-]\d+$")
    eps = []
    for ep in sorted(Path(d).glob("*")):
        if shift_re.search(ep.name):
            continue
        tp, ap = ep / "tokens.npy", ep / "actions.npy"
        if not tp.exists() or not ap.exists():
            continue
        t = np.load(tp).astype(np.int64)
        a = np.load(ap).astype(np.int64)
        if len(t) >= K + 1:
            eps.append((t, a))
    return eps


def sample_ctx(eps, K, rng):
    lo, hi = 8, 20
    t, a = eps[rng.integers(len(eps))]
    T = len(t)
    e = max(0, T - hi - K)
    l = max(0, T - lo - K)
    s = e if l <= e else rng.integers(e, l + 1)
    return t[s:s + K], a[s:s + K]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = WorldModel(
        vocab_size=1000, embed_dim=256, n_heads=8, n_layers=8,
        context_frames=4, tokens_per_frame=64,
    ).to(device)
    state = torch.load("checkpoints/transformer_best.pt",
                       map_location=device, weights_only=True)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    if sys.platform != "win32":
        try:
            model._backbone_forward = torch.compile(model._backbone_forward)
            print("torch.compile enabled")
        except Exception:
            pass

    episodes = load_episodes("data/death_episodes", 4)
    print(f"Loaded {len(episodes)} episodes")
    rng = np.random.default_rng(42)
    n_tests = 100
    max_steps = 20

    results = {"always_jump": [], "always_idle": [], "random": []}

    with torch.no_grad():
        for t in range(n_tests):
            ct, ca = sample_ctx(episodes, 4, rng)

            for pn, av in [("always_jump", 1), ("always_idle", 0), ("random", -1)]:
                status = np.full((1, 4, 1), model.ALIVE_TOKEN, dtype=np.int64)
                cx = torch.from_numpy(
                    np.concatenate([ct[None], status], axis=2)).to(device)
                cxa = torch.from_numpy(ca[None].copy()).to(device)
                surv = 0

                for step in range(max_steps):
                    pt, dp, ht = model.predict_next_frame(
                        cx, cxa, temperature=0.0, return_hidden=True)
                    if dp[0].item() > 0.5:
                        break
                    surv += 1
                    if av == -1:
                        act = rng.integers(0, 2)
                    else:
                        act = av
                    a = torch.tensor([[act]], dtype=torch.long, device=device)
                    ns = torch.full((1, 1), model.ALIVE_TOKEN,
                                    dtype=torch.long, device=device)
                    nf = torch.cat([pt, ns], dim=1).unsqueeze(1)
                    cx = torch.cat([cx[:, 1:], nf], dim=1)
                    cxa = torch.cat([cxa[:, 1:], a], dim=1)

                results[pn].append(surv)

            if (t + 1) % 20 == 0:
                print(f"  {t + 1}/{n_tests}...")

    print(f"\nResults ({n_tests} near-death contexts):")
    for n in results:
        v = results[n]
        print(f"  {n:15s}: mean={np.mean(v):.1f}  median={np.median(v):.1f}  "
              f"min={min(v)}  max={max(v)}")


if __name__ == "__main__":
    main()
