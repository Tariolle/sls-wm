"""Train a linear controller via CMA-ES in dream rollouts.

The controller maps the Transformer's hidden state h_t to a binary action
(jump/idle). Fitness = mean frames survived before the world model predicts
death. All rollouts are batched on GPU for maximum throughput.

Usage:
    python scripts/train_controller.py
    python scripts/train_controller.py --popsize 256 --n-episodes 16 --max-generations 200
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import cma
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepdash.world_model import WorldModel
from deepdash.controller import Controller


def _unwrap(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def load_episodes(episodes_dir, context_frames):
    """Load all tokenized episodes with enough frames for context + 1 rollout step."""
    episodes = []
    for ep in sorted(Path(episodes_dir).glob("*")):
        tp = ep / "tokens.npy"
        ap = ep / "actions.npy"
        if not tp.exists() or not ap.exists():
            continue
        tokens = np.load(tp).astype(np.int64)   # (T, TPF)
        actions = np.load(ap).astype(np.int64)   # (T,)
        if len(tokens) >= context_frames + 1:
            episodes.append((tokens, actions))
    return episodes


def sample_contexts(episodes, n, context_frames, rng, near_death_range=(6, 12)):
    """Sample n contexts biased toward near-death windows.

    Every episode ends with death at frame T-1. Contexts are sampled so the
    death event falls within the dream rollout horizon, giving the controller
    a meaningful survival signal.

    Args:
        near_death_range: (min, max) frames before death to start the context.
            E.g. (6, 12) means context starts at T-12 to T-6.

    Returns:
        ctx_tokens: (n, K, TPF) int64
        ctx_actions: (n, K) int64
    """
    K = context_frames
    lo, hi = near_death_range
    all_ctx_tokens = []
    all_ctx_actions = []
    for _ in range(n):
        ep_idx = rng.integers(len(episodes))
        tokens, actions = episodes[ep_idx]
        T = len(tokens)
        # Start context so death is reachable: T - hi to T - lo frames before end
        earliest = max(0, T - hi - K)
        latest = max(0, T - lo - K)
        if latest <= earliest:
            start = earliest
        else:
            start = rng.integers(earliest, latest + 1)
        all_ctx_tokens.append(tokens[start:start + K])
        all_ctx_actions.append(actions[start:start + K])
    return np.array(all_ctx_tokens), np.array(all_ctx_actions)


def dream_rollout_batched(model, ctx_tokens_np, ctx_actions_np, W, b,
                          max_steps, death_threshold, device):
    """Run batched dream rollouts for one candidate controller.

    Args:
        model: WorldModel (eval mode, possibly compiled).
        ctx_tokens_np: (n_episodes, K, TPF) int64 numpy.
        ctx_actions_np: (n_episodes, K) int64 numpy.
        W: (hidden_dim,) torch tensor — controller weights.
        b: float — controller bias.
        max_steps: max dream steps.
        death_threshold: probability threshold for death.
        device: torch device.

    Returns:
        mean_survival: float — mean frames survived across episodes.
    """
    m = _unwrap(model)
    TPF = m.tokens_per_frame
    B = ctx_tokens_np.shape[0]

    # Add ALIVE status column: (B, K, TPF+1)
    status = np.full((*ctx_tokens_np.shape[:2], 1), m.ALIVE_TOKEN, dtype=np.int64)
    ctx_with_status = np.concatenate([ctx_tokens_np, status], axis=2)

    ctx_t = torch.from_numpy(ctx_with_status).to(device)
    ctx_a = torch.from_numpy(ctx_actions_np).to(device)

    alive = torch.ones(B, dtype=torch.bool, device=device)
    survival = torch.zeros(B, dtype=torch.float32, device=device)

    W_dev = W.to(device)

    for step in range(max_steps):
        if not alive.any():
            break

        pred_tokens, death_prob, h_t = model.predict_next_frame(
            ctx_t, ctx_a, temperature=0.0, return_hidden=True)

        # Death check
        died = death_prob > death_threshold
        alive &= ~died
        survival += alive.float()

        # Controller action
        logits = h_t @ W_dev + b
        actions_new = (logits.sigmoid() > 0.5).long()

        # Shift context: drop oldest frame, append predicted with ALIVE status
        new_status = torch.full((B, 1), m.ALIVE_TOKEN, dtype=torch.long, device=device)
        new_frame = torch.cat([pred_tokens, new_status], dim=1).unsqueeze(1)  # (B, 1, TPF+1)
        ctx_t = torch.cat([ctx_t[:, 1:], new_frame], dim=1)
        ctx_a = torch.cat([ctx_a[:, 1:], actions_new.unsqueeze(1)], dim=1)

    return survival.mean().item()


def evaluate_population(model, episodes, candidates, n_episodes, context_frames,
                        max_steps, death_threshold, device, rng, hidden_dim):
    """Evaluate all CMA-ES candidates in batched dream rollouts.

    Batches ALL candidates × episodes into a single mega-batch per dream step.

    Returns:
        fitnesses: list of float — one per candidate (negated survival for CMA-ES).
    """
    m = _unwrap(model)
    TPF = m.tokens_per_frame
    popsize = len(candidates)
    B = popsize * n_episodes

    # Sample contexts: same set for all candidates in this generation
    ctx_tokens_np, ctx_actions_np = sample_contexts(
        episodes, n_episodes, context_frames, rng)

    # Tile contexts for all candidates: (popsize * n_episodes, K, ...)
    ctx_tokens_tiled = np.tile(ctx_tokens_np, (popsize, 1, 1))  # (B, K, TPF)
    ctx_actions_tiled = np.tile(ctx_actions_np, (popsize, 1))    # (B, K)

    # Add ALIVE status
    status = np.full((*ctx_tokens_tiled.shape[:2], 1), m.ALIVE_TOKEN, dtype=np.int64)
    ctx_with_status = np.concatenate([ctx_tokens_tiled, status], axis=2)

    ctx_t = torch.from_numpy(ctx_with_status).to(device)
    ctx_a = torch.from_numpy(ctx_actions_tiled).to(device)

    alive = torch.ones(B, dtype=torch.bool, device=device)
    survival = torch.zeros(B, dtype=torch.float32, device=device)

    # Precompute all controller weights on device: (popsize, hidden_dim)
    W_all = torch.zeros(B, hidden_dim, device=device)
    b_all = torch.zeros(B, device=device)
    for i, cand in enumerate(candidates):
        start = i * n_episodes
        end = start + n_episodes
        W_all[start:end] = torch.from_numpy(cand[:hidden_dim].copy()).float().to(device)
        b_all[start:end] = float(cand[hidden_dim])

    use_amp = device.type == "cuda"

    for step in range(max_steps):
        if not alive.any():
            break

        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            pred_tokens, death_prob, h_t = model.predict_next_frame(
                ctx_t, ctx_a, temperature=0.0, return_hidden=True)

        # Death check
        died = death_prob > death_threshold
        alive &= ~died
        survival += alive.float()

        # Controller action (batched for all candidates)
        logits = (h_t * W_all).sum(dim=1) + b_all
        actions_new = (logits.sigmoid() > 0.5).long()

        # Shift context
        new_status = torch.full((B, 1), m.ALIVE_TOKEN, dtype=torch.long, device=device)
        new_frame = torch.cat([pred_tokens, new_status], dim=1).unsqueeze(1)
        ctx_t = torch.cat([ctx_t[:, 1:], new_frame], dim=1)
        ctx_a = torch.cat([ctx_a[:, 1:], actions_new.unsqueeze(1)], dim=1)

    # Reshape survival: (popsize, n_episodes) → mean per candidate
    survival_per_candidate = survival.view(popsize, n_episodes).mean(dim=1)
    # CMA-ES minimizes, so negate
    return (-survival_per_candidate).cpu().tolist()


def main():
    parser = argparse.ArgumentParser(description="Train controller via CMA-ES in dream rollouts")
    parser.add_argument("--transformer-checkpoint", default="checkpoints/transformer_best.pt")
    parser.add_argument("--episodes-dir", default="data/episodes")
    # CMA-ES
    parser.add_argument("--max-generations", type=int, default=200)
    parser.add_argument("--popsize", type=int, default=256)
    parser.add_argument("--sigma0", type=float, default=0.5)
    parser.add_argument("--percentile-norm", action="store_true",
                        help="Normalize fitness via running 5th/95th percentile (DreamerV3-style)")
    # Rollout
    parser.add_argument("--n-episodes", type=int, default=16)
    parser.add_argument("--max-dream-steps", type=int, default=20)
    parser.add_argument("--death-threshold", type=float, default=0.5)
    parser.add_argument("--context-frames", type=int, default=4)
    # Model architecture (must match checkpoint)
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--tokens-per-frame", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    # Output
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile on the world model")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
    rng = np.random.default_rng(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load world model
    model = WorldModel(
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        context_frames=args.context_frames,
        dropout=args.dropout,
        tokens_per_frame=args.tokens_per_frame,
    ).to(device)
    state = torch.load(args.transformer_checkpoint, map_location=device, weights_only=True)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    if args.compile and sys.platform != "win32":
        try:
            model = torch.compile(model)
            print("Model compiled with torch.compile")
        except Exception as e:
            print(f"torch.compile failed: {e}")

    # Load episodes
    episodes = load_episodes(args.episodes_dir, args.context_frames)
    print(f"Loaded {len(episodes)} tokenized episodes")
    if not episodes:
        print("No tokenized episodes found. Run tokenize_episodes.py first.")
        return

    # Setup CMA-ES
    hidden_dim = args.embed_dim
    n_params = hidden_dim + 1  # W + b
    x0 = np.zeros(n_params)
    es = cma.CMAEvolutionStrategy(x0, args.sigma0, {
        "popsize": args.popsize,
        "seed": args.seed,
        "maxiter": args.max_generations,
        "verbose": -1,  # we do our own logging
    })

    # CSV log
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "controller_log.csv"
    log_file = open(log_path, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["generation", "best_fitness", "mean_fitness", "median_fitness",
                     "sigma", "time_s"])

    best_fitness_ever = -float("inf")
    best_params = None

    # Percentile normalization state (DreamerV3-style EMA of 5th/95th percentile)
    perc_ema_low = None
    perc_ema_high = None
    perc_decay = 0.99

    print(f"CMA-ES: popsize={args.popsize}, sigma0={args.sigma0}, "
          f"n_params={n_params}, n_episodes={args.n_episodes}")
    print(f"Dream rollout: max_steps={args.max_dream_steps}, "
          f"death_threshold={args.death_threshold}")
    B_total = args.popsize * args.n_episodes
    print(f"Batch size per dream step: {B_total}")
    print()

    gen = 0
    while not es.stop():
        gen += 1
        t0 = time.time()

        candidates = es.ask()

        with torch.no_grad():
            fitnesses = evaluate_population(
                model, episodes, candidates,
                n_episodes=args.n_episodes,
                context_frames=args.context_frames,
                max_steps=args.max_dream_steps,
                death_threshold=args.death_threshold,
                device=device, rng=rng, hidden_dim=hidden_dim)

        # Percentile normalization: stabilize CMA-ES across varying episode difficulty
        if args.percentile_norm:
            f_arr = np.array(fitnesses)
            p5, p95 = np.percentile(f_arr, 5), np.percentile(f_arr, 95)
            if perc_ema_low is None:
                perc_ema_low, perc_ema_high = p5, p95
            else:
                perc_ema_low = perc_decay * perc_ema_low + (1 - perc_decay) * p5
                perc_ema_high = perc_decay * perc_ema_high + (1 - perc_decay) * p95
            scale = max(1.0, perc_ema_high - perc_ema_low)
            fitnesses = ((f_arr - perc_ema_low) / scale).tolist()

        es.tell(candidates, fitnesses)

        elapsed = time.time() - t0

        # Fitnesses are negated survival, so best = most negative
        survivals = [-f for f in fitnesses]
        best_surv = max(survivals)
        mean_surv = np.mean(survivals)
        median_surv = np.median(survivals)

        writer.writerow([gen, f"{best_surv:.2f}", f"{mean_surv:.2f}",
                         f"{median_surv:.2f}", f"{es.sigma:.4f}", f"{elapsed:.1f}"])
        log_file.flush()

        if best_surv > best_fitness_ever:
            best_fitness_ever = best_surv
            best_idx = np.argmax(survivals)
            best_params = candidates[best_idx].copy()
            ctrl = Controller(hidden_dim)
            ctrl.set_params(best_params)
            ctrl.save(ckpt_dir / "controller_best.npy")

        print(f"Gen {gen:3d} | best {best_surv:5.1f} | mean {mean_surv:5.1f} | "
              f"median {median_surv:5.1f} | sigma {es.sigma:.3f} | "
              f"best_ever {best_fitness_ever:.1f} | {elapsed:.1f}s")

    log_file.close()
    print(f"\nDone. Best survival: {best_fitness_ever:.1f} steps")
    print(f"Checkpoint: {ckpt_dir / 'controller_best.npy'}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
