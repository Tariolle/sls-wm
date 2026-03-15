"""Train controller via Reinforce policy gradients in dream rollouts.

The controller learns to map the Transformer's hidden state h_t to
jump/idle actions by maximizing survival in dreamed episodes. Dense
reward signal from death probability at each step.

Usage:
    python scripts/train_controller_reinforce.py
    python scripts/train_controller_reinforce.py --n-iterations 500 --n-episodes 64
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepdash.world_model import WorldModel
from deepdash.controller import PolicyController

# Reuse infrastructure from CMA-ES script
from scripts.train_controller import load_episodes, sample_contexts


def _unwrap(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def dream_rollout(model, controller, ctx_tokens_np, ctx_actions_np,
                  max_steps, death_threshold, device, warmup_steps):
    """Run batched dream rollouts collecting Reinforce trajectories.

    Transformer is frozen (no_grad). Controller has gradients enabled.

    Returns:
        log_probs: list of (B,) tensors per post-warmup step
        rewards: list of (B,) tensors per post-warmup step (= -death_prob)
        entropies: list of (B,) tensors per post-warmup step
        survival: (B,) float total steps survived
    """
    m = _unwrap(model)
    B = ctx_tokens_np.shape[0]

    # Prepare context with ALIVE status
    status = np.full((*ctx_tokens_np.shape[:2], 1), m.ALIVE_TOKEN, dtype=np.int64)
    ctx_with_status = np.concatenate([ctx_tokens_np, status], axis=2)
    ctx_t = torch.from_numpy(ctx_with_status).to(device)
    ctx_a = torch.from_numpy(ctx_actions_np).to(device)

    alive = torch.ones(B, dtype=torch.bool, device=device)
    survival = torch.zeros(B, dtype=torch.float32, device=device)

    all_log_probs = []
    all_rewards = []
    all_entropies = []

    use_amp = device.type == "cuda"

    for step in range(max_steps):
        if not alive.any():
            break

        # Transformer forward (frozen)
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                pred_tokens, death_prob, h_t = model.predict_next_frame(
                    ctx_t, ctx_a, temperature=0.0, return_hidden=True)

        # Death check
        died = death_prob > death_threshold
        alive &= ~died
        survival += alive.float()

        # Controller action (WITH gradients)
        h_t_float = h_t.float()
        action, log_prob, entropy = controller.act(h_t_float)

        if step >= warmup_steps:
            alive_mask = alive.float().detach()
            all_log_probs.append(log_prob * alive_mask)
            all_rewards.append(-death_prob.detach().float() * alive_mask)
            all_entropies.append(entropy * alive_mask)

        # Shift context
        new_status = torch.full((B, 1), m.ALIVE_TOKEN, dtype=torch.long, device=device)
        new_frame = torch.cat([pred_tokens, new_status], dim=1).unsqueeze(1)
        ctx_t = torch.cat([ctx_t[:, 1:], new_frame], dim=1)
        ctx_a = torch.cat([ctx_a[:, 1:], action.unsqueeze(1)], dim=1)

    return all_log_probs, all_rewards, all_entropies, survival


def compute_reinforce_loss(log_probs, rewards, entropies, gamma=0.99,
                           entropy_coeff=0.01):
    """Compute Reinforce loss with discounted returns and baseline.

    Args:
        log_probs: list of T tensors, each (B,)
        rewards: list of T tensors, each (B,)
        entropies: list of T tensors, each (B,)
        gamma: discount factor
        entropy_coeff: weight for entropy bonus

    Returns:
        loss: scalar (differentiable through log_probs)
        mean_return: float for logging
        mean_entropy: float for logging
    """
    T = len(rewards)
    if T == 0:
        return torch.tensor(0.0, requires_grad=True), 0.0, 0.0

    rewards_t = torch.stack(rewards)      # (T, B)
    log_probs_t = torch.stack(log_probs)  # (T, B)
    entropies_t = torch.stack(entropies)  # (T, B)

    # Discounted returns (backwards)
    B = rewards_t.shape[1]
    returns = torch.zeros_like(rewards_t)
    G = torch.zeros(B, device=rewards_t.device)
    for t in reversed(range(T)):
        G = rewards_t[t] + gamma * G
        returns[t] = G

    # Baseline: per-step mean across batch
    baseline = returns.mean(dim=1, keepdim=True)  # (T, 1)
    advantages = returns - baseline

    # Reinforce loss: -E[log_prob * advantage]
    policy_loss = -(log_probs_t * advantages.detach()).sum(dim=0).mean()

    # Entropy bonus (maximize entropy = minimize negative entropy)
    entropy_loss = -entropy_coeff * entropies_t.sum(dim=0).mean()

    loss = policy_loss + entropy_loss

    return loss, returns.mean().item(), entropies_t.mean().item()


def evaluate_deterministic(model, controller, episodes, n_episodes,
                           context_frames, max_steps, death_threshold,
                           device, rng):
    """Run deterministic evaluation (no sampling)."""
    m = _unwrap(model)
    ctx_tokens, ctx_actions = sample_contexts(
        episodes, n_episodes, context_frames, rng)

    B = n_episodes
    status = np.full((*ctx_tokens.shape[:2], 1), m.ALIVE_TOKEN, dtype=np.int64)
    ctx_with_status = np.concatenate([ctx_tokens, status], axis=2)
    ctx_t = torch.from_numpy(ctx_with_status).to(device)
    ctx_a = torch.from_numpy(ctx_actions).to(device)

    alive = torch.ones(B, dtype=torch.bool, device=device)
    survival = torch.zeros(B, dtype=torch.float32, device=device)
    use_amp = device.type == "cuda"

    for step in range(max_steps):
        if not alive.any():
            break

        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                pred_tokens, death_prob, h_t = model.predict_next_frame(
                    ctx_t, ctx_a, temperature=0.0, return_hidden=True)

        died = death_prob > death_threshold
        alive &= ~died
        survival += alive.float()

        action = controller.act_deterministic(h_t.float())

        new_status = torch.full((B, 1), m.ALIVE_TOKEN, dtype=torch.long, device=device)
        new_frame = torch.cat([pred_tokens, new_status], dim=1).unsqueeze(1)
        ctx_t = torch.cat([ctx_t[:, 1:], new_frame], dim=1)
        ctx_a = torch.cat([ctx_a[:, 1:], action.unsqueeze(1)], dim=1)

    return survival.mean().item()


def main():
    parser = argparse.ArgumentParser(
        description="Train controller via Reinforce in dream rollouts")
    parser.add_argument("--transformer-checkpoint",
                        default="checkpoints/transformer_best.pt")
    parser.add_argument("--episodes-dir", default="data/episodes")
    # Reinforce
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--entropy-coeff", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--n-iterations", type=int, default=500)
    # Rollout
    parser.add_argument("--n-episodes", type=int, default=64)
    parser.add_argument("--max-dream-steps", type=int, default=20)
    parser.add_argument("--death-threshold", type=float, default=0.5)
    parser.add_argument("--context-frames", type=int, default=4)
    # Controller
    parser.add_argument("--mlp-hidden", type=int, default=64)
    # Model architecture (must match checkpoint)
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--tokens-per-frame", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    # Output
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
    rng = np.random.default_rng(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load frozen world model
    model = WorldModel(
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        context_frames=args.context_frames,
        dropout=args.dropout,
        tokens_per_frame=args.tokens_per_frame,
    ).to(device)
    state = torch.load(args.transformer_checkpoint, map_location=device,
                       weights_only=True)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    if sys.platform != "win32":
        try:
            model._backbone_forward = torch.compile(model._backbone_forward)
            print("torch.compile enabled (backbone only)")
        except Exception as e:
            print(f"torch.compile not available: {e}")

    # Load episodes
    episodes = load_episodes(args.episodes_dir, args.context_frames)
    print(f"Loaded {len(episodes)} tokenized episodes")
    if not episodes:
        print("No tokenized episodes found.")
        return

    # Create controller
    controller = PolicyController(
        hidden_dim=args.embed_dim, mlp_hidden=args.mlp_hidden).to(device)
    n_params = sum(p.numel() for p in controller.parameters())
    print(f"Controller: MLP {args.embed_dim}→{args.mlp_hidden}→1 "
          f"({n_params} params)")

    optimizer = torch.optim.Adam(controller.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.n_iterations, eta_min=args.lr * 0.01)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # CSV log
    log_path = ckpt_dir / "controller_reinforce_log.csv"
    log_file = open(log_path, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["iteration", "mean_survival", "mean_return", "loss",
                     "entropy", "lr", "eval_survival", "time_s"])

    best_eval = -float("inf")

    print(f"Reinforce: lr={args.lr}, gamma={args.gamma}, "
          f"entropy_coeff={args.entropy_coeff}")
    print(f"Dream: n_episodes={args.n_episodes}, "
          f"max_steps={args.max_dream_steps}")
    print()

    for iteration in range(1, args.n_iterations + 1):
        t0 = time.time()

        # Sample near-death contexts
        ctx_tokens, ctx_actions = sample_contexts(
            episodes, args.n_episodes, args.context_frames, rng)

        # Dream rollout with gradients
        controller.train()
        log_probs, rewards, entropies, survival = dream_rollout(
            model, controller, ctx_tokens, ctx_actions,
            max_steps=args.max_dream_steps,
            death_threshold=args.death_threshold,
            device=device,
            warmup_steps=args.context_frames)

        if len(log_probs) == 0:
            print(f"Iter {iteration}: all died during warmup, skipping")
            continue

        # Reinforce loss
        loss, mean_return, mean_entropy = compute_reinforce_loss(
            log_probs, rewards, entropies,
            gamma=args.gamma, entropy_coeff=args.entropy_coeff)

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(controller.parameters(),
                                        args.max_grad_norm)
        optimizer.step()
        scheduler.step()

        elapsed = time.time() - t0
        mean_surv = survival.mean().item()
        lr = optimizer.param_groups[0]["lr"]

        # Periodic deterministic evaluation
        eval_surv = ""
        if iteration % args.eval_interval == 0:
            controller.eval()
            with torch.no_grad():
                es = evaluate_deterministic(
                    model, controller, episodes, args.n_episodes,
                    args.context_frames, args.max_dream_steps,
                    args.death_threshold, device, rng)
            eval_surv = f"{es:.2f}"

            if es > best_eval:
                best_eval = es
                torch.save(controller.state_dict(),
                           ckpt_dir / "controller_reinforce_best.pt")

        writer.writerow([
            iteration, f"{mean_surv:.2f}", f"{mean_return:.4f}",
            f"{loss.item():.4f}", f"{mean_entropy:.4f}",
            f"{lr:.1e}", eval_surv, f"{elapsed:.1f}"])
        log_file.flush()

        eval_str = f" | eval={eval_surv}" if eval_surv else ""
        print(f"Iter {iteration:3d} | surv={mean_surv:5.1f} | "
              f"ret={mean_return:+.3f} | loss={loss.item():.4f} | "
              f"ent={mean_entropy:.3f} | lr={lr:.1e}{eval_str} | "
              f"{elapsed:.1f}s")

    log_file.close()
    print(f"\nDone. Best eval survival: {best_eval:.1f}")
    torch.save(controller.state_dict(),
               ckpt_dir / "controller_reinforce_final.pt")


if __name__ == "__main__":
    main()
