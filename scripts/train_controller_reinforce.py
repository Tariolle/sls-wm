"""Train controller via actor-critic Reinforce in dream rollouts.

DART-style Transformer encoder policy with actor + critic heads.
Uniform context sampling (not near-death biased) with percentile
return normalization (DreamerV3-style) so safe-context gradients
don't drown out obstacle-encounter signal.

Usage:
    python scripts/train_controller_reinforce.py
    python scripts/train_controller_reinforce.py --n-iterations 500 --n-episodes 128
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
from deepdash.controller import TransformerPolicy

from scripts.train_controller import load_episodes


def _unwrap(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def sample_contexts_uniform(episodes, n, context_frames, rng):
    """Sample n contexts uniformly from episodes (not near-death biased).

    Returns:
        ctx_tokens: (n, K, TPF) int64
        ctx_actions: (n, K) int64
    """
    K = context_frames
    all_ctx_tokens = []
    all_ctx_actions = []
    for _ in range(n):
        ep_idx = rng.integers(len(episodes))
        tokens, actions = episodes[ep_idx]
        T = len(tokens)
        latest = T - K
        if latest <= 0:
            start = 0
        else:
            start = rng.integers(0, latest)
        all_ctx_tokens.append(tokens[start:start + K])
        all_ctx_actions.append(actions[start:start + K])
    return np.array(all_ctx_tokens), np.array(all_ctx_actions)


class PercentileNormalizer:
    """DreamerV3-style running percentile return normalization.

    Tracks EMA of 5th and 95th percentile of returns. Normalizes
    advantages to roughly [-1, 1] so gradient scale is consistent
    across safe and dangerous contexts.
    """

    def __init__(self, decay=0.99, percentile_low=5, percentile_high=95):
        self.decay = decay
        self.pct_low = percentile_low
        self.pct_high = percentile_high
        self.low = 0.0
        self.high = 0.0
        self.initialized = False

    def update_and_normalize(self, values):
        """Update percentiles and normalize values.

        Args:
            values: (T, B) tensor
        Returns:
            normalized: (T, B) tensor
        """
        flat = values.detach().flatten().cpu().numpy()
        p_low = float(np.percentile(flat, self.pct_low))
        p_high = float(np.percentile(flat, self.pct_high))

        if not self.initialized:
            self.low = p_low
            self.high = p_high
            self.initialized = True
        else:
            self.low = self.decay * self.low + (1 - self.decay) * p_low
            self.high = self.decay * self.high + (1 - self.decay) * p_high

        scale = max(self.high - self.low, 1e-8)
        return (values - self.low) / scale


def dream_rollout(model, controller, ctx_tokens_np, ctx_actions_np,
                  max_steps, death_threshold, device, warmup_steps):
    """Run batched dream rollouts collecting actor-critic trajectories.

    Returns:
        log_probs: list of (B,) tensors per post-warmup step
        rewards: list of (B,) tensors per post-warmup step (= -death_prob)
        entropies: list of (B,) tensors per post-warmup step
        values: list of (B,) tensors per post-warmup step (critic estimates)
        survival: (B,) float total steps survived
    """
    m = _unwrap(model)
    B = ctx_tokens_np.shape[0]

    status = np.full((*ctx_tokens_np.shape[:2], 1), m.ALIVE_TOKEN, dtype=np.int64)
    ctx_with_status = np.concatenate([ctx_tokens_np, status], axis=2)
    ctx_t = torch.from_numpy(ctx_with_status).to(device)
    ctx_a = torch.from_numpy(ctx_actions_np).to(device)

    alive = torch.ones(B, dtype=torch.bool, device=device)
    survival = torch.zeros(B, dtype=torch.float32, device=device)

    all_log_probs = []
    all_rewards = []
    all_entropies = []
    all_values = []

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

        with torch.no_grad():
            tok_embeds = m.token_embed(pred_tokens).float()

        action, log_prob, entropy, value = controller.act(
            tok_embeds, h_t.float())

        if step >= warmup_steps:
            alive_mask = alive.float().detach()
            all_log_probs.append(log_prob * alive_mask)
            all_rewards.append(-death_prob.detach().float() * alive_mask)
            all_entropies.append(entropy * alive_mask)
            all_values.append(value * alive_mask)

        new_status = torch.full((B, 1), m.ALIVE_TOKEN, dtype=torch.long,
                                device=device)
        new_frame = torch.cat([pred_tokens, new_status], dim=1).unsqueeze(1)
        ctx_t = torch.cat([ctx_t[:, 1:], new_frame], dim=1)
        ctx_a = torch.cat([ctx_a[:, 1:], action.unsqueeze(1)], dim=1)

    return all_log_probs, all_rewards, all_entropies, all_values, survival


def compute_actor_critic_loss(log_probs, rewards, entropies, values,
                              normalizer, gamma=0.99, lam=0.95,
                              entropy_coeff=0.01, critic_coeff=0.5):
    """Compute actor-critic loss with GAE and percentile normalization.

    Returns:
        loss: scalar
        mean_return: float
        mean_entropy: float
        mean_value: float
    """
    T = len(rewards)
    if T == 0:
        return torch.tensor(0.0, requires_grad=True), 0.0, 0.0, 0.0

    rewards_t = torch.stack(rewards)
    log_probs_t = torch.stack(log_probs)
    entropies_t = torch.stack(entropies)
    values_t = torch.stack(values)

    # GAE
    B = rewards_t.shape[1]
    advantages = torch.zeros_like(rewards_t)
    gae = torch.zeros(B, device=rewards_t.device)

    for t in reversed(range(T)):
        if t == T - 1:
            next_value = torch.zeros(B, device=rewards_t.device)
        else:
            next_value = values_t[t + 1].detach()
        delta = rewards_t[t] + gamma * next_value - values_t[t].detach()
        gae = delta + gamma * lam * gae
        advantages[t] = gae

    returns = advantages + values_t.detach()

    # Percentile normalization of advantages
    advantages = normalizer.update_and_normalize(advantages)

    # Actor loss
    actor_loss = -(log_probs_t * advantages.detach()).sum(dim=0).mean()

    # Critic loss
    critic_loss = ((values_t - returns) ** 2).sum(dim=0).mean()

    # Entropy bonus
    entropy_loss = -entropy_coeff * entropies_t.sum(dim=0).mean()

    loss = actor_loss + critic_coeff * critic_loss + entropy_loss

    return loss, returns.mean().item(), entropies_t.mean().item(), \
        values_t.mean().item()


def evaluate_deterministic(model, controller, episodes, n_episodes,
                           context_frames, max_steps, death_threshold,
                           device, rng):
    """Run deterministic evaluation with uniform sampling."""
    m = _unwrap(model)
    ctx_tokens, ctx_actions = sample_contexts_uniform(
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

        with torch.no_grad():
            tok_embeds = m.token_embed(pred_tokens).float()
        action = controller.act_deterministic(tok_embeds, h_t.float())

        new_status = torch.full((B, 1), m.ALIVE_TOKEN, dtype=torch.long,
                                device=device)
        new_frame = torch.cat([pred_tokens, new_status], dim=1).unsqueeze(1)
        ctx_t = torch.cat([ctx_t[:, 1:], new_frame], dim=1)
        ctx_a = torch.cat([ctx_a[:, 1:], action.unsqueeze(1)], dim=1)

    return survival.mean().item()


def main():
    parser = argparse.ArgumentParser(
        description="Train controller via actor-critic Reinforce")
    parser.add_argument("--transformer-checkpoint",
                        default="checkpoints/transformer_best.pt")
    parser.add_argument("--episodes-dir", default="data/episodes")
    # Actor-critic
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95,
                        help="GAE lambda")
    parser.add_argument("--entropy-coeff", type=float, default=0.01)
    parser.add_argument("--critic-coeff", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--n-iterations", type=int, default=500)
    # Rollout
    parser.add_argument("--n-episodes", type=int, default=128)
    parser.add_argument("--max-dream-steps", type=int, default=22)
    parser.add_argument("--death-threshold", type=float, default=0.5)
    parser.add_argument("--context-frames", type=int, default=4)
    # Policy architecture
    parser.add_argument("--policy-embed-dim", type=int, default=128)
    parser.add_argument("--policy-n-heads", type=int, default=4)
    parser.add_argument("--policy-n-layers", type=int, default=3)
    parser.add_argument("--policy-dropout", type=float, default=0.1)
    # World model architecture (must match checkpoint)
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

    # Create DART-style Transformer actor-critic policy
    controller = TransformerPolicy(
        wm_embed_dim=args.embed_dim,
        n_tokens=args.tokens_per_frame,
        embed_dim=args.policy_embed_dim,
        n_heads=args.policy_n_heads,
        n_layers=args.policy_n_layers,
        dropout=args.policy_dropout,
    ).to(device)
    n_params = sum(p.numel() for p in controller.parameters())
    print(f"Controller: TransformerPolicy {args.policy_n_layers}L/"
          f"{args.policy_n_heads}H/{args.policy_embed_dim}d "
          f"({n_params:,} params, actor-critic)")

    optimizer = torch.optim.Adam(controller.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.n_iterations, eta_min=args.lr * 0.01)

    normalizer = PercentileNormalizer()

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_path = ckpt_dir / "controller_reinforce_log.csv"
    log_file = open(log_path, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["iteration", "mean_survival", "mean_return",
                     "loss", "mean_value", "entropy", "lr",
                     "eval_survival", "time_s"])

    best_eval = -float("inf")

    print(f"\nActor-Critic: lr={args.lr}, gamma={args.gamma}, "
          f"lam={args.lam}")
    print(f"Coefficients: entropy={args.entropy_coeff}, "
          f"critic={args.critic_coeff}")
    print(f"Dream: n_episodes={args.n_episodes}, "
          f"max_steps={args.max_dream_steps}")
    print(f"Sampling: uniform (not near-death)")
    print(f"Normalization: percentile (5th/95th EMA)\n")

    for iteration in range(1, args.n_iterations + 1):
        t0 = time.time()

        # Uniform context sampling
        ctx_tokens, ctx_actions = sample_contexts_uniform(
            episodes, args.n_episodes, args.context_frames, rng)

        controller.train()
        log_probs, rewards, entropies, values, survival = dream_rollout(
            model, controller, ctx_tokens, ctx_actions,
            max_steps=args.max_dream_steps,
            death_threshold=args.death_threshold,
            device=device,
            warmup_steps=args.context_frames)

        if len(log_probs) == 0:
            print(f"Iter {iteration}: all died during warmup, skipping")
            continue

        loss, mean_return, mean_entropy, mean_value = \
            compute_actor_critic_loss(
                log_probs, rewards, entropies, values, normalizer,
                gamma=args.gamma, lam=args.lam,
                entropy_coeff=args.entropy_coeff,
                critic_coeff=args.critic_coeff)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(controller.parameters(),
                                        args.max_grad_norm)
        optimizer.step()
        scheduler.step()

        elapsed = time.time() - t0
        mean_surv = survival.mean().item()
        lr = optimizer.param_groups[0]["lr"]

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
            f"{loss.item():.4f}", f"{mean_value:.4f}",
            f"{mean_entropy:.4f}", f"{lr:.1e}", eval_surv,
            f"{elapsed:.1f}"])
        log_file.flush()

        eval_str = f" | eval={eval_surv}" if eval_surv else ""
        print(f"Iter {iteration:3d} | surv={mean_surv:5.1f} | "
              f"ret={mean_return:+.3f} | val={mean_value:+.3f} | "
              f"loss={loss.item():.3f} | ent={mean_entropy:.3f} | "
              f"lr={lr:.1e}{eval_str} | {elapsed:.1f}s")

    log_file.close()
    print(f"\nDone. Best eval survival: {best_eval:.1f}")
    torch.save(controller.state_dict(),
               ckpt_dir / "controller_reinforce_final.pt")


if __name__ == "__main__":
    main()
