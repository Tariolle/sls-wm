"""Train controller via PPO in dream rollouts.

CNN actor-critic policy on 8x8 token grid. PPO reuses dream rollout
data for multiple gradient updates via clipped surrogate objective.
GAE advantages, survival reward.

Usage:
    python scripts/train_controller_ppo.py
    python scripts/train_controller_ppo.py --n-iterations 20000
"""

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepdash.world_model import WorldModel
from deepdash.controller import CNNPolicy


def load_episodes(episodes_dir, context_frames):
    """Load base (non-shifted) tokenized episodes with enough frames."""
    shift_re = re.compile(r"_s[+-]\d+_[+-]\d+$")
    episodes = []
    for ep in sorted(Path(episodes_dir).glob("*")):
        if shift_re.search(ep.name):
            continue
        tp = ep / "tokens.npy"
        ap = ep / "actions.npy"
        if not tp.exists() or not ap.exists():
            continue
        tokens = np.load(tp).astype(np.int64)
        actions = np.load(ap).astype(np.int64)
        if len(tokens) >= context_frames * 3:
            episodes.append((tokens, actions, ep.name))
    return episodes


def _unwrap(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def sample_contexts(episodes, n, context_frames, rng):
    """Sample n contexts uniformly, excluding last 2*K frames of each episode.

    The last 2*K frames are cut because:
    - In death episodes: outcome is already determined (too late to react)
    - In expert episodes: win animation noise
    This gives the controller at least K dream frames of runway to influence
    the outcome after context.
    """
    K = context_frames
    all_ctx_tokens = []
    all_ctx_actions = []
    for _ in range(n):
        ep_idx = rng.integers(len(episodes))
        tokens, actions = episodes[ep_idx]
        T = len(tokens)
        # Context starts at [0, T - 3*K], so after K context frames,
        # there are at least 2*K frames of runway before episode end
        latest = T - K * 3
        start = rng.integers(0, latest + 1) if latest > 0 else 0
        all_ctx_tokens.append(tokens[start:start + K])
        all_ctx_actions.append(actions[start:start + K])
    return np.array(all_ctx_tokens), np.array(all_ctx_actions)


def dream_rollout(model, controller, ctx_tokens_np, ctx_actions_np,
                  max_steps, death_threshold, device, warmup_steps,
                  jump_penalty=0.0):
    """Roll out dreams and cache data for PPO updates.

    Hard death cutoff: rollout ends for an episode when death_prob > threshold.

    Returns:
        rollout: dict with cached tensors for PPO
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

    # Cached data for PPO
    all_token_ids = []       # (T, B, 64)
    all_h_t = []             # (T, B, 256)
    all_actions = []         # (T, B)
    all_old_log_probs = []   # (T, B)
    all_rewards = []         # (T, B)
    all_values = []          # (T, B)
    all_alive_masks = []     # (T, B)

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
            action, log_prob, _, value = controller.act(
                pred_tokens, h_t.float())

        if step >= warmup_steps:
            alive_mask = alive.float()
            all_token_ids.append(pred_tokens)
            all_h_t.append(h_t.float())
            all_actions.append(action)
            all_old_log_probs.append(log_prob)
            reward = alive_mask.clone()
            if jump_penalty > 0:
                reward -= jump_penalty * action.float() * alive_mask
            all_rewards.append(reward)
            all_values.append(value)
            all_alive_masks.append(alive_mask)

        new_status = torch.full((B, 1), m.ALIVE_TOKEN, dtype=torch.long,
                                device=device)
        new_frame = torch.cat([pred_tokens, new_status], dim=1).unsqueeze(1)
        ctx_t = torch.cat([ctx_t[:, 1:], new_frame], dim=1)
        ctx_a = torch.cat([ctx_a[:, 1:], action.unsqueeze(1)], dim=1)

    if not all_rewards:
        return None, survival

    rollout = {
        'token_ids': torch.stack(all_token_ids),         # (T, B, 64)
        'h_t': torch.stack(all_h_t),                     # (T, B, 256)
        'actions': torch.stack(all_actions),              # (T, B)
        'old_log_probs': torch.stack(all_old_log_probs),  # (T, B)
        'rewards': torch.stack(all_rewards),              # (T, B)
        'values': torch.stack(all_values),                # (T, B)
        'alive_masks': torch.stack(all_alive_masks),      # (T, B)
    }
    return rollout, survival


def compute_gae(rewards, values, gamma, lam):
    """Compute GAE advantages and returns.

    Args:
        rewards: (T, B)
        values: (T, B)
    Returns:
        advantages: (T, B)
        returns: (T, B)
    """
    T, B = rewards.shape
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(B, device=rewards.device)

    for t in reversed(range(T)):
        if t == T - 1:
            next_value = torch.zeros(B, device=rewards.device)
        else:
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns


class PercentileNormalizer:
    """EMA-based percentile advantage normalization (DreamerV3/TWISTER).

    Tracks the 5th-95th percentile range of returns via EMA and normalizes
    advantages by this range. Prevents outlier returns from dominating.
    """
    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.low = None
        self.high = None

    def update(self, returns):
        p5 = torch.quantile(returns, 0.05).item()
        p95 = torch.quantile(returns, 0.95).item()
        if self.low is None:
            self.low, self.high = p5, p95
        else:
            self.low = self.momentum * self.low + (1 - self.momentum) * p5
            self.high = self.momentum * self.high + (1 - self.momentum) * p95

    def normalize(self, advantages):
        scale = max(1.0, self.high - self.low)
        return advantages / scale


def pmpo_actor_loss(log_probs, advantages, alpha=0.5):
    """PMPO policy loss (Dreamer 4): uses sign of advantages.

    Splits transitions into positive (A >= 0) and negative (A < 0) sets,
    averages log-likelihood separately, then combines. Ignores advantage
    magnitude, giving equal focus to all improving/worsening actions.
    """
    pos_mask = advantages >= 0
    neg_mask = ~pos_mask

    loss = torch.tensor(0.0, device=log_probs.device)
    if pos_mask.any():
        loss = loss - alpha * log_probs[pos_mask].mean()
    if neg_mask.any():
        loss = loss + (1 - alpha) * log_probs[neg_mask].mean()
    return loss


def ppo_update(controller, optimizer, rollout, advantages, returns,
               clip_eps=0.2, entropy_coeff=0.01, critic_coeff=0.5,
               max_grad_norm=0.5, n_epochs=4, minibatch_size=None,
               pct_normalizer=None, ema_controller=None, ema_decay=0.98,
               mtp_coeff=0.1):
    """PMPO + MTP update with multiple epochs on cached rollout data.

    Uses PMPO (sign-based advantages) instead of clipped PPO objective.
    Adds multi-token prediction auxiliary loss for better timing.

    Returns:
        mean_loss, mean_entropy, mean_value
    """
    T, B = rollout['rewards'].shape
    N = T * B

    if minibatch_size is None:
        minibatch_size = N

    # Update percentile normalizer with current returns
    if pct_normalizer is not None:
        pct_normalizer.update(returns.reshape(-1))

    # Flatten rollout for minibatch sampling
    token_ids_flat = rollout['token_ids'].reshape(N, -1)
    h_t_flat = rollout['h_t'].reshape(N, -1)
    actions_flat = rollout['actions'].reshape(N)
    old_log_probs_flat = rollout['old_log_probs'].reshape(N)
    advantages_flat = advantages.reshape(N)
    returns_flat = returns.reshape(N)
    alive_flat = rollout['alive_masks'].reshape(N)

    # Build MTP targets: for each transition at time t, the future actions
    # at t+1, t+2, ..., t+L-1 (padded with 0 if out of bounds)
    L = controller.mtp_steps
    actions_seq = rollout['actions']  # (T, B)
    mtp_targets = torch.zeros(T, B, L, device=actions_flat.device)
    for k in range(L):
        end = T - k
        if end > 0:
            mtp_targets[:end, :, k] = actions_seq[k:k + end]
    mtp_targets_flat = mtp_targets.reshape(N, L)

    # Only train on alive transitions
    alive_idx = alive_flat.nonzero(as_tuple=True)[0]
    if len(alive_idx) == 0:
        return 0.0, 0.0, 0.0

    total_loss = 0.0
    total_entropy = 0.0
    total_value = 0.0
    n_updates = 0

    for epoch in range(n_epochs):
        perm = alive_idx[torch.randperm(len(alive_idx), device=alive_idx.device)]

        for start in range(0, len(perm), minibatch_size):
            idx = perm[start:start + minibatch_size]

            mb_token_ids = token_ids_flat[idx]
            mb_h_t = h_t_flat[idx]
            mb_actions = actions_flat[idx]
            mb_advantages = advantages_flat[idx]
            mb_returns = returns_flat[idx]
            mb_mtp_targets = mtp_targets_flat[idx]

            # Forward pass with current policy
            prob, value = controller(mb_token_ids, mb_h_t)
            prob = prob.clamp(1e-6, 1 - 1e-6)
            dist = torch.distributions.Bernoulli(probs=prob)
            log_prob = dist.log_prob(mb_actions.float())
            entropy = dist.entropy()

            # Clipped surrogate objective
            ratio = (log_prob - old_log_probs_flat[idx]).exp()
            if pct_normalizer is not None:
                norm_adv = pct_normalizer.normalize(mb_advantages)
            else:
                norm_adv = (mb_advantages - mb_advantages.mean()) / \
                    (mb_advantages.std() + 1e-8)
            surr1 = ratio * norm_adv
            surr2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * norm_adv
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic loss
            critic_loss = F.mse_loss(value, mb_returns)

            # Entropy bonus
            entropy_loss = -entropy_coeff * entropy.mean()

            # MTP auxiliary loss: predict future actions
            mtp_probs = controller.predict_future_actions(
                mb_token_ids, mb_h_t)
            mtp_loss = F.binary_cross_entropy(
                mtp_probs, mb_mtp_targets, reduction='mean')

            loss = actor_loss + critic_coeff * critic_loss + \
                entropy_loss + mtp_coeff * mtp_loss

            # Skip NaN updates
            if torch.isnan(loss) or torch.isinf(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(controller.parameters(),
                                            max_grad_norm)
            optimizer.step()

            # Update EMA target critic
            if ema_controller is not None:
                with torch.no_grad():
                    for p, p_ema in zip(controller.parameters(),
                                        ema_controller.parameters()):
                        p_ema.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)

            total_loss += loss.item()
            total_entropy += entropy.mean().item()
            total_value += value.mean().item()
            n_updates += 1

    n_updates = max(n_updates, 1)
    return total_loss / n_updates, total_entropy / n_updates, \
        total_value / n_updates


def evaluate_fixed(model, controller, ctx_tokens_np, ctx_actions_np,
                    max_steps, death_threshold, device):
    """Run deterministic evaluation on fixed pre-sampled contexts."""
    m = _unwrap(model)
    B = ctx_tokens_np.shape[0]

    status = np.full((*ctx_tokens_np.shape[:2], 1), m.ALIVE_TOKEN, dtype=np.int64)
    ctx_with_status = np.concatenate([ctx_tokens_np, status], axis=2)
    ctx_t = torch.from_numpy(ctx_with_status).to(device)
    ctx_a = torch.from_numpy(ctx_actions_np).to(device)

    alive = torch.ones(B, dtype=torch.bool, device=device)
    survival = torch.zeros(B, dtype=torch.float32, device=device)
    total_jumps = 0
    total_actions = 0
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

        action = controller.act_deterministic(pred_tokens, h_t.float())
        total_jumps += (action[alive] == 1).sum().item()
        total_actions += alive.sum().item()

        new_status = torch.full((B, 1), m.ALIVE_TOKEN, dtype=torch.long,
                                device=device)
        new_frame = torch.cat([pred_tokens, new_status], dim=1).unsqueeze(1)
        ctx_t = torch.cat([ctx_t[:, 1:], new_frame], dim=1)
        ctx_a = torch.cat([ctx_a[:, 1:], action.unsqueeze(1)], dim=1)

    jump_ratio = total_jumps / max(total_actions, 1)
    return survival.mean().item(), jump_ratio


def main():
    parser = argparse.ArgumentParser(
        description="Train controller via PPO in dream rollouts")
    parser.add_argument("--transformer-checkpoint",
                        default="checkpoints/transformer_best.pt")
    parser.add_argument("--episodes-dir", default="data/death_episodes")
    parser.add_argument("--expert-episodes-dir", default="data/expert_episodes")
    # PPO
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--lam", type=float, default=0.95,
                        help="GAE lambda")
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--entropy-coeff", type=float, default=0.01)
    parser.add_argument("--critic-coeff", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--n-iterations", type=int, default=5000)
    # Rollout
    parser.add_argument("--n-episodes", type=int, default=512)
    parser.add_argument("--max-dream-steps", type=int, default=30)
    parser.add_argument("--death-threshold", type=float, default=0.5)
    parser.add_argument("--jump-penalty", type=float, default=0.2,
                        help="Per-jump reward penalty to discourage over-jumping")
    parser.add_argument("--context-frames", type=int, default=4)
    # Policy architecture (CNN)
    parser.add_argument("--token-embed-dim", type=int, default=16)
    # World model architecture (must match checkpoint)
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--tokens-per-frame", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    # Output / initialization
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to BC-pretrained controller checkpoint")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint and append to CSV log")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--n-eval-episodes", type=int, default=512)
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
            model = torch.compile(model)
            print("torch.compile enabled (full model)")
        except Exception as e:
            print(f"torch.compile not available: {e}")

    # Load episodes (death + expert) with global split
    from deepdash.data_split import get_val_episodes, is_val_episode
    val_set = get_val_episodes(args.episodes_dir, args.expert_episodes_dir)

    all_eps = load_episodes(args.episodes_dir, args.context_frames) + \
              load_episodes(args.expert_episodes_dir, args.context_frames)
    # Train on all episodes, eval on val only
    train_episodes = [(t, a) for t, a, name in all_eps
                      if not is_val_episode(name, val_set)]
    val_episodes = [(t, a) for t, a, name in all_eps
                    if is_val_episode(name, val_set)]
    episodes = [(t, a) for t, a, name in all_eps]  # all for training rollouts
    print(f"Loaded {len(episodes)} episodes "
          f"({len(train_episodes)} train, {len(val_episodes)} val)")
    if not episodes:
        print("No tokenized episodes found.")
        return

    # CNN actor-critic policy on 8x8 token grid
    controller = CNNPolicy(
        vocab_size=args.vocab_size,
        grid_size=int(args.tokens_per_frame ** 0.5),
        token_embed_dim=args.token_embed_dim,
        h_dim=args.embed_dim,
    ).to(device)
    if args.pretrained:
        state = torch.load(args.pretrained, map_location=device,
                           weights_only=True)
        controller.load_state_dict(state)
        print(f"Loaded pretrained controller from {args.pretrained}")
    n_params = sum(p.numel() for p in controller.parameters())
    print(f"Controller: CNNPolicy embed={args.token_embed_dim} "
          f"({n_params:,} params, actor-critic)")

    import copy
    ema_controller = copy.deepcopy(controller)
    ema_controller.eval()
    for p in ema_controller.parameters():
        p.requires_grad_(False)

    pct_normalizer = PercentileNormalizer(momentum=0.99)

    optimizer = torch.optim.Adam(controller.parameters(), lr=args.lr,
                                 eps=1e-5)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Resume: load latest checkpoint, optimizer state, and find start iteration
    start_iteration = 1
    best_eval = -float("inf")
    resume_ckpt = ckpt_dir / "controller_ppo_latest.pt"
    if args.resume and resume_ckpt.exists():
        ckpt = torch.load(resume_ckpt, map_location=device, weights_only=False)
        controller.load_state_dict(ckpt["controller"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_iteration = ckpt["iteration"] + 1
        best_eval = ckpt.get("best_eval", -float("inf"))
        # Restore RNG state for reproducible continuation
        rng = np.random.default_rng()
        rng.bit_generator.state = ckpt["rng_state"]
        # Restore EMA controller and percentile normalizer
        if "ema_controller" in ckpt:
            ema_controller.load_state_dict(ckpt["ema_controller"])
        if "pct_low" in ckpt:
            pct_normalizer.low = ckpt["pct_low"]
            pct_normalizer.high = ckpt["pct_high"]
        print(f"Resumed from iteration {ckpt['iteration']} "
              f"(best_eval={best_eval:.2f})")

    if start_iteration == 1:
        import json
        with open(ckpt_dir / "controller_ppo_args.json", "w") as f:
            json.dump(vars(args), f, indent=2)

    # Fixed eval contexts (from val episodes only)
    # Use a dedicated RNG so eval contexts are identical across resumes
    eval_rng = np.random.default_rng(args.seed)
    print(f"\nPre-sampling fixed eval contexts (val episodes)...")
    eval_source = val_episodes if val_episodes else episodes
    fixed_eval_tokens, fixed_eval_actions = sample_contexts(
        eval_source, args.n_eval_episodes, args.context_frames, eval_rng)
    print(f"  Eval: {args.n_eval_episodes} fixed contexts from {len(eval_source)} val episodes")

    log_path = ckpt_dir / "controller_ppo_log.csv"
    if args.resume and log_path.exists() and start_iteration > 1:
        log_file = open(log_path, "a", newline="")
        writer = csv.writer(log_file)
    else:
        log_file = open(log_path, "w", newline="")
        writer = csv.writer(log_file)
        writer.writerow(["iteration", "mean_survival", "mean_return",
                         "loss", "mean_value", "entropy", "lr",
                         "eval_survival", "jump_ratio", "time_s"])

    print(f"\nPPO: lr={args.lr} (constant), gamma={args.gamma}, lam={args.lam}")
    print(f"PPO: clip_eps={args.clip_eps}, epochs={args.ppo_epochs}, "
          f"minibatch={args.minibatch_size}")
    print(f"Entropy: {args.entropy_coeff} (fixed)")
    print(f"Dream: n_episodes={args.n_episodes}, "
          f"max_steps={args.max_dream_steps}")
    print(f"Sampling: uniform (excluding last 2*K frames)")
    print(f"Eval: {args.n_eval_episodes} fixed contexts\n")

    # Pre-training eval (BC baseline before any PPO updates)
    if start_iteration == 1:
        controller.eval()
        with torch.no_grad():
            bc_surv, bc_jr = evaluate_fixed(
                model, controller, fixed_eval_tokens, fixed_eval_actions,
                args.max_dream_steps, args.death_threshold, device)
        writer.writerow([
            0, "", "", "", "", "", f"{optimizer.param_groups[0]['lr']:.1e}",
            f"{bc_surv:.2f}", f"{bc_jr:.2f}", "0.0"])
        log_file.flush()
        print(f"BC baseline eval: survival={bc_surv:.2f}, jump_ratio={bc_jr:.2f}\n")

    for iteration in range(start_iteration, args.n_iterations + 1):
        t0 = time.time()

        # Uniform training contexts (excluding last 2*K frames)
        ctx_tokens, ctx_actions = sample_contexts(
            episodes, args.n_episodes, args.context_frames, rng)

        # Dream rollout (no gradients, cache data)
        controller.eval()
        rollout, survival = dream_rollout(
            model, controller, ctx_tokens, ctx_actions,
            max_steps=args.max_dream_steps,
            death_threshold=args.death_threshold,
            device=device,
            warmup_steps=args.context_frames,
            jump_penalty=args.jump_penalty)

        if rollout is None:
            print(f"Iter {iteration}: all died during warmup, skipping")
            continue

        # Compute GAE (once, reused across PPO epochs)
        with torch.no_grad():
            advantages, returns = compute_gae(
                rollout['rewards'], rollout['values'],
                args.gamma, args.lam)

        # PPO update (4 epochs on cached data)
        controller.train()
        mean_loss, mean_entropy, mean_value = ppo_update(
            controller, optimizer, rollout, advantages, returns,
            clip_eps=args.clip_eps, entropy_coeff=args.entropy_coeff,
            critic_coeff=args.critic_coeff,
            max_grad_norm=args.max_grad_norm,
            n_epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
            pct_normalizer=pct_normalizer,
            ema_controller=ema_controller)

        elapsed = time.time() - t0
        mean_surv = survival.mean().item()
        mean_return = rollout['rewards'].sum(dim=0).mean().item()
        lr = optimizer.param_groups[0]["lr"]

        # Periodic evaluation
        eval_surv = ""
        jump_ratio_str = ""
        if iteration % args.eval_interval == 0:
            controller.eval()
            with torch.no_grad():
                es, jr = evaluate_fixed(
                    model, controller, fixed_eval_tokens, fixed_eval_actions,
                    args.max_dream_steps, args.death_threshold, device)
            eval_surv = f"{es:.2f}"
            jump_ratio_str = f"{jr:.2f}"

            if es > best_eval:
                best_eval = es
                torch.save(controller.state_dict(),
                           ckpt_dir / "controller_ppo_best.pt")

        writer.writerow([
            iteration, f"{mean_surv:.2f}", f"{mean_return:.4f}",
            f"{mean_loss:.4f}", f"{mean_value:.4f}",
            f"{mean_entropy:.4f}", f"{lr:.1e}",
            eval_surv, jump_ratio_str, f"{elapsed:.1f}"])
        log_file.flush()

        # Save latest checkpoint for resume
        if iteration % args.eval_interval == 0:
            torch.save({
                "iteration": iteration,
                "controller": controller.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_eval": best_eval,
                "rng_state": rng.bit_generator.state,
                "ema_controller": ema_controller.state_dict(),
                "pct_low": pct_normalizer.low,
                "pct_high": pct_normalizer.high,
            }, ckpt_dir / "controller_ppo_latest.pt")

        eval_str = f" | eval={eval_surv} jmp={jump_ratio_str}" \
            if eval_surv else ""
        print(f"Iter {iteration:3d} | surv={mean_surv:5.1f} | "
              f"ret={mean_return:+.3f} | val={mean_value:+.3f} | "
              f"loss={mean_loss:.3f} | ent={mean_entropy:.3f} | "
              f"lr={lr:.1e}{eval_str} | {elapsed:.1f}s")

    log_file.close()
    print(f"\nDone. Best eval survival: {best_eval:.1f}")
    torch.save(controller.state_dict(),
               ckpt_dir / "controller_ppo_final.pt")


if __name__ == "__main__":
    main()
