"""Train controller via PPO in dream rollouts.

CNN actor-critic on z_t (token grid) + h_t projection. PPO reuses dream
rollout data for multiple gradient updates via clipped surrogate
objective. GAE advantages, survival reward.

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
from deepdash.wandb_utils import wandb_init, wandb_log, wandb_finish, wandb_run_id
from deepdash.world_model import WorldModel
from deepdash.controller import CNNPolicy


def load_episodes(episodes_dir, context_frames, vae=None, device=None):
    """Load episodes; if vae is passed, re-encode frames through it instead
    of reading tokens.npy."""
    shift_re = re.compile(r"_s[+-]\d+_[+-]\d+$")
    episodes = []
    for ep in sorted(Path(episodes_dir).glob("*")):
        if shift_re.search(ep.name):
            continue
        ap = ep / "actions.npy"
        if not ap.exists():
            continue
        actions = np.load(ap).astype(np.int64)
        if vae is not None:
            fp = ep / "frames.npy"
            if not fp.exists():
                continue
            with torch.no_grad():
                frames = np.load(fp)
                x = torch.from_numpy(frames).float().to(device) / 255.0
                x = x.unsqueeze(1)
                indices = vae.encode(x)
                tokens = indices.view(indices.size(0), -1).cpu().numpy().astype(np.int64)
        else:
            tp = ep / "tokens.npy"
            if not tp.exists():
                continue
            tokens = np.load(tp).astype(np.int64)
        if len(tokens) >= context_frames * 2:
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
        # Context starts at [0, T - 2*K], so after K context frames,
        # there is at least K frames of runway before episode end
        latest = T - K * 2
        start = rng.integers(0, latest + 1) if latest > 0 else 0
        all_ctx_tokens.append(tokens[start:start + K])
        all_ctx_actions.append(actions[start:start + K])
    return np.array(all_ctx_tokens), np.array(all_ctx_actions)


def dream_rollout(model, controller, ctx_tokens_np, ctx_actions_np,
                  max_steps, death_threshold, device, warmup_steps,
                  jump_penalty=0.0, amp_dtype=torch.bfloat16):
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

    TPF = ctx_tokens_np.shape[2]  # tokens per frame (no status)

    # Cached data for PPO
    all_z_t = []             # (T, B, TPF)
    all_h_t = []             # (T, B, 512)
    all_actions = []         # (T, B)
    all_old_log_probs = []   # (T, B)
    all_rewards = []         # (T, B)
    all_values = []          # (T, B)
    all_alive_masks = []     # (T, B)

    for step in range(max_steps):
        if not alive.any():
            break

        with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            pred_tokens, death_prob, h_t = model.predict_next_frame(
                    ctx_t, ctx_a, temperature=0.0, return_hidden=True)

        died = death_prob > death_threshold
        alive &= ~died
        survival += alive.float()

        # Controller sees current frame (last of context, strip status) + h_t
        z_t = ctx_t[:, -1, :TPF]

        with torch.no_grad():
            action, log_prob, _, value = controller.act(z_t, h_t.float())

        if step >= warmup_steps:
            alive_mask = alive.float()
            all_z_t.append(z_t.clone())
            all_h_t.append(h_t.float().clone())
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

    # Bootstrap value V(s_T) for envs still alive at truncation. Without
    # this, GAE would drop the tail value entirely on truncated rollouts
    # and bias returns low. Mask by `alive` so terminated envs contribute
    # zero bootstrap.
    if alive.any():
        with torch.no_grad(), torch.autocast(
                "cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            _, _, h_t_boot = model.predict_next_frame(
                ctx_t, ctx_a, temperature=0.0, return_hidden=True)
        z_t_boot = ctx_t[:, -1, :TPF]
        with torch.no_grad():
            _, _, _, bootstrap_value = controller.act(z_t_boot, h_t_boot.float())
        bootstrap_value = bootstrap_value * alive.float()
    else:
        bootstrap_value = torch.zeros(B, device=device)

    rollout = {
        'z_t': torch.stack(all_z_t),                      # (T, B, TPF)
        'h_t': torch.stack(all_h_t),                      # (T, B, 512)
        'actions': torch.stack(all_actions),              # (T, B)
        'old_log_probs': torch.stack(all_old_log_probs),  # (T, B)
        'rewards': torch.stack(all_rewards),              # (T, B)
        'values': torch.stack(all_values),                # (T, B)
        'alive_masks': torch.stack(all_alive_masks),      # (T, B)
        'bootstrap_value': bootstrap_value,               # (B,)
    }
    return rollout, survival


def compute_gae(rewards, values, gamma, lam, alive_masks, bootstrap_value):
    """Compute GAE advantages and returns with terminal masking.

    Args:
        rewards: (T, B)
        values: (T, B)
        alive_masks: (T, B) -- 1.0 if step t is a valid (alive) transition.
            Bootstrap from values[t+1] is gated by alive_masks[t+1] so dead
            transitions do not propagate ghost-alive value.
        bootstrap_value: (B,) -- critic value at the post-truncation state
            for rollouts that reached the horizon alive. Pass zeros if the
            caller wants the legacy "drop the tail" behavior.
    Returns:
        advantages: (T, B)
        returns: (T, B)
    """
    T, B = rewards.shape
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(B, device=rewards.device)

    for t in reversed(range(T)):
        if t == T - 1:
            next_value = bootstrap_value
            next_alive = alive_masks[t]
        else:
            next_value = values[t + 1]
            next_alive = alive_masks[t + 1]
        delta = rewards[t] + gamma * next_value * next_alive - values[t]
        gae = delta + gamma * lam * next_alive * gae
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
        # Tiny eps avoids divide-by-zero before any update; remove the
        # 1.0 floor so the scale tracks the actual return spread (the
        # floor under-scaled advantages on short rollouts where the
        # 5/95 spread is naturally below 1).
        scale = max(1e-8, self.high - self.low)
        return advantages / scale


def ppo_update(controller, optimizer, rollout, advantages, returns,
               clip_eps=0.2, entropy_coeff=0.01, critic_coeff=0.5,
               max_grad_norm=0.5, n_epochs=4, minibatch_size=None,
               pct_normalizer=None, ema_controller=None, ema_decay=0.98,
               mtp_coeff=0.0, amp_dtype=torch.bfloat16):
    """PPO update with multiple epochs on cached rollout data.

    With ``mtp_coeff > 0`` and a controller that exposes a ``mtp_head``
    (V3CNNPolicy), adds the V3-style multi-token prediction auxiliary
    loss (BCE on the next ``controller.mtp_steps`` actions).

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
    z_t_flat = rollout['z_t'].reshape(N, -1)
    h_t_flat = rollout['h_t'].reshape(N, -1)
    actions_flat = rollout['actions'].reshape(N)
    old_log_probs_flat = rollout['old_log_probs'].reshape(N)
    advantages_flat = advantages.reshape(N)
    returns_flat = returns.reshape(N)
    alive_flat = rollout['alive_masks'].reshape(N)

    # MTP targets: for each transition (t, b), the next L actions
    # (t+1..t+L-1, padded with 0 past the end). Built once per update.
    use_mtp = mtp_coeff > 0 and hasattr(controller, "mtp_head")
    mtp_targets_flat = None
    if use_mtp:
        L = controller.mtp_steps
        actions_seq = rollout['actions']  # (T, B)
        mtp_targets = torch.zeros(T, B, L, device=actions_flat.device,
                                  dtype=torch.float32)
        for k in range(L):
            end = T - k
            if end > 0:
                mtp_targets[:end, :, k] = actions_seq[k:k + end].float()
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

            mb_z_t = z_t_flat[idx]
            mb_h_t = h_t_flat[idx]
            mb_actions = actions_flat[idx]
            mb_advantages = advantages_flat[idx]
            mb_returns = returns_flat[idx]

            use_amp = amp_dtype is not None and mb_z_t.is_cuda
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                prob, value = controller(mb_z_t, mb_h_t)
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

                loss = actor_loss + critic_coeff * critic_loss + entropy_loss

                # MTP auxiliary loss (V3): BCE on next L action probabilities
                # predicted from the same features as the actor.
                if use_mtp:
                    mtp_probs = controller.predict_future_actions(mb_z_t, mb_h_t)
                    mtp_probs = mtp_probs.clamp(1e-6, 1 - 1e-6)
                    mtp_loss = F.binary_cross_entropy(
                        mtp_probs, mtp_targets_flat[idx], reduction='mean')
                    loss = loss + mtp_coeff * mtp_loss

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
                    max_steps, death_threshold, device, amp_dtype=torch.bfloat16):
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
    for step in range(max_steps):
        if not alive.any():
            break

        with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            pred_tokens, death_prob, h_t = model.predict_next_frame(
                    ctx_t, ctx_a, temperature=0.0, return_hidden=True)

        died = death_prob > death_threshold
        alive &= ~died
        survival += alive.float()

        TPF = ctx_t.shape[2] - 1  # strip status column
        z_t = ctx_t[:, -1, :TPF]
        action = controller.act_deterministic(z_t, h_t.float())
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
    parser.add_argument("--transformer-checkpoint", default=None)
    parser.add_argument("--fsq-checkpoint", default=None,
                        help="Re-encode frames through this FSQ instead of "
                             "using on-disk tokens.npy.")
    parser.add_argument("--episodes-dir", default="data/death_episodes")
    parser.add_argument("--expert-episodes-dir", default="data/expert_episodes")
    # PPO
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr-warmup-iters", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--lam", type=float, default=None,
                        help="GAE lambda")
    parser.add_argument("--clip-eps", type=float, default=None)
    parser.add_argument("--ppo-epochs", type=int, default=None)
    parser.add_argument("--minibatch-size", type=int, default=None)
    parser.add_argument("--entropy-coeff", type=float, default=None)
    parser.add_argument("--critic-coeff", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--n-iterations", type=int, default=None)
    # Rollout
    parser.add_argument("--n-episodes", type=int, default=None)
    parser.add_argument("--max-dream-steps", type=int, default=None)
    parser.add_argument("--death-threshold", type=float, default=None)
    parser.add_argument("--jump-penalty", type=float, default=None,
                        help="Per-jump reward penalty to discourage over-jumping")
    parser.add_argument("--context-frames", type=int, default=None)
    # World model architecture (defaults from configs/v3.yaml)
    parser.add_argument("--config", default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--tokens-per-frame", type=int, default=None)
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--controller-dropout", type=float, default=None)
    parser.add_argument("--token-embed-dim", type=int, default=None)
    parser.add_argument("--temporal-dim", type=int, default=None)
    parser.add_argument("--policy-class", type=str, default=None,
                        choices=["cnn", "v3_cnn"],
                        help="Controller architecture. 'cnn' = E6.10-era "
                             "CNNPolicy (h_proj + temporal_dim). 'v3_cnn' = "
                             "V3-deploy faithful (direct h_t concat, "
                             "ReLU+MaxPool, MTP head).")
    parser.add_argument("--mtp-coeff", type=float, default=None,
                        help="Coefficient on MTP auxiliary loss in PPO. "
                             "V3 default: 0.1. Set to 0 to disable. Only "
                             "active when controller has an mtp_head "
                             "(V3CNNPolicy).")
    parser.add_argument("--mtp-steps", type=int, default=None,
                        help="Number of future actions predicted by the MTP "
                             "head. V3 default: 8.")
    parser.add_argument("--compile-mode", type=str, default=None,
                        choices=["reduce-overhead", "default", "none"],
                        help="torch.compile mode for both the world model "
                             "and the controller. Default reduce-overhead.")
    # Output / initialization
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to BC-pretrained controller checkpoint")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Force cold-start (ignore pretrained in config)")
    parser.add_argument("--wandb-name", type=str, default=None,
                        help="Override wandb run name (default: ppo-{embed_dim}d)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint and append to CSV log")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--n-eval-episodes", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    from deepdash.config import apply_config
    apply_config(args, section="controller_ppo")
    if args.no_pretrained:
        args.pretrained = None

    # Optional knobs that older configs (e.g. v7-phase0) omit. Default
    # them here so downstream None comparisons can't crash.
    if args.lr_warmup_iters is None:
        args.lr_warmup_iters = 0
    if args.jump_penalty is None:
        args.jump_penalty = 0.0

    # Hard-block controller dropout under PPO. The eval-rollout vs
    # train-update asymmetry turns the importance ratio into a measure
    # of dropout-mask noise (see feedback_ppo_no_dropout.md). CNNPolicy
    # has no dropout module today, but a future config could add one.
    assert float(getattr(args, "controller_dropout", 0.0) or 0.0) == 0.0, (
        "controller_dropout must be 0 under PPO; eval/train mask asymmetry "
        "collapses entropy. See feedback_ppo_no_dropout.md."
    )

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
        adaln=getattr(args, 'adaln', False),
        fsq_dim=len(args.levels) if getattr(args, 'levels', None) else None,
    ).to(device)
    state = torch.load(args.transformer_checkpoint, map_location=device,
                       weights_only=True)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # AMP dtype: bfloat16 on A100+, float16 on Turing (RTX 2060 etc.)
    amp_dtype_str = getattr(args, 'amp_dtype', 'bfloat16')
    amp_dtype = getattr(torch, amp_dtype_str, torch.bfloat16)
    print(f"AMP dtype: {amp_dtype}")

    compile_mode = getattr(args, "compile_mode", None) or "reduce-overhead"
    if compile_mode != "none":
        try:
            model = torch.compile(model, mode=compile_mode)
            print(f"torch.compile enabled on world model (mode={compile_mode})")
        except Exception as e:
            print(f"torch.compile not available: {e}")

    vae = None
    if args.fsq_checkpoint is not None:
        from deepdash.fsq import FSQVAE
        vae = FSQVAE(levels=args.levels).to(device)
        fsq_state = torch.load(args.fsq_checkpoint, map_location=device,
                               weights_only=True)
        fsq_state = {k.removeprefix("_orig_mod."): v for k, v in fsq_state.items()}
        vae.load_state_dict(fsq_state)
        vae.eval()
        print(f"FSQ loaded from {args.fsq_checkpoint}; tokens will be re-encoded on the fly")

    # Load episodes (death + expert) with global split
    from deepdash.data_split import get_val_episodes, is_val_episode
    val_set = get_val_episodes(args.episodes_dir, args.expert_episodes_dir)

    all_eps = load_episodes(args.episodes_dir, args.context_frames,
                             vae=vae, device=device) + \
              load_episodes(args.expert_episodes_dir, args.context_frames,
                             vae=vae, device=device)
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

    # CNN actor-critic. Two architectures (selected via config policy_class):
    #   "cnn"     -- E6.10-era CNNPolicy with h_proj/h_norm to temporal_dim.
    #   "v3_cnn"  -- V3-deploy faithful: direct h_t concat, ReLU+MaxPool,
    #               MTP head. Use mtp_coeff>0 to activate the MTP loss.
    grid_size = int(args.tokens_per_frame ** 0.5)
    policy_class = (getattr(args, "policy_class", None) or "cnn").lower()
    if policy_class == "v3_cnn":
        from deepdash.controller import V3CNNPolicy
        controller = V3CNNPolicy(
            vocab_size=args.vocab_size,
            grid_size=grid_size,
            token_embed_dim=getattr(args, 'token_embed_dim', 16),
            h_dim=args.embed_dim,
            mtp_steps=int(getattr(args, "mtp_steps", None) or 8),
        ).to(device)
        policy_label = "V3CNNPolicy"
        policy_extra = (f" mtp_steps={controller.mtp_steps}")
    else:
        controller = CNNPolicy(
            vocab_size=args.vocab_size,
            grid_size=grid_size,
            token_embed_dim=getattr(args, 'token_embed_dim', 16),
            h_dim=args.embed_dim,
            temporal_dim=getattr(args, 'temporal_dim', 32),
        ).to(device)
        policy_label = "CNNPolicy"
        policy_extra = f" temporal_dim={getattr(args, 'temporal_dim', 32)}"
    if args.pretrained:
        state = torch.load(args.pretrained, map_location=device,
                           weights_only=True)
        # BC may have saved with the _orig_mod. prefix from torch.compile;
        # strip just in case (no-op when already clean).
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
        controller.load_state_dict(state)
        print(f"Loaded pretrained controller from {args.pretrained}")
    n_params = sum(p.numel() for p in controller.parameters())
    print(f"Controller: {policy_label} vocab={args.vocab_size} "
          f"embed={getattr(args, 'token_embed_dim', 16)} "
          f"h_dim={args.embed_dim}{policy_extra} "
          f"({n_params:,} params, actor-critic)")

    # Deepcopy the EMA *before* compile (deepcopy of OptimizedModule is
    # not always reliable). EMA stays uncompiled — it's only used for
    # weight blending and rollouts where compile of the controller
    # is unnecessary anyway.
    import copy
    ema_controller = copy.deepcopy(controller)
    ema_controller.eval()
    for p in ema_controller.parameters():
        p.requires_grad_(False)

    if compile_mode != "none":
        try:
            controller = torch.compile(controller, mode=compile_mode)
            print(f"torch.compile enabled on controller (mode={compile_mode})")
        except Exception as e:
            print(f"controller torch.compile failed: {e}")

    pct_normalizer = PercentileNormalizer(momentum=0.99)

    optimizer = torch.optim.Adam(controller.parameters(), lr=args.lr,
                                 eps=1e-5)
    scheduler = None
    if args.lr_warmup_iters > 0:
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0,
            total_iters=args.lr_warmup_iters)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Resume: load latest checkpoint, optimizer state, and find start iteration
    start_iteration = 1
    best_eval = -float("inf")
    wandb_resume_id = None
    resume_ckpt = ckpt_dir / "controller_ppo_latest.pt"
    if args.resume and resume_ckpt.exists():
        ckpt = torch.load(resume_ckpt, map_location=device, weights_only=False)
        controller.load_state_dict(ckpt["controller"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_iteration = ckpt["iteration"] + 1
        if scheduler is not None:
            if "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
            else:
                # Back-compat: checkpoint predates scheduler persistence.
                # LinearLR.get_lr() is multiplicative on group['lr'], so the
                # naive catch-up loop re-applies the warmup ramp on top of
                # the already-warmed saved lr. Jump past warmup and pin lr.
                scheduler.last_epoch = max(start_iteration - 1,
                                           args.lr_warmup_iters)
                for group in optimizer.param_groups:
                    group["lr"] = args.lr
        best_eval = ckpt.get("best_eval", -float("inf"))
        wandb_resume_id = ckpt.get("wandb_run_id")
        # Restore RNG state for reproducible continuation
        rng = np.random.default_rng()
        rng.bit_generator.state = ckpt["rng_state"]
        if "torch_rng_state" in ckpt:
            torch.set_rng_state(ckpt["torch_rng_state"])
        if "torch_cuda_rng_state_all" in ckpt and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ckpt["torch_cuda_rng_state_all"])
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

    run_name = args.wandb_name or f"ppo-{args.embed_dim}d"
    wandb_init(project="deepdash", name=run_name,
               config=vars(args), resume_id=wandb_resume_id)

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

    warmup_str = f" (linear warmup {args.lr_warmup_iters} iters)" if args.lr_warmup_iters > 0 else " (constant)"
    print(f"\nPPO: lr={args.lr}{warmup_str}, gamma={args.gamma}, lam={args.lam}")
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
                args.max_dream_steps, args.death_threshold, device, amp_dtype)
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
            jump_penalty=args.jump_penalty,
            amp_dtype=amp_dtype)

        if rollout is None:
            print(f"Iter {iteration}: all died during warmup, skipping")
            continue

        # Compute GAE (once, reused across PPO epochs)
        with torch.no_grad():
            advantages, returns = compute_gae(
                rollout['rewards'], rollout['values'],
                args.gamma, args.lam,
                alive_masks=rollout['alive_masks'],
                bootstrap_value=rollout['bootstrap_value'])

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
            ema_controller=ema_controller,
            mtp_coeff=float(getattr(args, "mtp_coeff", 0.0) or 0.0),
            amp_dtype=amp_dtype)

        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - t0
        mean_surv = survival.mean().item()
        mean_return = rollout['rewards'].sum(dim=0).mean().item()
        alive_actions = rollout['actions'][rollout['alive_masks'].bool()]
        train_jump_ratio = alive_actions.float().mean().item() if len(alive_actions) > 0 else 0.0
        lr = optimizer.param_groups[0]["lr"]

        # Periodic evaluation
        eval_surv = ""
        jump_ratio_str = ""
        if iteration % args.eval_interval == 0:
            controller.eval()
            with torch.no_grad():
                es, jr = evaluate_fixed(
                    model, controller, fixed_eval_tokens, fixed_eval_actions,
                    args.max_dream_steps, args.death_threshold, device, amp_dtype)
            eval_surv = f"{es:.2f}"
            jump_ratio_str = f"{jr:.2f}"

            if es > best_eval:
                best_eval = es
                # Strip _orig_mod. (torch.compile prefix) so deploy and other
                # consumers can load into an uncompiled controller.
                _clean = {k.removeprefix("_orig_mod."): v
                          for k, v in controller.state_dict().items()}
                torch.save(_clean, ckpt_dir / "controller_ppo_best.pt")

        writer.writerow([
            iteration, f"{mean_surv:.2f}", f"{mean_return:.4f}",
            f"{mean_loss:.4f}", f"{mean_value:.4f}",
            f"{mean_entropy:.4f}", f"{lr:.1e}",
            eval_surv, jump_ratio_str, f"{elapsed:.1f}"])
        log_file.flush()

        log_data = {
            "iteration": iteration,
            "ppo/train/survival": mean_surv,
            "ppo/train/return": mean_return,
            "ppo/train/loss": mean_loss,
            "ppo/train/value": mean_value,
            "ppo/train/entropy": mean_entropy,
            "ppo/train/jump_ratio": train_jump_ratio,
        }
        if eval_surv:
            log_data["ppo/eval/survival"] = float(eval_surv)
            log_data["ppo/eval/jump_ratio"] = float(jump_ratio_str)
        wandb_log(log_data)

        # Save latest checkpoint for resume. Strip _orig_mod. so an
        # uncompiled-controller resume path still loads cleanly.
        if iteration % args.eval_interval == 0:
            ctrl_clean = {k.removeprefix("_orig_mod."): v
                          for k, v in controller.state_dict().items()}
            ckpt_payload = {
                "iteration": iteration,
                "controller": ctrl_clean,
                "optimizer": optimizer.state_dict(),
                "best_eval": best_eval,
                "rng_state": rng.bit_generator.state,
                "torch_rng_state": torch.get_rng_state(),
                "torch_cuda_rng_state_all": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available() else None),
                "ema_controller": ema_controller.state_dict(),
                "pct_low": pct_normalizer.low,
                "pct_high": pct_normalizer.high,
                "wandb_run_id": wandb_run_id(),
            }
            if scheduler is not None:
                ckpt_payload["scheduler"] = scheduler.state_dict()
            torch.save(ckpt_payload, ckpt_dir / "controller_ppo_latest.pt")

        eval_str = f" | eval={eval_surv} jmp={jump_ratio_str}" \
            if eval_surv else ""
        print(f"Iter {iteration:3d} | surv={mean_surv:5.1f} | "
              f"ret={mean_return:+.3f} | val={mean_value:+.3f} | "
              f"loss={mean_loss:.3f} | ent={mean_entropy:.3f} | "
              f"lr={lr:.1e}{eval_str} | {elapsed:.1f}s")

    log_file.close()
    wandb_finish()
    print(f"\nDone. Best eval survival: {best_eval:.1f}")
    _final = {k.removeprefix("_orig_mod."): v
              for k, v in controller.state_dict().items()}
    torch.save(_final, ckpt_dir / "controller_ppo_final.pt")


if __name__ == "__main__":
    main()
