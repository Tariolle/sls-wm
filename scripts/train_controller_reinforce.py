"""Train controller via BC + Reinforce in dream rollouts.

Combined loss: alpha * BC_loss + (1-alpha) * RL_loss
- BC loss: binary cross-entropy on real episode actions (timing signal)
- RL loss: Reinforce with dense reward in dream rollouts (generalization)
- alpha decays with cosine schedule from bc-weight-start to bc-weight-end

Uses a DART-style Transformer encoder policy that sees individual tokens
with learnable positional encoding, enabling spatial attention for
timing-critical decisions.

Usage:
    python scripts/train_controller_reinforce.py
    python scripts/train_controller_reinforce.py --n-iterations 500 --n-episodes 64
"""

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepdash.world_model import WorldModel
from deepdash.controller import TransformerPolicy

# Reuse infrastructure from CMA-ES script
from scripts.train_controller import load_episodes, sample_contexts


def _unwrap(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def extract_hidden_batch(model, frame_tokens, actions, device, use_amp=False):
    """Run context prefill only (no MaskGIT decoding), return h_t.

    Args:
        frame_tokens: (B, K, TPF+1) long, with status tokens.
        actions: (B, K) long.
    Returns:
        h_t: (B, embed_dim) float.
    """
    m = _unwrap(model)
    K = m.context_frames

    parts = []
    for i in range(K):
        parts.append(m.token_embed(frame_tokens[:, i]))
        parts.append(m.action_embed(actions[:, i]).unsqueeze(1))
    x = torch.cat(parts, dim=1)

    ctx_len = K * (m.block_size + 1)
    ctx_mask = m.attn_mask[:ctx_len, :ctx_len]
    rope_cos = m.rope_cos[:ctx_len]
    rope_sin = m.rope_sin[:ctx_len]

    with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
        for block in m.blocks:
            x, _ = block(x, ctx_mask, rope_cos, rope_sin)
        x = m.ln_f(x)

    return x[:, -1].float()  # h_t at last context position


def extract_bc_data(model, episodes, context_frames, trim_tail, device,
                    batch_size=256):
    """Pre-extract (h_t, token_ids, action) tuples from real episodes for BC.

    Stores raw token IDs (not embeddings) to save memory. Embeddings are
    looked up on-the-fly during training.

    Returns:
        h_data: (N, embed_dim) float CPU tensor
        token_data: (N, tokens_per_frame) long CPU tensor
        labels: (N,) long CPU tensor (0=idle, 1=jump)
    """
    m = _unwrap(model)
    use_amp = device.type == "cuda"

    # Collect all windows
    all_ctx = []
    all_act = []
    all_next_tokens = []
    all_labels = []

    for tokens, actions in episodes:
        T = len(tokens)
        end = T - trim_tail
        if end <= context_frames:
            continue
        for t in range(end - context_frames):
            all_ctx.append(tokens[t:t + context_frames])
            all_act.append(actions[t:t + context_frames])
            all_next_tokens.append(tokens[t + context_frames])
            all_labels.append(actions[t + context_frames])

    n = len(all_labels)
    if n == 0:
        return torch.empty(0, m.embed_dim), \
               torch.empty(0, m.tokens_per_frame, dtype=torch.long), \
               torch.empty(0, dtype=torch.long)

    ctx_np = np.array(all_ctx)
    act_np = np.array(all_act)
    next_tokens_np = np.array(all_next_tokens)
    labels_np = np.array(all_labels)

    # Process in batches to get h_t
    all_h = []

    for i in range(0, n, batch_size):
        b_ctx = ctx_np[i:i + batch_size]
        b_act = act_np[i:i + batch_size]
        B = len(b_ctx)

        # Add ALIVE status
        status = np.full((*b_ctx.shape[:2], 1), m.ALIVE_TOKEN, dtype=np.int64)
        ctx_with_status = np.concatenate([b_ctx, status], axis=2)

        ctx_t = torch.from_numpy(ctx_with_status).to(device)
        act_t = torch.from_numpy(b_act).to(device)

        with torch.no_grad():
            h_t = extract_hidden_batch(model, ctx_t, act_t, device, use_amp)

        all_h.append(h_t.cpu())

        if (i // batch_size + 1) % 50 == 0:
            print(f"  BC extraction: {i + B}/{n}...")

    h_data = torch.cat(all_h)
    token_data = torch.from_numpy(next_tokens_np.astype(np.int64))
    labels = torch.from_numpy(labels_np).long()

    return h_data, token_data, labels


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

        # Get individual token embeddings (not mean-pooled)
        with torch.no_grad():
            tok_embeds = m.token_embed(pred_tokens).float()  # (B, 64, 256)

        # Controller action (WITH gradients)
        action, log_prob, entropy = controller.act(tok_embeds, h_t.float())

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
    """Compute Reinforce loss with discounted returns and baseline."""
    T = len(rewards)
    if T == 0:
        return torch.tensor(0.0, requires_grad=True), 0.0, 0.0

    rewards_t = torch.stack(rewards)      # (T, B)
    log_probs_t = torch.stack(log_probs)  # (T, B)
    entropies_t = torch.stack(entropies)  # (T, B)

    B = rewards_t.shape[1]
    returns = torch.zeros_like(rewards_t)
    G = torch.zeros(B, device=rewards_t.device)
    for t in reversed(range(T)):
        G = rewards_t[t] + gamma * G
        returns[t] = G

    baseline = returns.mean(dim=1, keepdim=True)
    advantages = returns - baseline

    policy_loss = -(log_probs_t * advantages.detach()).sum(dim=0).mean()
    entropy_loss = -entropy_coeff * entropies_t.sum(dim=0).mean()

    return policy_loss + entropy_loss, returns.mean().item(), \
        entropies_t.mean().item()


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

        with torch.no_grad():
            tok_embeds = m.token_embed(pred_tokens).float()
        action = controller.act_deterministic(tok_embeds, h_t.float())

        new_status = torch.full((B, 1), m.ALIVE_TOKEN, dtype=torch.long, device=device)
        new_frame = torch.cat([pred_tokens, new_status], dim=1).unsqueeze(1)
        ctx_t = torch.cat([ctx_t[:, 1:], new_frame], dim=1)
        ctx_a = torch.cat([ctx_a[:, 1:], action.unsqueeze(1)], dim=1)

    return survival.mean().item()


def main():
    parser = argparse.ArgumentParser(
        description="Train controller via BC + Reinforce in dream rollouts")
    parser.add_argument("--transformer-checkpoint",
                        default="checkpoints/transformer_best.pt")
    parser.add_argument("--episodes-dir", default="data/episodes")
    # Reinforce
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--entropy-coeff", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--n-iterations", type=int, default=500)
    # Behavior cloning
    parser.add_argument("--bc-weight-start", type=float, default=0.8)
    parser.add_argument("--bc-weight-end", type=float, default=0.1)
    parser.add_argument("--bc-batch-size", type=int, default=256)
    parser.add_argument("--bc-trim-tail", type=int, default=15)
    # Rollout
    parser.add_argument("--n-episodes", type=int, default=64)
    parser.add_argument("--max-dream-steps", type=int, default=20)
    parser.add_argument("--death-threshold", type=float, default=0.5)
    parser.add_argument("--context-frames", type=int, default=4)
    # Policy architecture (Transformer encoder)
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

    # Pre-extract BC data (stores token IDs, not embeddings)
    print(f"\nExtracting BC data (trim_tail={args.bc_trim_tail})...")
    bc_h, bc_tokens, bc_labels = extract_bc_data(
        model, episodes, args.context_frames, args.bc_trim_tail, device)
    n_bc = len(bc_labels)
    n_jump = (bc_labels == 1).sum().item()
    n_idle = n_bc - n_jump
    print(f"BC data: {n_bc} samples ({n_jump} jump, {n_idle} idle, "
          f"jump ratio={n_jump / max(n_bc, 1):.1%})")

    if n_jump > 0 and n_idle > 0:
        pos_weight = torch.tensor(n_idle / n_jump, device=device)
    else:
        pos_weight = torch.tensor(1.0, device=device)
    print(f"BC pos_weight: {pos_weight.item():.2f}")

    # Create DART-style Transformer policy
    m = _unwrap(model)
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
          f"({n_params:,} params)")

    optimizer = torch.optim.Adam(controller.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.n_iterations, eta_min=args.lr * 0.01)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # CSV log
    log_path = ckpt_dir / "controller_reinforce_log.csv"
    log_file = open(log_path, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["iteration", "mean_survival", "mean_return",
                     "rl_loss", "bc_loss", "total_loss", "alpha",
                     "entropy", "lr", "eval_survival", "time_s"])

    best_eval = -float("inf")

    print(f"\nBC+Reinforce: lr={args.lr}, gamma={args.gamma}, "
          f"entropy_coeff={args.entropy_coeff}")
    print(f"BC: alpha={args.bc_weight_start}->{args.bc_weight_end} (cosine), "
          f"batch_size={args.bc_batch_size}")
    print(f"Dream: n_episodes={args.n_episodes}, "
          f"max_steps={args.max_dream_steps}")
    print()

    for iteration in range(1, args.n_iterations + 1):
        t0 = time.time()

        # Cosine alpha schedule
        progress = (iteration - 1) / max(args.n_iterations - 1, 1)
        alpha = args.bc_weight_end + (args.bc_weight_start - args.bc_weight_end) \
            * 0.5 * (1 + math.cos(math.pi * progress))

        # --- BC loss ---
        controller.train()
        idx = rng.integers(0, n_bc, size=args.bc_batch_size)
        bc_h_batch = bc_h[idx].to(device)
        bc_tok_batch = bc_tokens[idx].to(device)
        bc_a_batch = bc_labels[idx].to(device)

        # Look up token embeddings from frozen WM
        with torch.no_grad():
            bc_tok_embeds = m.token_embed(bc_tok_batch).float()

        bc_logits = controller._logits(bc_tok_embeds, bc_h_batch)
        bc_loss = F.binary_cross_entropy_with_logits(
            bc_logits, bc_a_batch.float(), pos_weight=pos_weight)

        # --- RL loss ---
        ctx_tokens, ctx_actions = sample_contexts(
            episodes, args.n_episodes, args.context_frames, rng)

        log_probs, rewards, entropies, survival = dream_rollout(
            model, controller, ctx_tokens, ctx_actions,
            max_steps=args.max_dream_steps,
            death_threshold=args.death_threshold,
            device=device,
            warmup_steps=args.context_frames)

        if len(log_probs) > 0:
            rl_loss, mean_return, mean_entropy = compute_reinforce_loss(
                log_probs, rewards, entropies,
                gamma=args.gamma, entropy_coeff=args.entropy_coeff)
        else:
            rl_loss = torch.tensor(0.0, requires_grad=True, device=device)
            mean_return, mean_entropy = 0.0, 0.0

        # --- Combined loss ---
        total_loss = alpha * bc_loss + (1 - alpha) * rl_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(controller.parameters(),
                                        args.max_grad_norm)
        optimizer.step()
        scheduler.step()

        elapsed = time.time() - t0
        mean_surv = survival.mean().item()
        lr = optimizer.param_groups[0]["lr"]

        # Periodic evaluation
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
            f"{rl_loss.item():.4f}", f"{bc_loss.item():.4f}",
            f"{total_loss.item():.4f}", f"{alpha:.3f}",
            f"{mean_entropy:.4f}", f"{lr:.1e}", eval_surv,
            f"{elapsed:.1f}"])
        log_file.flush()

        eval_str = f" | eval={eval_surv}" if eval_surv else ""
        print(f"Iter {iteration:3d} | surv={mean_surv:5.1f} | "
              f"bc={bc_loss.item():.3f} | rl={rl_loss.item():.3f} | "
              f"a={alpha:.2f} | ent={mean_entropy:.3f} | "
              f"lr={lr:.1e}{eval_str} | {elapsed:.1f}s")

    log_file.close()
    print(f"\nDone. Best eval survival: {best_eval:.1f}")
    torch.save(controller.state_dict(),
               ckpt_dir / "controller_reinforce_final.pt")


if __name__ == "__main__":
    main()
