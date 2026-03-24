"""Visualize PPO controller dream rollouts as video grids.

Runs the trained controller in the world model's imagination,
decodes predicted tokens through the FSQ-VAE, and saves videos.

Usage:
    python scripts/vis_ppo_dream.py
    python scripts/vis_ppo_dream.py --n-episodes 8 --max-steps 60
    python scripts/vis_ppo_dream.py --controller checkpoints/controller_ppo_final.pt
"""

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepdash.fsq import FSQVAE
from deepdash.world_model import WorldModel
from deepdash.controller import CNNPolicy


def tokenize_episode(vae, ep_dir, device, batch_size=64):
    """Tokenize a single episode's frames.npy through the FSQ-VAE encoder."""
    frames = np.load(ep_dir / "frames.npy")  # (T, 64, 64) uint8
    all_tokens = []
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i + batch_size]
        x = torch.from_numpy(batch).float().unsqueeze(1).to(device) / 255.0
        with torch.no_grad():
            indices = vae.encode(x)  # (B, 8, 8)
        all_tokens.append(indices.cpu().reshape(-1, 64).numpy())
    tokens = np.concatenate(all_tokens, axis=0).astype(np.uint16)
    np.save(ep_dir / "tokens.npy", tokens)
    return tokens


def compute_val_set(death_dir, expert_dir="data/expert_episodes"):
    """Get global val set (shared across all models)."""
    from deepdash.data_split import get_val_episodes
    return get_val_episodes(death_dir, expert_dir)


def load_episodes(episodes_dir, context_frames, vae=None, device=None,
                  split_filter="all"):
    """Load base (non-shifted) tokenized episodes with enough frames.

    If vae is provided, episodes with frames.npy but no tokens.npy will be
    tokenized on the fly.
    """
    shift_re = re.compile(r"_s[+-]\d+_[+-]\d+$")
    val_set = compute_val_set(episodes_dir) if split_filter != "all" else set()
    episodes = []
    tokenized = 0
    for ep in sorted(Path(episodes_dir).glob("*")):
        if shift_re.search(ep.name):
            continue
        ap = ep / "actions.npy"
        if not ap.exists():
            continue
        if split_filter == "val" and ep.name not in val_set:
            continue
        if split_filter == "train" and ep.name in val_set:
            continue
        tp = ep / "tokens.npy"
        if not tp.exists():
            if vae is not None and (ep / "frames.npy").exists():
                print(f"  Tokenizing {ep.name}...", end="\r")
                tokenize_episode(vae, ep, device)
                tokenized += 1
            else:
                continue
        tokens = np.load(tp).astype(np.int64)
        actions = np.load(ap).astype(np.int64)
        if len(tokens) >= context_frames * 3:
            episodes.append((tokens, actions))
    if tokenized:
        print(f"  Tokenized {tokenized} episodes on the fly")
    return episodes


def decode_tokens(vae, tokens_np, device):
    """Decode 64 token IDs to (64, 64) uint8 grayscale image."""
    indices = torch.from_numpy(tokens_np.astype(np.int64)).reshape(1, 8, 8).to(device)
    with torch.no_grad():
        img = vae.decode_indices(indices)
    return (img[0, 0].cpu().numpy() * 255).astype(np.uint8)


def dream_rollout_visual(model, controller, vae, ctx_tokens_np, ctx_actions_np,
                         max_steps, death_threshold, device):
    """Run a single dream rollout with the controller, returning decoded frames.

    Returns:
        frames: list of (64, 64) uint8 images
        actions: list of int (0=idle, 1=jump)
        death_probs: list of float
        steps_survived: int
    """
    m = model._orig_mod if hasattr(model, "_orig_mod") else model
    K = ctx_tokens_np.shape[0]

    status = np.full((K, 1), m.ALIVE_TOKEN, dtype=np.int64)
    ctx_with_status = np.concatenate([ctx_tokens_np, status], axis=1)
    ctx_t = torch.from_numpy(ctx_with_status[None]).to(device)
    ctx_a = torch.from_numpy(ctx_actions_np[None]).to(device)

    frames = []
    actions = []
    death_probs = []

    # Decode last context frame as first visible frame
    frames.append(decode_tokens(vae, ctx_tokens_np[-1], device))
    actions.append(int(ctx_actions_np[-1]))
    death_probs.append(0.0)

    use_amp = device.type == "cuda"

    for step in range(max_steps):
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                pred_tokens, death_prob, h_t = model.predict_next_frame(
                    ctx_t, ctx_a, temperature=0.0, return_hidden=True)

        dp = death_prob[0].item()
        death_probs.append(dp)

        # Decode predicted frame
        pred_np = pred_tokens[0].cpu().numpy()
        frame = decode_tokens(vae, pred_np, device)
        frames.append(frame)

        if dp > death_threshold:
            actions.append(-1)  # dead, no action
            break

        # Controller picks action
        with torch.no_grad():
            action = controller.act_deterministic(pred_tokens, h_t.float())
        act = action[0].item()
        actions.append(act)

        # Shift context
        new_status = torch.full((1, 1), m.ALIVE_TOKEN, dtype=torch.long,
                                device=device)
        new_frame = torch.cat([pred_tokens, new_status], dim=1).unsqueeze(1)
        ctx_t = torch.cat([ctx_t[:, 1:], new_frame], dim=1)
        ctx_a = torch.cat([ctx_a[:, 1:], action.unsqueeze(1)], dim=1)

    steps_survived = len(frames) - 1  # exclude context frame
    return frames, actions, death_probs, steps_survived


def render_frame(frame_img, step, action, death_prob, survived, is_dead):
    """Render a single frame with HUD overlay, returns (H, W, 3) uint8."""
    scale = 4
    h, w = frame_img.shape
    H, W = h * scale, w * scale

    # Upscale with nearest neighbor
    big = cv2.resize(frame_img, (W, H), interpolation=cv2.INTER_NEAREST)
    canvas = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)

    if is_dead:
        # Red tint on death
        overlay = canvas.copy()
        overlay[:, :, 2] = np.minimum(overlay[:, :, 2].astype(int) + 80, 255).astype(np.uint8)
        canvas = cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0)

    # HUD text
    act_str = "JUMP" if action == 1 else ("DEAD" if action == -1 else "idle")
    dp_color = (0, 0, 255) if death_prob > 0.3 else (0, 200, 0)
    cv2.putText(canvas, f"t={step}", (4, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(canvas, f"d={death_prob:.2f}", (4, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, dp_color, 1)
    cv2.putText(canvas, act_str, (4, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    return canvas


def save_rollout_video(rollout, output_path, fps=30):
    """Save a single rollout as a video file."""
    r = rollout
    n_frames = len(r['frames'])

    sample = render_frame(r['frames'][0], 0, 0, 0.0, 0, False)
    h, w = sample.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    for t in range(n_frames):
        is_dead = t == n_frames - 1 and r['actions'][t] == -1
        frame = render_frame(
            r['frames'][t], t, r['actions'][t],
            r['death_probs'][t], r['steps_survived'], is_dead)
        writer.write(frame)

    writer.release()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize PPO controller dream rollouts")
    parser.add_argument("--controller",
                        default="checkpoints/controller_ppo_best.pt")
    parser.add_argument("--baseline",
                        default=None,
                        help="BC controller checkpoint for comparison "
                             "(e.g. checkpoints/controller_bc_best.pt)")
    parser.add_argument("--transformer-checkpoint",
                        default="checkpoints/transformer_best.pt")
    parser.add_argument("--vae-checkpoint", default="checkpoints/fsq_best.pt")
    parser.add_argument("--episodes-dir", default="data/death_episodes")
    parser.add_argument("--expert-episodes-dir", default="data/expert_episodes")
    parser.add_argument("--output-dir", default="outputs/ppo_dreams")
    parser.add_argument("--n-episodes", type=int, default=40,
                        help="Number of dream rollouts to visualize")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--death-threshold", type=float, default=0.5)
    parser.add_argument("--context-frames", type=int, default=4)
    parser.add_argument("--fps", type=float, default=15)
    parser.add_argument("--filter", choices=["all", "train", "val"],
                        default="val",
                        help="Filter episodes by train/val split")
    parser.add_argument("--seed", type=int, default=42)
    # Model architecture
    parser.add_argument("--levels", type=int, nargs="+", default=[8, 5, 5, 5])
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--embed-dim", type=int, default=384)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--tokens-per-frame", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--token-embed-dim", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load FSQ-VAE
    vae = FSQVAE(levels=args.levels).to(device)
    state = torch.load(args.vae_checkpoint, map_location=device, weights_only=True)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    vae.load_state_dict(state)
    vae.eval()
    print("FSQ-VAE loaded")

    # Load world model
    model = WorldModel(
        vocab_size=args.vocab_size, embed_dim=args.embed_dim,
        n_heads=args.n_heads, n_layers=args.n_layers,
        context_frames=args.context_frames, dropout=args.dropout,
        tokens_per_frame=args.tokens_per_frame,
    ).to(device)
    state = torch.load(args.transformer_checkpoint, map_location=device,
                       weights_only=True)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    print("World model loaded")

    # Load controllers
    def load_controller(path):
        ctrl = CNNPolicy(
            vocab_size=args.vocab_size,
            grid_size=int(args.tokens_per_frame ** 0.5),
            token_embed_dim=args.token_embed_dim,
            h_dim=args.embed_dim,
        ).to(device)
        state = torch.load(path, map_location=device, weights_only=True)
        ctrl.load_state_dict(state)
        ctrl.eval()
        return ctrl

    controllers = {"ppo": load_controller(args.controller)}
    print(f"PPO controller loaded: {args.controller}")
    if args.baseline:
        controllers["bc"] = load_controller(args.baseline)
        print(f"BC baseline loaded: {args.baseline}")

    # Load episodes (tokenize on the fly if needed)
    episodes = load_episodes(args.episodes_dir, args.context_frames, vae, device,
                             args.filter)
    n_death = len(episodes)
    expert_eps = load_episodes(args.expert_episodes_dir, args.context_frames, vae, device,
                               args.filter)
    episodes.extend(expert_eps)
    print(f"Loaded {len(episodes)} episodes ({n_death} death, {len(expert_eps)} expert)")

    # Pre-sample contexts (shared across controllers)
    rng = np.random.default_rng(args.seed)
    K = args.context_frames
    sampled_contexts = []
    for i in range(args.n_episodes):
        ep_idx = rng.integers(len(episodes))
        tokens, actions = episodes[ep_idx]
        T = len(tokens)
        latest = T - K * 3
        start = rng.integers(0, latest + 1) if latest > 0 else 0
        sampled_contexts.append((
            tokens[start:start + K],
            actions[start:start + K],
            ep_idx, start,
        ))

    # Run rollouts for each controller
    for ctrl_name, ctrl in controllers.items():
        print(f"\n--- {ctrl_name.upper()} rollouts ---")
        all_rollouts = []

        for i, (ctx_tokens, ctx_actions, ep_idx, start) in enumerate(sampled_contexts):
            print(f"  {i + 1}/{args.n_episodes}: ep={ep_idx}, start={start}",
                  end="")

            frames, acts, dps, survived = dream_rollout_visual(
                model, ctrl, vae, ctx_tokens, ctx_actions,
                args.max_steps, args.death_threshold, device)

            print(f" -> survived {survived} steps, "
                  f"jumps={sum(1 for a in acts if a == 1)}")

            all_rollouts.append({
                'frames': frames,
                'actions': acts,
                'death_probs': dps,
                'steps_survived': survived,
            })

        # Write individual videos
        out_dir = Path(args.output_dir) / ctrl_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, r in enumerate(all_rollouts):
            out_path = out_dir / f"rollout_{i:03d}.mp4"
            save_rollout_video(r, out_path, fps=args.fps)
        print(f"Saved {len(all_rollouts)} videos to {out_dir}/")

        # Summary
        survivals = [r['steps_survived'] for r in all_rollouts]
        jump_rates = []
        for r in all_rollouts:
            alive_acts = [a for a in r['actions'] if a >= 0]
            if alive_acts:
                jump_rates.append(sum(1 for a in alive_acts if a == 1) / len(alive_acts))
        print(f"{ctrl_name.upper()} survival: mean={np.mean(survivals):.1f}, "
              f"min={min(survivals)}, max={max(survivals)}")
        print(f"{ctrl_name.upper()} jump rate: {np.mean(jump_rates):.2f}")


if __name__ == "__main__":
    main()
