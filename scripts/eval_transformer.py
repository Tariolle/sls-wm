"""Evaluate the Transformer world model: single-step and multi-step rollouts.

Visualizes predicted vs actual frame sequences by decoding tokens through VQ-VAE.

Usage:
    python scripts/eval_transformer.py
    python scripts/eval_transformer.py --rollout-steps 20 --num-samples 4
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepdash.vqvae import VQVAE
from deepdash.world_model import WorldModel, TOKENS_PER_FRAME


def decode_tokens_to_image(vqvae, tokens, device):
    """Decode (36,) token indices to (64, 64) uint8 image."""
    indices = torch.from_numpy(tokens.astype(np.int64)).reshape(1, 6, 6).to(device)
    img = vqvae.decode_indices(indices)  # (1, 1, 64, 64)
    return (img[0, 0].cpu().numpy() * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Transformer world model")
    parser.add_argument("--transformer-checkpoint", default="checkpoints/transformer_best.pt")
    parser.add_argument("--vqvae-checkpoint", default="checkpoints/vqvae_best.pt")
    parser.add_argument("--episodes-dir", default="data/episodes")
    parser.add_argument("--num-samples", type=int, default=4,
                        help="Number of episodes to visualize")
    parser.add_argument("--rollout-steps", type=int, default=20,
                        help="Number of autoregressive rollout steps")
    parser.add_argument("--context-frames", type=int, default=8)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load VQ-VAE
    vqvae = VQVAE().to(device)
    vqvae.load_state_dict(torch.load(args.vqvae_checkpoint, map_location=device, weights_only=True))
    vqvae.eval()

    # Load Transformer
    model = WorldModel(
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        context_frames=args.context_frames,
        dropout=args.dropout,
    ).to(device)
    model.load_state_dict(torch.load(args.transformer_checkpoint, map_location=device, weights_only=True))
    model.eval()

    # Find episodes with enough frames
    episodes_dir = Path(args.episodes_dir)
    K = args.context_frames
    min_frames = K + args.rollout_steps
    candidates = []
    for ep in sorted(episodes_dir.glob("*")):
        tp = ep / "tokens.npy"
        ap = ep / "actions.npy"
        if not tp.exists() or not ap.exists():
            continue
        tokens = np.load(tp)
        if len(tokens) >= min_frames:
            candidates.append(ep)

    if not candidates:
        print(f"No episodes with >= {min_frames} frames found.")
        return

    rng = np.random.default_rng(args.seed)
    selected = rng.choice(candidates, size=min(args.num_samples, len(candidates)), replace=False)

    print(f"Evaluating {len(selected)} episodes, {args.rollout_steps} rollout steps each")

    # --- Single-step accuracy across all episodes (batched, non-autoregressive) ---
    total_correct, total_tokens = 0, 0
    batch_size = 256
    with torch.no_grad():
        for ep_idx, ep in enumerate(candidates):
            print(f"  Single-step: episode {ep_idx + 1}/{len(candidates)} ({ep.name})", end="\r")
            tokens = np.load(ep / "tokens.npy").astype(np.int64)
            actions = np.load(ep / "actions.npy").astype(np.int64)
            T = len(tokens)
            if T < K + 1:
                continue

            # Build all windows for this episode
            all_frames = []
            all_actions = []
            for i in range(T - K):
                all_frames.append(tokens[i:i + K + 1])  # (K+1, 36)
                all_actions.append(actions[i:i + K])      # (K,)
            all_frames = np.array(all_frames)
            all_actions = np.array(all_actions)

            # Process in batches
            for b in range(0, len(all_frames), batch_size):
                f_batch = torch.from_numpy(all_frames[b:b + batch_size]).to(device)
                a_batch = torch.from_numpy(all_actions[b:b + batch_size]).to(device)
                target = f_batch[:, -1]  # (B, 36)
                logits = model(f_batch, a_batch)  # (B, 36, vocab)
                preds = logits.argmax(dim=-1)
                total_correct += (preds == target).sum().item()
                total_tokens += target.numel()

    print()  # clear \r line
    print(f"Single-step accuracy: {total_correct / total_tokens:.4f} "
          f"({total_correct}/{total_tokens})")

    # --- Multi-step rollout visualization ---
    for ep in selected:
        tokens = np.load(ep / "tokens.npy")
        actions = np.load(ep / "actions.npy")
        meta_path = ep / "metadata.json"
        level = "?"
        if meta_path.exists():
            level = json.loads(meta_path.read_text()).get("level", "?")

        # Start from frame 0, use first K frames as context
        ctx_tokens = tokens[:K].copy()  # (K, 36)
        ctx_actions = actions[:K].copy()  # (K,)

        predicted_frames = []
        actual_frames = []

        with torch.no_grad():
            for step in range(args.rollout_steps):
                t = K + step
                if t >= len(tokens):
                    break
                print(f"  Rollout {ep.name}: step {step + 1}/{args.rollout_steps}", end="\r")

                # Predict next frame
                ctx_t = torch.from_numpy(
                    ctx_tokens.astype(np.int64)).unsqueeze(0).to(device)
                ctx_a = torch.from_numpy(
                    ctx_actions.astype(np.int64)).unsqueeze(0).to(device)
                pred_tokens = model.predict_next_frame(ctx_t, ctx_a)  # (1, 36)
                pred_np = pred_tokens[0].cpu().numpy()

                # Debug: check token accuracy for this step
                actual_tokens = tokens[t].astype(np.int64)
                match = (pred_np == actual_tokens).sum()
                if step < 3:
                    print(f"  Step {step+1}: {match}/36 tokens correct")

                # Decode both predicted and actual
                pred_img = decode_tokens_to_image(vqvae, pred_np, device)
                actual_img = decode_tokens_to_image(vqvae, tokens[t], device)
                predicted_frames.append(pred_img)
                actual_frames.append(actual_img)

                # Shift context: drop oldest, add predicted
                ctx_tokens = np.concatenate([
                    ctx_tokens[1:], pred_np.reshape(1, 36)
                ], axis=0)
                ctx_actions = np.concatenate([
                    ctx_actions[1:], actions[t:t + 1]
                ], axis=0)

        print()  # clear \r line

        # Plot: 3 rows — actual, predicted, diff
        n_steps = len(predicted_frames)
        fig, axes = plt.subplots(3, n_steps, figsize=(n_steps * 1.5, 4.5))
        fig.suptitle(f"{ep.name} (level {level}) — {n_steps}-step rollout",
                     fontsize=12)

        for i in range(n_steps):
            axes[0, i].imshow(actual_frames[i], cmap="gray", vmin=0, vmax=255)
            axes[0, i].axis("off")
            axes[1, i].imshow(predicted_frames[i], cmap="gray", vmin=0, vmax=255)
            axes[1, i].axis("off")
            diff = np.abs(actual_frames[i].astype(float) - predicted_frames[i].astype(float))
            axes[2, i].imshow(diff, cmap="hot", vmin=0, vmax=128)
            axes[2, i].axis("off")
            if i == 0:
                axes[0, i].set_ylabel("Actual", fontsize=10)
                axes[1, i].set_ylabel("Predicted", fontsize=10)
                axes[2, i].set_ylabel("Diff", fontsize=10)
            axes[0, i].set_title(f"t+{i + 1}", fontsize=8)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
