"""Evaluate the Transformer world model: single-step and multi-step rollouts.

Visualizes predicted vs actual frame sequences by decoding tokens through the tokenizer.

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
from deepdash.world_model import WorldModel


def load_tokenizer(args, device):
    """Load the appropriate tokenizer (FSQ or VQ-VAE)."""
    if args.tokenizer == "fsq":
        from deepdash.fsq import FSQVAE
        model = FSQVAE(levels=args.levels).to(device)
        grid_size = 8
    else:
        from deepdash.vqvae import VQVAE
        model = VQVAE(num_embeddings=args.num_embeddings,
                      embedding_dim=args.embedding_dim).to(device)
        grid_size = 6
    model.load_state_dict(torch.load(args.tokenizer_checkpoint, map_location=device,
                                     weights_only=True))
    model.eval()
    return model, grid_size


def decode_tokens_to_image(tokenizer, tokens, grid_size, device):
    """Decode flat token indices to (64, 64) uint8 image."""
    indices = torch.from_numpy(tokens.astype(np.int64)).reshape(1, grid_size, grid_size).to(device)
    img = tokenizer.decode_indices(indices)  # (1, 1, 64, 64)
    return (img[0, 0].cpu().numpy() * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Transformer world model")
    parser.add_argument("--transformer-checkpoint", default="checkpoints/transformer_best.pt")
    parser.add_argument("--tokenizer", choices=["fsq", "vqvae"], default="fsq")
    parser.add_argument("--tokenizer-checkpoint", default=None,
                        help="Default: checkpoints/{tokenizer}_best.pt")
    parser.add_argument("--episodes-dir", default="data/episodes")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=20)
    parser.add_argument("--context-frames", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--tokens-per-frame", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    # Sampling
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature for rollouts. 0 = greedy (default)")
    parser.add_argument("--top-k", type=int, default=0,
                        help="Top-k sampling. 0 = disabled")
    parser.add_argument("--top-p", type=float, default=0.0,
                        help="Nucleus sampling threshold. 0 = disabled")
    # Tokenizer-specific
    parser.add_argument("--levels", type=int, nargs="+", default=[8, 5, 5, 5])
    parser.add_argument("--num-embeddings", type=int, default=1024)
    parser.add_argument("--embedding-dim", type=int, default=8)
    args = parser.parse_args()

    if args.tokenizer_checkpoint is None:
        args.tokenizer_checkpoint = f"checkpoints/{args.tokenizer}_best.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    tokenizer, grid_size = load_tokenizer(args, device)
    TPF = args.tokens_per_frame

    # Load Transformer
    model = WorldModel(
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        context_frames=args.context_frames,
        dropout=args.dropout,
        tokens_per_frame=args.tokens_per_frame,
    ).to(device)
    model.load_state_dict(torch.load(args.transformer_checkpoint, map_location=device,
                                     weights_only=True))
    model.eval()

    # Find episodes
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
    selected = rng.choice(candidates, size=min(args.num_samples, len(candidates)),
                          replace=False)

    print(f"Evaluating {len(selected)} episodes, {args.rollout_steps} rollout steps each")

    # --- Single-step accuracy ---
    total_correct, total_tokens = 0, 0
    batch_size = 256
    with torch.no_grad():
        for ep_idx, ep in enumerate(candidates):
            print(f"  Single-step: episode {ep_idx + 1}/{len(candidates)} ({ep.name})",
                  end="\r")
            tokens = np.load(ep / "tokens.npy").astype(np.int64)
            actions = np.load(ep / "actions.npy").astype(np.int64)
            T = len(tokens)
            if T < K + 1:
                continue

            meta_path = ep / "metadata.json"
            level_id = 0
            if meta_path.exists():
                level_id = json.loads(meta_path.read_text()).get("level", 1) - 1

            is_clear = "clear" in ep.name

            # Build windows with status tokens
            all_frames = []
            all_actions = []
            for i in range(T - K):
                window = tokens[i:i + K + 1]  # (K+1, TPF)
                # Append status column
                status = np.full((K + 1, 1), model.ALIVE_TOKEN, dtype=np.int64)
                is_death = (not is_clear) and (i + K == T - 1)
                if is_death:
                    status[K] = model.DEATH_TOKEN
                frame_with_status = np.concatenate([window, status], axis=1)
                all_frames.append(frame_with_status)
                all_actions.append(actions[i:i + K])

            all_frames = np.array(all_frames)
            all_actions = np.array(all_actions)

            for b in range(0, len(all_frames), batch_size):
                f_batch = torch.from_numpy(all_frames[b:b + batch_size]).to(device)
                a_batch = torch.from_numpy(all_actions[b:b + batch_size]).to(device)
                bs = f_batch.size(0)
                l_batch = torch.full((bs,), level_id, dtype=torch.long, device=device)
                target = f_batch[:, -1, :TPF]  # visual tokens only
                logits, _ = model(f_batch, a_batch, l_batch)
                preds = logits[:, :TPF].argmax(dim=-1)
                total_correct += (preds == target).sum().item()
                total_tokens += target.numel()

    print()
    print(f"Single-step accuracy: {total_correct / total_tokens:.4f} "
          f"({total_correct}/{total_tokens})")

    # --- Multi-step rollout ---
    for ep in selected:
        tokens = np.load(ep / "tokens.npy")
        actions = np.load(ep / "actions.npy")
        meta_path = ep / "metadata.json"
        level = "?"
        level_id = 0
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            level = meta.get("level", "?")
            level_id = meta.get("level", 1) - 1

        # Context: first K frames with ALIVE status
        ctx_tokens = tokens[:K].copy()  # (K, TPF)
        ctx_status = np.full((K, 1), model.ALIVE_TOKEN, dtype=np.int64)
        ctx_with_status = np.concatenate([ctx_tokens, ctx_status], axis=1)  # (K, TPF+1)
        ctx_actions = actions[:K].copy()

        predicted_frames = []
        actual_frames = []

        level_t = torch.tensor([level_id], dtype=torch.long, device=device)

        with torch.no_grad():
            for step in range(args.rollout_steps):
                t = K + step
                if t >= len(tokens):
                    break
                print(f"  Rollout {ep.name}: step {step + 1}/{args.rollout_steps}",
                      end="\r")

                ctx_t = torch.from_numpy(ctx_with_status.astype(np.int64)).unsqueeze(0).to(device)
                ctx_a = torch.from_numpy(ctx_actions.astype(np.int64)).unsqueeze(0).to(device)
                pred_visual, death_prob = model.predict_next_frame(
                    ctx_t, ctx_a, level_t,
                    temperature=args.temperature, top_k=args.top_k,
                    top_p=args.top_p)
                pred_np = pred_visual[0].cpu().numpy()
                dp = death_prob[0].item()

                actual_tokens_np = tokens[t].astype(np.int64)
                match = (pred_np == actual_tokens_np).sum()
                if step < 3:
                    print(f"  Step {step+1}: {match}/{TPF} tokens correct, death={dp:.3f}")

                pred_img = decode_tokens_to_image(tokenizer, pred_np, grid_size, device)
                actual_img = decode_tokens_to_image(tokenizer, tokens[t], grid_size, device)
                predicted_frames.append(pred_img)
                actual_frames.append(actual_img)

                # Shift context: drop oldest, add predicted with ALIVE status
                new_frame = np.concatenate([
                    pred_np.reshape(1, TPF),
                    np.array([[model.ALIVE_TOKEN]], dtype=np.int64)
                ], axis=1)
                ctx_with_status = np.concatenate([
                    ctx_with_status[1:], new_frame
                ], axis=0)
                ctx_actions = np.concatenate([
                    ctx_actions[1:], actions[t:t + 1]
                ], axis=0)

        print()

        n_steps = len(predicted_frames)
        fig, axes = plt.subplots(3, n_steps, figsize=(n_steps * 1.5, 4.5))
        fig.suptitle(f"{ep.name} (level {level}) — {n_steps}-step rollout", fontsize=12)

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
