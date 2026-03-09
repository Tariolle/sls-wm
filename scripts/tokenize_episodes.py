"""Encode episode frames through frozen tokenizer to produce token sequences.

Supports both VQ-VAE (6x6, 36 tokens) and FSQ-VAE (8x8, 64 tokens).

Usage:
    python scripts/tokenize_episodes.py --model fsq --checkpoint checkpoints/fsq_best.pt
    python scripts/tokenize_episodes.py --model vqvae --checkpoint checkpoints/vqvae_best.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Tokenize episodes with frozen tokenizer")
    parser.add_argument("--episodes-dir", default="data/episodes")
    parser.add_argument("--model", choices=["vqvae", "fsq"], default="fsq")
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint path (default: checkpoints/{model}_best.pt)")
    parser.add_argument("--batch-size", type=int, default=128)
    # VQ-VAE specific
    parser.add_argument("--num-embeddings", type=int, default=1024)
    parser.add_argument("--embedding-dim", type=int, default=8)
    # FSQ specific
    parser.add_argument("--levels", type=int, nargs="+", default=[7, 5, 5, 5, 5])
    args = parser.parse_args()

    if args.checkpoint is None:
        args.checkpoint = f"checkpoints/{args.model}_best.pt" if args.model == "fsq" \
            else "checkpoints/vqvae_best.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.model == "fsq":
        from deepdash.fsq import FSQVAE
        model = FSQVAE(levels=args.levels).to(device)
        grid_size = 8
        tokens_per_frame = 64
    else:
        from deepdash.vqvae import VQVAE
        model = VQVAE(num_embeddings=args.num_embeddings,
                      embedding_dim=args.embedding_dim).to(device)
        grid_size = 6
        tokens_per_frame = 36

    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded {args.model.upper()} from {args.checkpoint}")

    episodes_dir = Path(args.episodes_dir)
    episodes = sorted(ep for ep in episodes_dir.glob("*") if (ep / "frames.npy").exists())
    print(f"Found {len(episodes)} episodes")

    total_frames = 0
    with torch.no_grad():
        for ep in episodes:
            frames = np.load(ep / "frames.npy")  # (T, 64, 64) uint8
            T = len(frames)
            total_frames += T

            all_tokens = []
            for i in range(0, T, args.batch_size):
                batch = frames[i:i + args.batch_size]
                x = torch.from_numpy(batch).float().unsqueeze(1).to(device) / 255.0
                indices = model.encode(x)  # (B, grid, grid)
                all_tokens.append(indices.cpu().reshape(-1, tokens_per_frame).numpy())

            tokens = np.concatenate(all_tokens, axis=0).astype(np.uint16)
            np.save(ep / "tokens.npy", tokens)
            print(f"  {ep.name}: {T} frames -> {tokens.shape} tokens")

    print(f"\nDone. Tokenized {total_frames} frames across {len(episodes)} episodes.")


if __name__ == "__main__":
    main()
