"""Encode episode frames through frozen VQ-VAE to produce token sequences.

For each episode, saves tokens.npy (T, 36) uint16 — flattened 6x6 codebook indices.

Usage:
    python scripts/tokenize_episodes.py
    python scripts/tokenize_episodes.py --checkpoint checkpoints/vqvae_best.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepdash.vqvae import VQVAE


def main():
    parser = argparse.ArgumentParser(description="Tokenize episodes with frozen VQ-VAE")
    parser.add_argument("--episodes-dir", default="data/episodes")
    parser.add_argument("--checkpoint", default="checkpoints/vqvae_best.pt")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-embeddings", type=int, default=1024)
    parser.add_argument("--embedding-dim", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = VQVAE(
        num_embeddings=args.num_embeddings,
        embedding_dim=args.embedding_dim,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded VQ-VAE from {args.checkpoint}")

    episodes_dir = Path(args.episodes_dir)
    episodes = sorted(ep for ep in episodes_dir.glob("*") if (ep / "frames.npy").exists())
    print(f"Found {len(episodes)} episodes")

    total_frames = 0
    with torch.no_grad():
        for ep in episodes:
            frames = np.load(ep / "frames.npy")  # (T, 64, 64) uint8
            T = len(frames)
            total_frames += T

            # Encode in batches
            all_tokens = []
            for i in range(0, T, args.batch_size):
                batch = frames[i:i + args.batch_size]
                # (B, 64, 64) uint8 → (B, 1, 64, 64) float [0,1]
                x = torch.from_numpy(batch).float().unsqueeze(1).to(device) / 255.0
                indices = model.encode(x)  # (B, 6, 6)
                all_tokens.append(indices.cpu().reshape(-1, 36).numpy())

            tokens = np.concatenate(all_tokens, axis=0).astype(np.uint16)  # (T, 36)
            np.save(ep / "tokens.npy", tokens)
            print(f"  {ep.name}: {T} frames -> {tokens.shape} tokens")

    print(f"\nDone. Tokenized {total_frames} frames across {len(episodes)} episodes.")


if __name__ == "__main__":
    main()
