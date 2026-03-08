"""Evaluate VAE or VQ-VAE: generate side-by-side original vs reconstruction images."""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepdash.vae import VAE
from deepdash.vqvae import VQVAE


def load_image(path):
    """Load grayscale PNG as [0,1] float tensor (1, H, W)."""
    img = np.array(Image.open(path).convert("L")).astype(np.float32) / 255.0
    return torch.from_numpy(img).unsqueeze(0)


def tensor_to_image(t):
    """Convert (1, H, W) grayscale tensor to PIL Image."""
    arr = (t.clamp(0, 1).squeeze(0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def main():
    parser = argparse.ArgumentParser(description="Evaluate VAE/VQ-VAE reconstructions")
    parser.add_argument("--checkpoint", default="checkpoints/vqvae_best.pt")
    parser.add_argument("--data-dir", default="data/val.npy")
    parser.add_argument("--output-dir", default="eval_output")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--model", choices=["vae", "vqvae"], default="vqvae")
    parser.add_argument("--num-embeddings", type=int, default=1024, help="Codebook size (must match checkpoint)")
    parser.add_argument("--embedding-dim", type=int, default=8, help="Codebook vector dimension (must match checkpoint)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for frame selection")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(checkpoint_path):
        if args.model == "vqvae":
            m = VQVAE(num_embeddings=args.num_embeddings, embedding_dim=args.embedding_dim)
        else:
            m = VAE()
        m.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        m.to(device)
        m.eval()
        return m

    model_best = load_model(args.checkpoint)
    ckpt_dir = Path(args.checkpoint).parent
    final_path = ckpt_dir / "vqvae_final.pt" if args.model == "vqvae" else ckpt_dir / "vae_final.pt"
    model_final = load_model(str(final_path)) if final_path.exists() else None

    data = np.load(args.data_dir)  # (N, 64, 64) uint8
    if len(data) == 0:
        print(f"No frames found in {args.data_dir}")
        return

    import random
    random.seed(args.seed)
    indices = random.sample(range(len(data)), min(args.num_samples, len(data)))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    originals, recons_best, recons_final = [], [], []
    with torch.no_grad():
        for i, idx in enumerate(indices):
            frame = data[idx].astype(np.float32) / 255.0
            img = torch.from_numpy(frame).unsqueeze(0).unsqueeze(0).to(device)

            orig_pil = tensor_to_image(img[0])
            recon_best_pil = tensor_to_image(model_best(img)[0][0])
            originals.append(orig_pil)
            recons_best.append(recon_best_pil)

            if model_final:
                recons_final.append(tensor_to_image(model_final(img)[0][0]))

            combined = Image.new("L", (orig_pil.width * 2, orig_pil.height))
            combined.paste(orig_pil, (0, 0))
            combined.paste(recon_best_pil, (orig_pil.width, 0))
            combined.save(out_dir / f"sample_{i:02d}.png")

    print(f"Saved {len(indices)} comparisons to {out_dir}/")

    import matplotlib.pyplot as plt
    n = len(originals)
    rows = 3 if model_final else 2
    fig, axes = plt.subplots(rows, n, figsize=(n * 2, rows * 2))
    for i in range(n):
        axes[0, i].imshow(originals[i], cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(recons_best[i], cmap="gray")
        axes[1, i].axis("off")
        if model_final:
            axes[2, i].imshow(recons_final[i], cmap="gray")
            axes[2, i].axis("off")
    axes[0, 0].set_title("Original", fontsize=10)
    axes[1, 0].set_title("Best", fontsize=10)
    if model_final:
        axes[2, 0].set_title("Final", fontsize=10)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
