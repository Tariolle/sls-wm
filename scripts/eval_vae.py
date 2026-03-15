"""Evaluate VQ-VAE or FSQ-VAE: generate side-by-side original vs reconstruction images."""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_image(path):
    """Load grayscale PNG as [0,1] float tensor (1, H, W)."""
    img = np.array(Image.open(path).convert("L")).astype(np.float32) / 255.0
    return torch.from_numpy(img).unsqueeze(0)


def tensor_to_image(t):
    """Convert (1, H, W) grayscale tensor to PIL Image."""
    arr = (t.clamp(0, 1).squeeze(0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def main():
    parser = argparse.ArgumentParser(description="Evaluate tokenizer reconstructions")
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint path (default: checkpoints/{model}_best.pt)")
    parser.add_argument("--model", choices=["vqvae", "fsq"], default="fsq")
    parser.add_argument("--data-dir", default=None,
                        help="Path to .npy file with frames (default: sample from episodes)")
    parser.add_argument("--episodes-dir", default="data/death_episodes",
                        help="Episode directory (used when --data-dir is not set)")
    parser.add_argument("--output-dir", default="eval_output")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--num-embeddings", type=int, default=1024)
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--levels", type=int, nargs="+", default=[8, 5, 5, 5])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.checkpoint is None:
        args.checkpoint = f"checkpoints/{args.model}_best.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(checkpoint_path):
        if args.model == "fsq":
            from deepdash.fsq import FSQVAE
            m = FSQVAE(levels=args.levels)
        else:
            from deepdash.vqvae import VQVAE
            m = VQVAE(num_embeddings=args.num_embeddings, embedding_dim=args.embedding_dim)
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        # Strip _orig_mod. prefix from torch.compile checkpoints
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
        m.load_state_dict(state)
        m.to(device)
        m.eval()
        return m

    model_best = load_model(args.checkpoint)

    if args.data_dir is not None:
        data = np.load(args.data_dir)  # (N, 64, 64) uint8
    else:
        # Load frames from episodes
        episodes_dir = Path(args.episodes_dir)
        all_frames = []
        for ep in sorted(episodes_dir.glob("*")):
            fp = ep / "frames.npy"
            if fp.exists():
                all_frames.append(np.load(fp))
        data = np.concatenate(all_frames, axis=0)
    if len(data) == 0:
        print("No frames found")
        return

    import random
    random.seed(args.seed)
    indices = random.sample(range(len(data)), min(args.num_samples, len(data)))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    originals, recons = [], []
    with torch.no_grad():
        for i, idx in enumerate(indices):
            frame = data[idx].astype(np.float32) / 255.0
            img = torch.from_numpy(frame).unsqueeze(0).unsqueeze(0).to(device)

            orig_pil = tensor_to_image(img[0])
            recon_pil = tensor_to_image(model_best(img)[0][0])
            originals.append(orig_pil)
            recons.append(recon_pil)

            combined = Image.new("L", (orig_pil.width * 2, orig_pil.height))
            combined.paste(orig_pil, (0, 0))
            combined.paste(recon_pil, (orig_pil.width, 0))
            combined.save(out_dir / f"sample_{i:02d}.png")

    print(f"Saved {len(indices)} comparisons to {out_dir}/")

    import matplotlib.pyplot as plt
    n = len(originals)
    fig, axes = plt.subplots(2, n, figsize=(n * 2, 4))
    for i in range(n):
        axes[0, i].imshow(originals[i], cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(recons[i], cmap="gray")
        axes[1, i].axis("off")
    axes[0, 0].set_title("Original", fontsize=30, pad=10)
    axes[1, 0].set_title("Reconstruction", fontsize=30, pad=10)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
