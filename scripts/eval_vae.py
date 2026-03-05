"""Evaluate VAE: generate side-by-side original vs reconstruction images."""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepdash.vae import VAE


def load_image(path):
    """Load grayscale PNG as [0,1] float tensor (1, H, W)."""
    img = np.array(Image.open(path).convert("L")).astype(np.float32) / 255.0
    return torch.from_numpy(img).unsqueeze(0)


def tensor_to_image(t):
    """Convert (1, H, W) grayscale tensor to PIL Image."""
    arr = (t.clamp(0, 1).squeeze(0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def main():
    parser = argparse.ArgumentParser(description="Evaluate VAE reconstructions")
    parser.add_argument("--checkpoint", default="checkpoints/vae_best.pt")
    parser.add_argument("--data-dir", default="data/val")
    parser.add_argument("--output-dir", default="eval_output")
    parser.add_argument("--num-samples", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # Find PNG files (may be in a subdirectory from ImageFolder restructuring)
    data_path = Path(args.data_dir)
    pngs = sorted(data_path.rglob("*.png"))
    if not pngs:
        print(f"No PNGs found in {data_path}")
        return

    # Sample evenly across the dataset
    step = max(1, len(pngs) // args.num_samples)
    selected = pngs[::step][:args.num_samples]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    with torch.no_grad():
        for i, path in enumerate(selected):
            img = load_image(path).unsqueeze(0).to(device)
            recon, _, _ = model(img)

            orig_pil = tensor_to_image(img[0])
            recon_pil = tensor_to_image(recon[0])

            # Side-by-side
            combined = Image.new("L", (orig_pil.width * 2, orig_pil.height))
            combined.paste(orig_pil, (0, 0))
            combined.paste(recon_pil, (orig_pil.width, 0))
            combined.save(out_dir / f"sample_{i:02d}.png")

    print(f"Saved {len(selected)} comparisons to {out_dir}/")


if __name__ == "__main__":
    main()
