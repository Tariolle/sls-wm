"""Train the VQ-VAE on preprocessed Geometry Dash frames."""

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepdash.vqvae import VQVAE, vqvae_loss


class FlatImageDataset(Dataset):
    def __init__(self, data_dir, augment=False):
        files = sorted(Path(data_dir).rglob("*.png"))
        load_transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
        self.imgs = [load_transform(Image.open(f)) for f in files]
        self.augment = augment

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        if self.augment:
            # Random shift: pad 4px, then random crop back to 64x64
            img = transforms.functional.pad(img, 4, padding_mode='edge')
            i, j, h, w = transforms.RandomCrop.get_params(img, (64, 64))
            img = transforms.functional.crop(img, i, j, h, w)
        return img, 0


def make_loader(data_dir, batch_size, shuffle=True, augment=False):
    dataset = FlatImageDataset(data_dir, augment=augment)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, total_recon, total_vq, n = 0, 0, 0, 0
    for imgs, _ in loader:
        imgs = imgs.to(device)
        recon, vq_loss_val, _ = model(imgs)
        loss, recon_l, vq_l = vqvae_loss(recon, imgs, vq_loss_val)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_recon += recon_l.item() * bs
        total_vq += vq_l.item() * bs
        n += bs
    return total_loss / n, total_recon / n, total_vq / n


@torch.no_grad()
def val_epoch(model, loader, device):
    model.eval()
    total_loss, total_recon, total_vq, n = 0, 0, 0, 0
    for imgs, _ in loader:
        imgs = imgs.to(device)
        recon, vq_loss_val, _ = model(imgs)
        loss, recon_l, vq_l = vqvae_loss(recon, imgs, vq_loss_val)
        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_recon += recon_l.item() * bs
        total_vq += vq_l.item() * bs
        n += bs
    return total_loss / n, total_recon / n, total_vq / n


def main():
    parser = argparse.ArgumentParser(description="Train VQ-VAE on Geometry Dash frames")
    parser.add_argument("--data-dir", default="data", help="Root data dir (must contain train/ and val/ subdirs)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--num-embeddings", type=int, default=1024, help="Codebook size")
    parser.add_argument("--embedding-dim", type=int, default=8, help="Codebook vector dimension")
    parser.add_argument("--commitment-cost", type=float, default=0.25, help="Commitment loss weight")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader = make_loader(str(Path(args.data_dir) / "train"), args.batch_size, shuffle=True, augment=True)
    val_loader = make_loader(str(Path(args.data_dir) / "val"), args.batch_size, shuffle=False)
    print(f"Train: {len(train_loader.dataset)} images, Val: {len(val_loader.dataset)} images")

    model = VQVAE(
        num_embeddings=args.num_embeddings,
        embedding_dim=args.embedding_dim,
        commitment_cost=args.commitment_cost,
    ).to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device, weights_only=True))
        print(f"Resumed from {args.resume}")
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    print(f"Codebook: {args.num_embeddings} entries x {args.embedding_dim}d")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(exist_ok=True)
    best_val_recon = float("inf")

    log_path = ckpt_dir / "train_log.csv"
    resuming = args.resume and log_path.exists()
    log_file = open(log_path, "a" if resuming else "w", newline="")
    log_writer = csv.writer(log_file)
    if not resuming:
        log_writer.writerow(["epoch", "train_loss", "train_recon", "train_vq", "val_loss", "val_recon", "val_vq", "lr", "time_s"])

    try:
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_loss, train_recon, train_vq = train_epoch(model, train_loader, optimizer, device)
            val_loss, val_recon, val_vq = val_epoch(model, val_loader, device)
            scheduler.step()
            dt = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch:3d}/{args.epochs} ({dt:.1f}s) | "
                f"Train: {train_loss:.4f} (recon={train_recon:.4f}, vq={train_vq:.4f}) | "
                f"Val: {val_loss:.4f} (recon={val_recon:.4f}, vq={val_vq:.4f}) | "
                f"LR: {lr:.1e}"
            )

            log_writer.writerow([epoch, f"{train_loss:.6f}", f"{train_recon:.6f}", f"{train_vq:.6f}",
                                 f"{val_loss:.6f}", f"{val_recon:.6f}", f"{val_vq:.6f}", f"{lr:.1e}", f"{dt:.1f}"])
            log_file.flush()

            if val_recon < best_val_recon:
                best_val_recon = val_recon
                torch.save(model.state_dict(), ckpt_dir / "vqvae_best.pt")
    except KeyboardInterrupt:
        print("\nInterrupted — saving final checkpoint...")

    log_file.close()
    torch.save(model.state_dict(), ckpt_dir / "vqvae_final.pt")
    print(f"\nTraining complete. Best val recon: {best_val_recon:.4f}")
    print(f"Checkpoints saved to {ckpt_dir}/")
    print(f"Training log saved to {log_path}")


if __name__ == "__main__":
    main()
