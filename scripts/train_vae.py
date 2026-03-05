"""Train the VAE on preprocessed Geometry Dash frames."""

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepdash.vae import VAE, vae_loss


def make_loader(data_dir, batch_size, shuffle=True):
    """Load PNG frames as normalized [0,1] tensors."""
    transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
    dataset = ImageFolder(data_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)


def train_epoch(model, loader, optimizer, device, beta=1.0):
    model.train()
    total_loss, total_recon, total_kl, n = 0, 0, 0, 0
    for imgs, _ in loader:
        imgs = imgs.to(device)
        recon, mu, logvar = model(imgs)
        loss, recon_l, kl_l = vae_loss(recon, imgs, mu, logvar, beta=beta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_recon += recon_l.item() * bs
        total_kl += kl_l.item() * bs
        n += bs
    return total_loss / n, total_recon / n, total_kl / n


@torch.no_grad()
def val_epoch(model, loader, device, beta=1.0):
    model.eval()
    total_loss, total_recon, total_kl, n = 0, 0, 0, 0
    for imgs, _ in loader:
        imgs = imgs.to(device)
        recon, mu, logvar = model(imgs)
        loss, recon_l, kl_l = vae_loss(recon, imgs, mu, logvar, beta=beta)
        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_recon += recon_l.item() * bs
        total_kl += kl_l.item() * bs
        n += bs
    return total_loss / n, total_recon / n, total_kl / n


def main():
    parser = argparse.ArgumentParser(description="Train VAE on Geometry Dash frames")
    parser.add_argument("--data-dir", default="data", help="Root data dir (must contain train/ and val/ subdirs with class folders)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--beta-max", type=float, default=0.1, help="Max KL weight (default: 0.1)")
    parser.add_argument("--beta-warmup", type=int, default=20, help="Epochs to linearly anneal beta from 0 to beta-max")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ImageFolder expects subdirs as classes — our frames are flat PNGs.
    # We need a dummy class subdir structure: data/train/frames/*.png
    train_dir = Path(args.data_dir) / "train"
    val_dir = Path(args.data_dir) / "val"

    # Check if frames are directly in train/ (no subdirs) — create symlink wrapper if needed
    train_pngs = list(train_dir.glob("*.png"))
    if train_pngs:
        # Frames are flat — wrap in a dummy subdir for ImageFolder
        wrapper_train = train_dir / "0"
        wrapper_val = val_dir / "0"
        if not wrapper_train.exists():
            print("ImageFolder requires class subdirs. Creating wrapper dirs...")
            wrapper_train.mkdir()
            wrapper_val.mkdir()
            for p in train_dir.glob("*.png"):
                p.rename(wrapper_train / p.name)
            for p in val_dir.glob("*.png"):
                p.rename(wrapper_val / p.name)
            print("Done.")

    train_loader = make_loader(str(train_dir), args.batch_size, shuffle=True)
    val_loader = make_loader(str(val_dir), args.batch_size, shuffle=False)
    print(f"Train: {len(train_loader.dataset)} images, Val: {len(val_loader.dataset)} images")

    model = VAE().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(exist_ok=True)
    best_val_loss = float("inf")

    log_path = ckpt_dir / "train_log.csv"
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "train_loss", "train_recon", "train_kl", "val_loss", "val_recon", "val_kl", "beta", "lr", "time_s"])

    print(f"Beta annealing: 0 -> {args.beta_max} over {args.beta_warmup} epochs")

    for epoch in range(1, args.epochs + 1):
        beta = min(args.beta_max, args.beta_max * epoch / args.beta_warmup)
        t0 = time.time()
        train_loss, train_recon, train_kl = train_epoch(model, train_loader, optimizer, device, beta=beta)
        val_loss, val_recon, val_kl = val_epoch(model, val_loader, device, beta=beta)
        scheduler.step(val_loss)
        dt = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d}/{args.epochs} ({dt:.1f}s) | "
            f"Train: {train_loss:.4f} (recon={train_recon:.4f}, kl={train_kl:.4f}) | "
            f"Val: {val_loss:.4f} (recon={val_recon:.4f}, kl={val_kl:.4f}) | "
            f"beta={beta:.4f} LR: {lr:.1e}"
        )

        log_writer.writerow([epoch, f"{train_loss:.6f}", f"{train_recon:.6f}", f"{train_kl:.6f}", f"{val_loss:.6f}", f"{val_recon:.6f}", f"{val_kl:.6f}", f"{beta:.4f}", f"{lr:.1e}", f"{dt:.1f}"])
        log_file.flush()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_dir / "vae_best.pt")

    log_file.close()
    torch.save(model.state_dict(), ckpt_dir / "vae_final.pt")
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to {ckpt_dir}/")
    print(f"Training log saved to {log_path}")


if __name__ == "__main__":
    main()
