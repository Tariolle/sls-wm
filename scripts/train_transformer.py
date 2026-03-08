"""Train the Transformer world model on tokenized episode data.

Usage:
    python scripts/tokenize_episodes.py   # must run first
    python scripts/train_transformer.py
    python scripts/train_transformer.py --context-frames 4 --epochs 100
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepdash.world_model import WorldModel, TOKENS_PER_FRAME


class EpisodeTokenDataset(Dataset):
    """Sequential dataset of (context_frames + target, actions) from tokenized episodes."""

    def __init__(self, episodes_dir, context_frames=4, levels=None):
        self.context_frames = context_frames
        self.samples = []  # list of (tokens_window, actions_window)

        episodes_dir = Path(episodes_dir)
        for ep in sorted(episodes_dir.glob("*")):
            tokens_path = ep / "tokens.npy"
            actions_path = ep / "actions.npy"
            if not tokens_path.exists() or not actions_path.exists():
                continue

            # Filter by level if specified
            if levels is not None:
                meta_path = ep / "metadata.json"
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                    if meta.get("level") not in levels:
                        continue

            tokens = np.load(tokens_path)   # (T, 36) uint16
            actions = np.load(actions_path)  # (T,) uint8

            T = len(tokens)
            # Need at least K context frames + 1 target frame
            if T < context_frames + 1:
                continue

            # Sliding window: each sample is K+1 consecutive frames + K actions
            for i in range(T - context_frames):
                frame_window = tokens[i:i + context_frames + 1]  # (K+1, 36)
                action_window = actions[i:i + context_frames]     # (K,)
                self.samples.append((frame_window, action_window))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frames, actions = self.samples[idx]
        return (
            torch.from_numpy(frames.astype(np.int64)),
            torch.from_numpy(actions.astype(np.int64)),
        )


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, total_correct, total_tokens = 0, 0, 0
    for frame_tokens, actions in loader:
        frame_tokens = frame_tokens.to(device)
        actions = actions.to(device)

        # Target is the last frame's tokens
        target = frame_tokens[:, -1]  # (B, 36)

        logits = model(frame_tokens, actions)  # (B, 36, vocab)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, model.vocab_size),
            target.reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        bs = frame_tokens.size(0)
        total_loss += loss.item() * bs * TOKENS_PER_FRAME
        preds = logits.argmax(dim=-1)
        total_correct += (preds == target).sum().item()
        total_tokens += bs * TOKENS_PER_FRAME

    return total_loss / total_tokens, total_correct / total_tokens


@torch.no_grad()
def val_epoch(model, loader, device):
    model.eval()
    total_loss, total_correct, total_tokens = 0, 0, 0
    for frame_tokens, actions in loader:
        frame_tokens = frame_tokens.to(device)
        actions = actions.to(device)

        target = frame_tokens[:, -1]
        logits = model(frame_tokens, actions)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, model.vocab_size),
            target.reshape(-1),
        )

        bs = frame_tokens.size(0)
        total_loss += loss.item() * bs * TOKENS_PER_FRAME
        preds = logits.argmax(dim=-1)
        total_correct += (preds == target).sum().item()
        total_tokens += bs * TOKENS_PER_FRAME

    return total_loss / total_tokens, total_correct / total_tokens


def main():
    parser = argparse.ArgumentParser(description="Train Transformer world model")
    parser.add_argument("--episodes-dir", default="data/episodes")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--context-frames", type=int, default=8)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Fraction of episodes to use for validation")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint (loads model, optimizer, scheduler, epoch)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Split episodes into train/val by episode (not by frame)
    episodes_dir = Path(args.episodes_dir)
    all_episodes = sorted(ep for ep in episodes_dir.glob("*")
                          if (ep / "tokens.npy").exists())
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(all_episodes))
    val_count = max(1, int(len(all_episodes) * args.val_ratio))
    val_episodes = {all_episodes[i].name for i in indices[:val_count]}

    # Build datasets by filtering episode names
    print(f"Total tokenized episodes: {len(all_episodes)}")
    print(f"Val episodes: {val_count}, Train episodes: {len(all_episodes) - val_count}")

    # Create temp dirs with symlinks? No — just build datasets with filtering
    train_samples = []
    val_samples = []
    K = args.context_frames

    for ep in all_episodes:
        tokens_path = ep / "tokens.npy"
        actions_path = ep / "actions.npy"
        tokens = np.load(tokens_path)
        actions = np.load(actions_path)
        T = len(tokens)
        if T < K + 1:
            continue

        target_list = val_samples if ep.name in val_episodes else train_samples
        for i in range(T - K):
            frame_window = tokens[i:i + K + 1].astype(np.int64)
            action_window = actions[i:i + K].astype(np.int64)
            target_list.append((frame_window, action_window))

    class SampleDataset(Dataset):
        def __init__(self, samples):
            self.samples = samples
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            f, a = self.samples[idx]
            return torch.from_numpy(f), torch.from_numpy(a)

    train_dataset = SampleDataset(train_samples)
    val_dataset = SampleDataset(val_samples)
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)

    model = WorldModel(
        vocab_size=1024,
        n_actions=2,
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        context_frames=args.context_frames,
        dropout=args.dropout,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    print(f"Context: {args.context_frames} frames, Sequence length: {model.seq_len}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)

    scheduler = None  # created after resume logic sets start_epoch

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(exist_ok=True)

    start_epoch = 1
    best_val_loss = float("inf")

    if args.resume:
        resume_path = ckpt_dir / "transformer_state.pt"
        if resume_path.exists():
            state = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            start_epoch = state["epoch"] + 1
            best_val_loss = state["best_val_loss"]
            # Scheduler created below with correct last_epoch
            print(f"Resumed from epoch {state['epoch']} (best val loss: {best_val_loss:.4f})")
        else:
            # Fallback: load model weights only from final/best checkpoint
            fallback = ckpt_dir / "transformer_final.pt"
            if not fallback.exists():
                fallback = ckpt_dir / "transformer_best.pt"
            if fallback.exists():
                model.load_state_dict(torch.load(fallback, map_location=device, weights_only=True))
                # Guess epoch from log file
                log_path_check = ckpt_dir / "transformer_log.csv"
                if log_path_check.exists():
                    with open(log_path_check) as f:
                        lines = f.readlines()
                    start_epoch = len(lines)  # header + N data rows, so len = N+1, start = N+1
                    for _ in range(start_epoch - 1):
                        scheduler.step()
                print(f"Resumed model weights from {fallback.name} (epoch ~{start_epoch - 1}, no optimizer state)")
            else:
                print("No checkpoint found, starting fresh.")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5,
        last_epoch=start_epoch - 2 if start_epoch > 1 else -1)

    log_path = ckpt_dir / "transformer_log.csv"
    if args.resume and start_epoch > 1:
        log_file = open(log_path, "a", newline="")
    else:
        log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    if not (args.resume and start_epoch > 1):
        log_writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "time_s"])

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            t0 = time.time()
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
            val_loss, val_acc = val_epoch(model, val_loader, device)
            scheduler.step()
            dt = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch:3d}/{args.epochs} ({dt:.1f}s) | "
                f"Train: loss={train_loss:.4f} acc={train_acc:.3f} | "
                f"Val: loss={val_loss:.4f} acc={val_acc:.3f} | "
                f"LR: {lr:.1e}"
            )

            log_writer.writerow([
                epoch, f"{train_loss:.6f}", f"{train_acc:.4f}",
                f"{val_loss:.6f}", f"{val_acc:.4f}", f"{lr:.1e}", f"{dt:.1f}"
            ])
            log_file.flush()

            # Save full training state (for --resume)
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
            }, ckpt_dir / "transformer_state.pt")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), ckpt_dir / "transformer_best.pt")
    except KeyboardInterrupt:
        print("\nInterrupted — saving checkpoint...")
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
        }, ckpt_dir / "transformer_state.pt")

    log_file.close()
    torch.save(model.state_dict(), ckpt_dir / "transformer_final.pt")
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to {ckpt_dir}/")


if __name__ == "__main__":
    main()
