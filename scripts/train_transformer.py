"""Train the Transformer world model on tokenized episode data.

Usage:
    python scripts/tokenize_episodes.py --model fsq   # must run first
    python scripts/train_transformer.py
    python scripts/train_transformer.py --context-frames 4 --epochs 400
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
from deepdash.world_model import WorldModel


def train_epoch(model, loader, optimizer, cpc_weight, device, token_noise=0.0):
    model.train()
    tpf = model.tokens_per_frame
    bs_tok = model.block_size  # tpf + 1 (visual + status)
    total_loss, total_correct, total_tokens = 0, 0, 0
    total_death_correct, total_death_samples = 0, 0
    total_cpc_loss = 0.0

    for frame_tokens, actions, level_ids in loader:
        frame_tokens = frame_tokens.to(device)
        actions = actions.to(device)
        level_ids = level_ids.to(device)

        # Target is the last frame block (visual + status)
        target = frame_tokens[:, -1]  # (B, 65)

        # Scheduled sampling: corrupt context visual tokens (not status tokens)
        if token_noise > 0:
            ctx = frame_tokens[:, :-1].clone()  # (B, K, 65)
            visual = ctx[:, :, :tpf]
            mask = torch.rand_like(visual, dtype=torch.float) < token_noise
            random_tokens = torch.randint(0, model.vocab_size, visual.shape,
                                          device=device)
            visual = torch.where(mask, random_tokens, visual)
            ctx[:, :, :tpf] = visual
            frame_tokens = torch.cat([ctx, frame_tokens[:, -1:]], dim=1)

        logits, cpc_loss = model(frame_tokens, actions, level_ids)
        # Loss over all 65 target positions (64 visual + 1 status)
        token_loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, model.full_vocab_size),
            target.reshape(-1),
        )
        loss = token_loss + cpc_weight * cpc_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        bs = frame_tokens.size(0)
        # Visual token metrics (first tpf positions)
        visual_target = target[:, :tpf]
        visual_logits = logits[:, :tpf]
        total_loss += torch.nn.functional.cross_entropy(
            visual_logits.reshape(-1, model.full_vocab_size),
            visual_target.reshape(-1),
        ).item() * bs * tpf
        preds = visual_logits.argmax(dim=-1)
        total_correct += (preds == visual_target).sum().item()
        total_tokens += bs * tpf

        # Death token metrics (position tpf)
        status_target = target[:, tpf]  # (B,)
        status_pred = logits[:, tpf].argmax(dim=-1)  # (B,)
        total_death_correct += (status_pred == status_target).sum().item()
        total_death_samples += bs
        total_cpc_loss += cpc_loss.item() * bs

    return (total_loss / total_tokens, total_correct / total_tokens,
            total_death_correct / total_death_samples,
            total_cpc_loss / total_death_samples)


@torch.no_grad()
def val_epoch(model, loader, device):
    model.eval()
    tpf = model.tokens_per_frame
    total_loss, total_correct, total_tokens = 0, 0, 0
    total_death_correct, total_death_samples = 0, 0

    for frame_tokens, actions, level_ids in loader:
        frame_tokens = frame_tokens.to(device)
        actions = actions.to(device)
        level_ids = level_ids.to(device)

        target = frame_tokens[:, -1]
        logits, _ = model(frame_tokens, actions, level_ids)

        visual_target = target[:, :tpf]
        visual_logits = logits[:, :tpf]
        token_loss = torch.nn.functional.cross_entropy(
            visual_logits.reshape(-1, model.full_vocab_size),
            visual_target.reshape(-1),
        )

        bs = frame_tokens.size(0)
        total_loss += token_loss.item() * bs * tpf
        preds = visual_logits.argmax(dim=-1)
        total_correct += (preds == visual_target).sum().item()
        total_tokens += bs * tpf

        status_target = target[:, tpf]
        status_pred = logits[:, tpf].argmax(dim=-1)
        total_death_correct += (status_pred == status_target).sum().item()
        total_death_samples += bs

    return (total_loss / total_tokens, total_correct / total_tokens,
            total_death_correct / total_death_samples)


def main():
    parser = argparse.ArgumentParser(description="Train Transformer world model")
    parser.add_argument("--episodes-dir", default="data/episodes")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--context-frames", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=1000,
                        help="Tokenizer vocabulary size (1000 for FSQ, 1024 for VQ-VAE)")
    parser.add_argument("--tokens-per-frame", type=int, default=64,
                        help="Tokens per frame (64 for 8x8 FSQ, 36 for 6x6 VQ-VAE)")
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--cpc-weight", type=float, default=0.1)
    parser.add_argument("--token-noise", type=float, default=0.10,
                        help="Scheduled sampling noise rate")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Split episodes into train/val by episode
    episodes_dir = Path(args.episodes_dir)
    all_episodes = sorted(ep for ep in episodes_dir.glob("*")
                          if (ep / "tokens.npy").exists())
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(all_episodes))
    val_count = max(1, int(len(all_episodes) * args.val_ratio))
    val_episodes = {all_episodes[i].name for i in indices[:val_count]}

    print(f"Total tokenized episodes: {len(all_episodes)}")
    print(f"Val episodes: {val_count}, Train episodes: {len(all_episodes) - val_count}")

    K = args.context_frames
    TPF = args.tokens_per_frame
    train_samples = []
    val_samples = []

    n_deaths = 0
    for ep in all_episodes:
        tokens = np.load(ep / "tokens.npy")  # (T, TPF)
        actions = np.load(ep / "actions.npy")  # (T,)
        T = len(tokens)
        if T < K + 1:
            continue

        # Level ID
        level_id = 0
        meta_path = ep / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            level_id = meta.get("level", 1) - 1

        is_clear = "clear" in ep.name

        target_list = val_samples if ep.name in val_episodes else train_samples
        for i in range(T - K):
            frame_window = tokens[i:i + K + 1].astype(np.int64)  # (K+1, TPF)
            action_window = actions[i:i + K].astype(np.int64)

            # Append status token to each frame
            # Target frame (index K in window): DEATH if last frame of death episode
            status = np.full((K + 1, 1), 0, dtype=np.int64)  # placeholder, filled below
            is_death_frame = (not is_clear) and (i + K == T - 1)
            for f_idx in range(K + 1):
                if f_idx == K and is_death_frame:
                    status[f_idx] = 1  # will be mapped to DEATH_TOKEN
                # Context frames are always alive (status=0 -> ALIVE_TOKEN)

            # Pack: (K+1, TPF+1) where last col is status
            frame_with_status = np.concatenate([frame_window, status], axis=1)
            n_deaths += int(is_death_frame)
            target_list.append((frame_with_status, action_window, level_id))

    # Map status 0/1 to actual token indices (done after model is created)
    # For now store raw, remap in dataset __getitem__

    class SampleDataset(Dataset):
        def __init__(self, samples, alive_token, death_token):
            self.samples = samples
            self.alive_token = alive_token
            self.death_token = death_token

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            f, a, lvl = self.samples[idx]
            f = torch.from_numpy(f.copy())
            # Remap status column: 0 -> ALIVE, 1 -> DEATH
            status = f[:, -1]
            status[status == 0] = self.alive_token
            status[status == 1] = self.death_token
            f[:, -1] = status
            return f, torch.from_numpy(a), lvl

    # Create model first to get token indices
    model = WorldModel(
        vocab_size=args.vocab_size,
        n_actions=2,
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        context_frames=args.context_frames,
        dropout=args.dropout,
        tokens_per_frame=args.tokens_per_frame,
    ).to(device)

    train_dataset = SampleDataset(train_samples, model.ALIVE_TOKEN, model.DEATH_TOKEN)
    val_dataset = SampleDataset(val_samples, model.ALIVE_TOKEN, model.DEATH_TOKEN)
    total_samples = len(train_dataset) + len(val_dataset)
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Death frames: {n_deaths}/{total_samples} ({100*n_deaths/total_samples:.1f}%)")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    print(f"Context: {args.context_frames} frames, Sequence length: {model.seq_len}")
    print(f"Vocab: {args.vocab_size} visual + 2 status = {model.full_vocab_size}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)

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
            print(f"Resumed from epoch {state['epoch']} (best val loss: {best_val_loss:.4f})")
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
        log_writer.writerow(["epoch", "train_loss", "train_acc", "train_death_acc",
                             "train_cpc", "val_loss", "val_acc", "val_death_acc",
                             "lr", "time_s"])

    patience_counter = 0

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            t0 = time.time()
            train_loss, train_acc, train_death_acc, train_cpc = train_epoch(
                model, train_loader, optimizer,
                args.cpc_weight, device, token_noise=args.token_noise)
            val_loss, val_acc, val_death_acc = val_epoch(
                model, val_loader, device)
            scheduler.step()
            dt = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch:3d}/{args.epochs} ({dt:.1f}s) | "
                f"Train: loss={train_loss:.4f} acc={train_acc:.3f} "
                f"death={train_death_acc:.3f} cpc={train_cpc:.3f} | "
                f"Val: loss={val_loss:.4f} acc={val_acc:.3f} death={val_death_acc:.3f} | "
                f"LR: {lr:.1e}"
            )

            log_writer.writerow([
                epoch, f"{train_loss:.6f}", f"{train_acc:.4f}", f"{train_death_acc:.4f}",
                f"{train_cpc:.4f}", f"{val_loss:.6f}", f"{val_acc:.4f}",
                f"{val_death_acc:.4f}", f"{lr:.1e}", f"{dt:.1f}"
            ])
            log_file.flush()

            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
            }, ckpt_dir / "transformer_state.pt")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), ckpt_dir / "transformer_best.pt")
            else:
                patience_counter += 1
                if args.patience > 0 and patience_counter >= args.patience:
                    print(f"\nEarly stopping: val loss did not improve for {args.patience} epochs.")
                    break
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
