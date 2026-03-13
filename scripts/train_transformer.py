"""Train the Transformer world model on tokenized episode data.

Usage:
    python scripts/tokenize_episodes.py --model fsq   # must run first
    python scripts/train_transformer.py
    python scripts/train_transformer.py --context-frames 4 --epochs 400
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepdash.world_model import WorldModel


def _unwrap(model):
    """Access underlying model whether torch.compiled or not."""
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def focal_cross_entropy(logits, targets, gamma=2.0, label_smoothing=0.0):
    """Focal loss: downweights easy (well-classified) tokens, upweights hard ones.

    FL(p_t) = -(1 - p_t)^gamma * log(p_t)

    With gamma=0 this is standard cross-entropy. Higher gamma focuses more on
    hard tokens (e.g. changing regions between frames).

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    """
    ce = F.cross_entropy(logits, targets, reduction='none',
                         label_smoothing=label_smoothing)
    if gamma == 0:
        return ce.mean()
    pt = torch.exp(-ce)
    return (((1 - pt) ** gamma) * ce).mean()


def train_epoch(model, loader, optimizer, scaler, cpc_weight, device,
                token_noise=0.0, label_smoothing=0.0, focal_gamma=2.0):
    model.train()
    m = _unwrap(model)
    tpf = m.tokens_per_frame
    vocab = m.full_vocab_size
    vs = m.vocab_size
    total_loss, total_correct, total_tokens = 0, 0, 0
    total_death_correct, total_death_samples = 0, 0
    total_cpc_loss = 0.0

    use_amp = device.type == "cuda"

    for frame_tokens, actions in loader:
        frame_tokens = frame_tokens.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)

        target = frame_tokens[:, -1]  # (B, 65)

        # Scheduled sampling: corrupt context visual tokens (not status tokens)
        if token_noise > 0:
            ctx = frame_tokens[:, :-1].clone()
            visual = ctx[:, :, :tpf]
            mask = torch.rand_like(visual, dtype=torch.float) < token_noise
            random_tokens = torch.randint(0, vs, visual.shape, device=device)
            visual = torch.where(mask, random_tokens, visual)
            ctx[:, :, :tpf] = visual
            frame_tokens = torch.cat([ctx, frame_tokens[:, -1:]], dim=1)

        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            logits, cpc_loss, mask = model(frame_tokens, actions)
            # Loss on masked tokens only
            token_loss = focal_cross_entropy(
                logits[mask],      # (n_masked, vocab)
                target[mask],      # (n_masked,)
                gamma=focal_gamma,
                label_smoothing=label_smoothing,
            )
            loss = token_loss + cpc_weight * cpc_loss

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        bs = frame_tokens.size(0)
        with torch.no_grad():
            visual_preds = logits[:, :tpf].argmax(dim=-1)
            visual_target = target[:, :tpf]
            total_correct += (visual_preds == visual_target).sum().item()
            total_tokens += bs * tpf
            total_loss += token_loss.item() * bs

            status_target = target[:, tpf]
            status_pred = logits[:, tpf].argmax(dim=-1)
            total_death_correct += (status_pred == status_target).sum().item()
            total_death_samples += bs
            total_cpc_loss += cpc_loss.item() * bs

    return (total_loss / total_death_samples, total_correct / total_tokens,
            total_death_correct / total_death_samples,
            total_cpc_loss / total_death_samples)


@torch.no_grad()
def val_epoch(model, loader, device):
    model.eval()
    m = _unwrap(model)
    tpf = m.tokens_per_frame
    vocab = m.full_vocab_size
    total_loss, total_correct, total_tokens = 0, 0, 0
    total_death_correct, total_death_samples = 0, 0
    total_cpc_loss = 0.0

    use_amp = device.type == "cuda"

    for frame_tokens, actions in loader:
        frame_tokens = frame_tokens.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)

        target = frame_tokens[:, -1]

        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            logits, cpc_loss, mask = model(frame_tokens, actions, mask_ratio=1.0)

        # All tokens masked → loss on all positions
        token_loss = F.cross_entropy(
            logits.reshape(-1, vocab),
            target.reshape(-1),
        )

        bs = frame_tokens.size(0)
        total_loss += token_loss.item() * bs
        total_cpc_loss += cpc_loss.item() * bs

        visual_preds = logits[:, :tpf].argmax(dim=-1)
        visual_target = target[:, :tpf]
        total_correct += (visual_preds == visual_target).sum().item()
        total_tokens += bs * tpf

        status_target = target[:, tpf]
        status_pred = logits[:, tpf].argmax(dim=-1)
        total_death_correct += (status_pred == status_target).sum().item()
        total_death_samples += bs

    return (total_loss / total_death_samples, total_correct / total_tokens,
            total_death_correct / total_death_samples,
            total_cpc_loss / total_death_samples)


def main():
    parser = argparse.ArgumentParser(description="Train Transformer world model")
    parser.add_argument("--episodes-dir", default="data/episodes")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--context-frames", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=1000,
                        help="Tokenizer vocabulary size (1000 for FSQ, 1024 for VQ-VAE)")
    parser.add_argument("--tokens-per-frame", type=int, default=64,
                        help="Tokens per frame (64 for 8x8 FSQ, 36 for 6x6 VQ-VAE)")
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--cpc-weight", type=float, default=0.1)
    parser.add_argument("--token-noise", type=float, default=0.10,
                        help="Scheduled sampling noise rate")
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                        help="Label smoothing for cross-entropy loss")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Focal loss gamma (0 = standard CE)")
    parser.add_argument("--death-oversample", type=int, default=15,
                        help="Repeat death-frame samples this many times (1 = no oversampling)")
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
    train_frames, train_actions, train_weights = [], [], []
    val_frames, val_actions = [], []

    n_deaths = 0
    for ep in all_episodes:
        tokens = np.load(ep / "tokens.npy")  # (T, TPF)
        actions = np.load(ep / "actions.npy")  # (T,)
        T = len(tokens)
        if T < K + 1:
            continue

        is_clear = "clear" in ep.name

        is_val = ep.name in val_episodes
        f_list = val_frames if is_val else train_frames
        a_list = val_actions if is_val else train_actions

        for i in range(T - K):
            frame_window = tokens[i:i + K + 1].astype(np.int64)  # (K+1, TPF)
            action_window = actions[i:i + K].astype(np.int64)

            # Append status token to each frame
            # Target frame (index K in window): DEATH if last frame of death episode
            status = np.full((K + 1, 1), 0, dtype=np.int64)
            is_death_frame = (not is_clear) and (i + K == T - 1)
            if is_death_frame:
                status[K] = 1  # will be mapped to DEATH_TOKEN
            n_deaths += int(is_death_frame)

            # Pack: (K+1, TPF+1) where last col is status
            frame_with_status = np.concatenate([frame_window, status], axis=1)
            f_list.append(frame_with_status)
            a_list.append(action_window)
            if not is_val:
                train_weights.append(
                    float(args.death_oversample) if is_death_frame else 1.0)

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

    # Pre-stack into contiguous tensors (avoids per-item numpy->torch overhead)
    print("Stacking into tensors...")
    train_frames_t = torch.from_numpy(np.stack(train_frames))
    train_actions_t = torch.from_numpy(np.stack(train_actions))
    val_frames_t = torch.from_numpy(np.stack(val_frames))
    val_actions_t = torch.from_numpy(np.stack(val_actions))
    del train_frames, train_actions
    del val_frames, val_actions

    # Remap status column upfront: 0 -> ALIVE_TOKEN, 1 -> DEATH_TOKEN
    for t in (train_frames_t, val_frames_t):
        status = t[:, :, -1]
        status[status == 0] = model.ALIVE_TOKEN
        status[status == 1] = model.DEATH_TOKEN

    train_dataset = TensorDataset(train_frames_t, train_actions_t)
    val_dataset = TensorDataset(val_frames_t, val_actions_t)
    print(f"Train samples: {len(train_dataset)} unique, Val samples: {len(val_dataset)}")
    print(f"Death frames: {n_deaths} unique, weighted {args.death_oversample}x via sampler")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    print(f"Context: {args.context_frames} frames, Sequence length: {model.seq_len}")
    print(f"Vocab: {args.vocab_size} visual + 2 status = {model.full_vocab_size}")

    train_sampler = WeightedRandomSampler(
        train_weights, num_samples=int(sum(train_weights)), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              sampler=train_sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    scaler = torch.GradScaler(device.type)

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
            if "scaler" in state:
                scaler.load_state_dict(state["scaler"])
            start_epoch = state["epoch"] + 1
            best_val_loss = state["best_val_loss"]
            print(f"Resumed from epoch {state['epoch']} (best val loss: {best_val_loss:.4f})")
        else:
            print("No checkpoint found, starting fresh.")

    # torch.compile for fused ops (requires Triton — Linux/Colab only)
    if sys.platform != "win32":
        try:
            model = torch.compile(model)
            print("torch.compile enabled")
        except Exception as e:
            print(f"torch.compile not available, running eager: {e}")
    else:
        print("Skipping torch.compile (not supported on Windows)")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5,
        last_epoch=start_epoch - 2 if start_epoch > 1 else -1)

    log_path = ckpt_dir / "transformer_log.csv"
    # Append if log exists and last epoch is start_epoch-1 (seamless resume)
    append = False
    if log_path.exists() and start_epoch > 1:
        with open(log_path) as f:
            rows = list(csv.reader(f))
            if rows:
                try:
                    last_logged = int(rows[-1][0])
                    append = last_logged == start_epoch - 1
                except (ValueError, IndexError):
                    pass
    if append:
        log_file = open(log_path, "a", newline="")
    else:
        log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    if not append:
        log_writer.writerow(["epoch", "train_loss", "train_acc", "train_death_acc",
                             "train_cpc", "val_loss", "val_acc", "val_death_acc",
                             "val_cpc", "lr", "time_s"])

    patience_counter = 0

    print("First epoch will be slower due to torch.compile tracing...")
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            t0 = time.time()
            train_loss, train_acc, train_death_acc, train_cpc = train_epoch(
                model, train_loader, optimizer, scaler,
                args.cpc_weight, device, token_noise=args.token_noise,
                label_smoothing=args.label_smoothing,
                focal_gamma=args.focal_gamma)
            val_loss, val_acc, val_death_acc, val_cpc = val_epoch(
                model, val_loader, device)
            scheduler.step()
            dt = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch:3d}/{args.epochs} ({dt:.1f}s) | "
                f"Train: loss={train_loss:.4f} acc={train_acc:.3f} "
                f"death={train_death_acc:.3f} cpc={train_cpc:.3f} | "
                f"Val: loss={val_loss:.4f} acc={val_acc:.3f} "
                f"death={val_death_acc:.3f} cpc={val_cpc:.3f} | "
                f"LR: {lr:.1e}"
            )

            log_writer.writerow([
                epoch, f"{train_loss:.6f}", f"{train_acc:.4f}", f"{train_death_acc:.4f}",
                f"{train_cpc:.4f}", f"{val_loss:.6f}", f"{val_acc:.4f}",
                f"{val_death_acc:.4f}", f"{val_cpc:.4f}", f"{lr:.1e}", f"{dt:.1f}"
            ])
            log_file.flush()

            # Save full state
            torch.save({
                "epoch": epoch,
                "model": _unwrap(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_val_loss": best_val_loss,
            }, ckpt_dir / "transformer_state.pt")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(_unwrap(model).state_dict(), ckpt_dir / "transformer_best.pt")
            else:
                patience_counter += 1
                if args.patience > 0 and patience_counter >= args.patience:
                    print(f"\nEarly stopping: val loss did not improve for {args.patience} epochs.")
                    break

            # Defragment CUDA memory periodically
            if epoch % 2 == 0 and device.type == "cuda":
                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("\nInterrupted — saving checkpoint...")
        torch.save({
            "epoch": epoch,
            "model": _unwrap(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_val_loss": best_val_loss,
        }, ckpt_dir / "transformer_state.pt")

    log_file.close()
    torch.save(_unwrap(model).state_dict(), ckpt_dir / "transformer_final.pt")
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to {ckpt_dir}/")


if __name__ == "__main__":
    main()
