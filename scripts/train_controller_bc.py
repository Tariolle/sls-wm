"""Behavioral cloning: pretrain CNNPolicy on expert episodes.

For each frame in expert episodes, feeds context through the world model
to get h_t, then trains the controller to predict the expert's action
given (token_ids, h_t) with binary cross-entropy.

Usage:
    python scripts/train_controller_bc.py
    python scripts/train_controller_bc.py --epochs 50 --lr 1e-3
"""

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepdash.world_model import WorldModel
from deepdash.controller import CNNPolicy


def _unwrap(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def load_episodes(episodes_dir, context_frames):
    """Load tokenized episodes (base only, no shifts) with enough frames."""
    shift_re = re.compile(r"_s[+-]\d+_[+-]\d+$")
    K = context_frames
    episodes = []
    for ep in sorted(Path(episodes_dir).glob("*")):
        if shift_re.search(ep.name):
            continue
        tp = ep / "tokens.npy"
        ap = ep / "actions.npy"
        if not tp.exists() or not ap.exists():
            continue
        tokens = np.load(tp).astype(np.int64)
        actions = np.load(ap).astype(np.int64)
        if len(tokens) >= K * 3:
            episodes.append((tokens, actions, ep.name))
    return episodes


def extract_bc_samples(episodes, context_frames, trim_end=0):
    """Extract (context_tokens, context_actions, target_tokens, target_action) tuples.

    Args:
        trim_end: exclude last N frames per episode (e.g. 2*K for death episodes).
    """
    K = context_frames
    all_ctx_tokens = []
    all_ctx_actions = []
    all_target_tokens = []
    all_target_actions = []

    for tokens, actions in episodes:
        T = len(tokens) - trim_end
        for i in range(K, T):
            all_ctx_tokens.append(tokens[i - K:i])
            all_ctx_actions.append(actions[i - K:i])
            all_target_tokens.append(tokens[i])
            all_target_actions.append(actions[i])

    return (np.array(all_ctx_tokens),      # (N, K, 64)
            np.array(all_ctx_actions),      # (N, K)
            np.array(all_target_tokens),    # (N, 64)
            np.array(all_target_actions))   # (N,)


@torch.no_grad()
def compute_hidden_states(model, ctx_tokens, ctx_actions, device, batch_size=256):
    """Run world model on contexts to get h_t for each sample."""
    m = _unwrap(model)
    N = len(ctx_tokens)
    all_h_t = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        ct = ctx_tokens[start:end]
        ca = ctx_actions[start:end]

        # Add alive status token
        status = np.full((*ct.shape[:2], 1), m.ALIVE_TOKEN, dtype=np.int64)
        ct_with_status = np.concatenate([ct, status], axis=2)

        ct_t = torch.from_numpy(ct_with_status).to(device)
        ca_t = torch.from_numpy(ca).to(device)

        with torch.autocast("cuda", dtype=torch.float16,
                            enabled=device.type == "cuda"):
            _, _, h_t = m.predict_next_frame(
                ct_t, ca_t, temperature=0.0, return_hidden=True)

        all_h_t.append(h_t.float().cpu())

    return torch.cat(all_h_t, dim=0)  # (N, 256)


def main():
    parser = argparse.ArgumentParser(
        description="Behavioral cloning for controller")
    parser.add_argument("--episodes-dir", default="data/death_episodes")
    parser.add_argument("--expert-episodes-dir", default="data/expert_episodes")
    parser.add_argument("--transformer-checkpoint",
                        default="checkpoints/transformer_best.pt")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--jump-class-weight", type=float, default=1.5,
                        help="Jump class weight for BCE (0 = auto from data ratio)")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    # Model architecture (must match transformer)
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--embed-dim", type=int, default=384)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--tokens-per-frame", type=int, default=64)
    parser.add_argument("--context-frames", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    import json
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(ckpt_dir / "controller_bc_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Load world model (frozen, for h_t extraction only)
    wm = WorldModel(
        vocab_size=args.vocab_size, embed_dim=args.embed_dim,
        n_heads=args.n_heads, n_layers=args.n_layers,
        context_frames=args.context_frames, dropout=args.dropout,
        tokens_per_frame=args.tokens_per_frame,
    ).to(device)
    state = torch.load(args.transformer_checkpoint, map_location=device,
                       weights_only=True)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    wm.load_state_dict(state)
    wm.eval()
    print("World model loaded")

    # Load episodes (death + expert)
    from deepdash.data_split import get_val_episodes, is_val_episode
    K = args.context_frames
    val_set = get_val_episodes(args.episodes_dir, args.expert_episodes_dir)

    all_eps = load_episodes(args.episodes_dir, K) + \
              load_episodes(args.expert_episodes_dir, K)
    print(f"Loaded {len(all_eps)} episodes")
    if not all_eps:
        print("No episodes found!")
        return

    # Episode-level split (consistent with FSQ/Transformer)
    train_eps = [(t, a) for t, a, name in all_eps if not is_val_episode(name, val_set)]
    val_eps = [(t, a) for t, a, name in all_eps if is_val_episode(name, val_set)]

    # Extract BC samples: trim last 2*K frames (death: outcome determined, expert: win animation)
    train_ctx, train_act, train_tok, train_actions = \
        extract_bc_samples(train_eps, K, trim_end=K * 2)
    val_ctx, val_act, val_tok, val_actions = \
        extract_bc_samples(val_eps, K, trim_end=K * 2)

    # Concatenate for unified indexing
    ctx_tokens = np.concatenate([train_ctx, val_ctx])
    ctx_actions = np.concatenate([train_act, val_act])
    target_tokens = np.concatenate([train_tok, val_tok])
    target_actions = np.concatenate([train_actions, val_actions])
    N = len(target_actions)

    train_idx = np.arange(len(train_actions))
    val_idx = np.arange(len(train_actions), N)

    print(f"BC samples: {len(train_idx)} train, {len(val_idx)} val, {N} total")
    print(f"Action distribution: "
          f"{target_actions.sum()}/{N} jumps "
          f"({target_actions.mean() * 100:.1f}%)")

    # Precompute h_t for all samples (one-time cost)
    print("Computing hidden states from world model...")
    t0 = time.time()
    all_h_t = compute_hidden_states(wm, ctx_tokens, ctx_actions, device)
    print(f"  Done in {time.time() - t0:.1f}s")

    # Free world model memory
    del wm
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Prepare tensors
    all_target_tokens = torch.from_numpy(target_tokens).long()
    all_target_actions = torch.from_numpy(target_actions).float()

    # Class weight: upweight jumps so model can't just predict idle
    jump_ratio = target_actions.mean()
    if args.jump_class_weight > 0:
        pos_weight = torch.tensor(args.jump_class_weight, device=device)
    else:
        pos_weight = torch.tensor((1 - jump_ratio) / jump_ratio, device=device)
    print(f"Jump class weight: {pos_weight.item():.2f}x (data ratio: {(1-jump_ratio)/jump_ratio:.2f}x)")

    # Initialize controller
    controller = CNNPolicy(vocab_size=args.vocab_size).to(device)
    optimizer = torch.optim.AdamW(controller.parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    n_params = sum(p.numel() for p in controller.parameters())
    print(f"CNNPolicy: {n_params:,} parameters")

    # Logging
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "controller_bc_log.csv"
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "train_loss", "train_acc",
                         "val_loss", "val_acc", "lr", "time_s"])

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # --- Train ---
        controller.train()
        perm = rng.permutation(len(train_idx))
        train_loss_sum, train_correct, train_total = 0.0, 0, 0

        for start in range(0, len(perm), args.batch_size):
            batch_perm = perm[start:start + args.batch_size]
            idx = train_idx[batch_perm]

            tokens = all_target_tokens[idx].to(device)
            h_t = all_h_t[idx].to(device)
            actions = all_target_actions[idx].to(device)

            features = controller._encode(tokens, h_t)
            logits = controller.actor(features).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(
                logits, actions, pos_weight=pos_weight)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
            optimizer.step()

            bs = len(idx)
            train_loss_sum += loss.item() * bs
            pred = (logits > 0).float()
            train_correct += (pred == actions).sum().item()
            train_total += bs

        scheduler.step()
        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total

        # --- Val ---
        controller.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for start in range(0, len(val_idx), args.batch_size):
                idx = val_idx[start:start + args.batch_size]

                tokens = all_target_tokens[idx].to(device)
                h_t = all_h_t[idx].to(device)
                actions = all_target_actions[idx].to(device)

                features = controller._encode(tokens, h_t)
                logits = controller.actor(features).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(
                    logits, actions, pos_weight=pos_weight)

                bs = len(idx)
                val_loss_sum += loss.item() * bs
                pred = (logits > 0).float()
                val_correct += (pred == actions).sum().item()
                val_total += bs

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total
        lr = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0

        print(f"Epoch {epoch:3d}/{args.epochs} ({dt:.1f}s) | "
              f"Train: loss={train_loss:.4f} acc={train_acc:.3f} | "
              f"Val: loss={val_loss:.4f} acc={val_acc:.3f} | "
              f"LR: {lr:.1e}")

        log_writer.writerow([
            epoch, f"{train_loss:.6f}", f"{train_acc:.4f}",
            f"{val_loss:.6f}", f"{val_acc:.4f}",
            f"{lr:.1e}", f"{dt:.1f}"])
        log_file.flush()

        # Save best + early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(controller.state_dict(),
                       ckpt_dir / "controller_bc_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"Early stopping at epoch {epoch}")
                break

    log_file.close()
    torch.save(controller.state_dict(), ckpt_dir / "controller_bc_final.pt")
    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints: {ckpt_dir}/controller_bc_best.pt")


if __name__ == "__main__":
    main()
