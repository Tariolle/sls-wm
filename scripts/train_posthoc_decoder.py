"""Train a post-hoc decoder on frozen FSQ encoder z_q outputs.

Used by E6.5+ pure-JEPA runs where the decoder is removed from training
(`use_recon: false`). The encoder is shaped purely by prediction CE +
CWU-reg, so the training-time FSQ checkpoint has fresh/random decoder
weights. This script loads that checkpoint, freezes the encoder (and
the FSQ quantizer — no learnable params there), and trains a new
Decoder module on `MSE(decoder(z_q), frame)` over all episodes.

The resulting decoder is saved to `<checkpoint_dir>/fsq_posthoc.pt` and
is drop-in-replaceable at inference: load the original fsq_best.pt
first, then overwrite `vae.decoder` state from `fsq_posthoc.pt`.

Because training decisions cannot be affected (decoder is trained
*after* the encoder/transformer are frozen), dream visualisation is
biased neither for nor against the encoder's representational choices.
This is the "faithful viewer" pattern.

Usage:
    python scripts/train_posthoc_decoder.py --config configs/e6.7-recon-cauchysls.yaml
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepdash.config import apply_config
from deepdash.fsq import FSQVAE, fsqvae_loss


class FramesDataset(Dataset):
    """Streams (frame_uint8) from pre-collected episode directories.

    Shifted duplicates (episodes with `_s<sign><n>_<sign><n>$` suffix)
    are excluded — they are training-time augmentation for the WM, not
    additional unique content, and including them would upweight their
    frames in the decoder loss.
    """

    def __init__(self, episode_dirs):
        shift_re = re.compile(r"_s[+-]\d+_[+-]\d+$")
        self.offsets = []
        self.frames_arrays = []
        global_offset = 0
        for ep in episode_dirs:
            if shift_re.search(ep.name):
                continue
            fp = ep / "frames.npy"
            if not fp.exists():
                continue
            arr = np.load(fp, mmap_mode="r")
            self.frames_arrays.append(arr)
            n = arr.shape[0]
            self.offsets.append((global_offset, global_offset + n, len(self.frames_arrays) - 1))
            global_offset += n
        self.total = global_offset

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        # Binary search the bucket. Small N, linear is fine.
        for start, end, arr_idx in self.offsets:
            if start <= idx < end:
                frame = self.frames_arrays[arr_idx][idx - start]
                return torch.from_numpy(np.ascontiguousarray(frame)).float() / 255.0
        raise IndexError(idx)


@torch.no_grad()
def encode_batch(fsq, frames, device):
    """frames: (B, 64, 64) float [0, 1]. Returns z_q (B, D, 8, 8)."""
    x = frames.unsqueeze(1).to(device)
    z_e = fsq.encoder(x)
    z_q, _ = fsq.fsq(z_e)
    return z_q, x


def main():
    parser = argparse.ArgumentParser(
        description="Train a post-hoc decoder on frozen FSQ encoder z_q")
    parser.add_argument("--config", default=None)
    parser.add_argument("--fsq-checkpoint", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--episodes-dir", default="data/death_episodes")
    parser.add_argument("--expert-episodes-dir", default="data/expert_episodes")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--levels", type=int, nargs="+", default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    args = parser.parse_args()

    apply_config(args, section="posthoc_decoder")

    torch.manual_seed(args.seed if args.seed is not None else 42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(ckpt_dir / "posthoc_decoder_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Load the full FSQVAE from the WM run's fsq_best.pt. The encoder +
    # quantizer are frozen; only the decoder receives gradient in this
    # script. The decoder weights in the checkpoint are whatever came out
    # of training — typically fresh/random for E6.5 runs (use_recon=False),
    # or already-trained for E6.4/V5 lineage (use_recon=True). Either way
    # we re-initialize the decoder before training so results don't depend
    # on the starting point.
    fsq = FSQVAE(levels=args.levels).to(device)
    fsq_state = torch.load(args.fsq_checkpoint, map_location=device,
                           weights_only=True)
    fsq_state = {k.removeprefix("_orig_mod."): v for k, v in fsq_state.items()}
    fsq.load_state_dict(fsq_state)
    # Freeze encoder + quantizer.
    for p in fsq.encoder.parameters():
        p.requires_grad = False
    for p in fsq.fsq.parameters():
        p.requires_grad = False
    # Re-init decoder (drop whatever was in the checkpoint) so the
    # training outcome is deterministic w.r.t. the seed.
    for m in fsq.decoder.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    fsq.encoder.eval()
    fsq.fsq.eval()

    print(f"FSQ encoder + quantizer frozen, decoder re-initialized.")
    print(f"Trainable params: {sum(p.numel() for p in fsq.decoder.parameters()):,}")

    # Build dataset from raw frames.npy. Post-hoc decoder doesn't need the
    # windowing / K-frame context — it's a per-frame regression z_q → frame.
    episode_dirs = sorted(Path(args.episodes_dir).glob("*")) + \
                   sorted(Path(args.expert_episodes_dir).glob("*"))
    dataset = FramesDataset(episode_dirs)
    print(f"Frames dataset: {len(dataset):,} frames")

    # Train/val split on frame level (not episode level — for a per-frame
    # regression the distinction doesn't matter much).
    n_val = max(1, int(len(dataset) * args.val_ratio))
    rng = np.random.default_rng(args.seed if args.seed is not None else 42)
    perm = rng.permutation(len(dataset))
    val_idx = set(perm[:n_val].tolist())
    train_ds = torch.utils.data.Subset(
        dataset, [i for i in range(len(dataset)) if i not in val_idx])
    val_ds = torch.utils.data.Subset(dataset, list(val_idx))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    optimizer = torch.optim.AdamW(fsq.decoder.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)

    best_val = float("inf")
    print(f"Training {args.epochs} epochs, batch {args.batch_size}, lr {args.lr}")

    for epoch in range(1, args.epochs + 1):
        fsq.decoder.train()
        total_train_loss, n_train = 0.0, 0
        for frames in train_loader:
            z_q, x = encode_batch(fsq, frames, device)
            recon = fsq.decoder(z_q)
            loss = fsqvae_loss(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * x.size(0)
            n_train += x.size(0)
        train_loss = total_train_loss / max(1, n_train)

        fsq.decoder.eval()
        total_val_loss, n_val_seen = 0.0, 0
        with torch.no_grad():
            for frames in val_loader:
                z_q, x = encode_batch(fsq, frames, device)
                recon = fsq.decoder(z_q)
                loss = fsqvae_loss(recon, x)
                total_val_loss += loss.item() * x.size(0)
                n_val_seen += x.size(0)
        val_loss = total_val_loss / max(1, n_val_seen)

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train={train_loss:.4f} val={val_loss:.4f} lr={lr:.2e}")

        if val_loss < best_val:
            best_val = val_loss
            # Save the full FSQVAE state dict (encoder + quantizer +
            # decoder). This is a drop-in replacement for fsq_best.pt at
            # inference time — play_dream etc. will load this path and
            # get the frozen encoder with the new post-hoc decoder.
            torch.save(fsq.state_dict(), ckpt_dir / "fsq_posthoc.pt")

    print(f"Training complete. Best val loss: {best_val:.4f}")
    print(f"Post-hoc decoder saved to {ckpt_dir}/fsq_posthoc.pt")
    print("At inference, pass --vae-checkpoint {ckpt_dir}/fsq_posthoc.pt "
          "instead of fsq_best.pt.")


if __name__ == "__main__":
    main()
