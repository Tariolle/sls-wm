"""Encode episode frames through frozen FSQ to produce token sequences.

V7 Phase 0 reproduction of V3-deploy's tokenization pipeline (commit 75fe40a).
Verbatim port; FSQVAE.encode signature is unchanged in current code.

Creates one ``tokens.npy`` per base episode plus optional shift-augmented
sibling directories named ``<ep>_s{dx:+d}_{dy:+d}``. Shifted aug_dirs
contain only ``tokens.npy`` (encoded from shifted pixels) plus symlinks
back to the original ``actions.npy`` and ``frames.npy``. Shifted pixels
themselves are NOT persisted on disk (only their token encoding is).

Usage:
    # No shift (just tokenize originals):
    python scripts/tokenize_episodes.py --checkpoint checkpoints/fsq_best.pt
    # V3-deploy default for transformer training:
    python scripts/tokenize_episodes.py --checkpoint checkpoints/fsq_best.pt \\
        --shifts-v -4 -2 0 2 4
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _shift_frames(frames, dx, dy):
    """Shift frames by (dx, dy) pixels with edge padding.

    dx>0 shifts content right (new pixels appear on left edge).
    dy>0 shifts content down (new pixels appear on top edge).
    """
    if dx == 0 and dy == 0:
        return frames
    shifted = np.roll(np.roll(frames, dx, axis=-1), dy, axis=-2)
    if dx > 0:
        shifted[..., :, :dx] = frames[..., :, :1]
    elif dx < 0:
        shifted[..., :, dx:] = frames[..., :, -1:]
    if dy > 0:
        shifted[..., :dy, :] = frames[..., :1, :]
    elif dy < 0:
        shifted[..., dy:, :] = frames[..., -1:, :]
    return shifted


def _tokenize_frames(model, frames, batch_size, tokens_per_frame, device):
    """Encode frames through the model and return flat token array."""
    all_tokens = []
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i + batch_size]
        x = torch.from_numpy(batch).float().unsqueeze(1).to(device) / 255.0
        indices = model.encode(x)
        all_tokens.append(indices.cpu().reshape(-1, tokens_per_frame).numpy())
    return np.concatenate(all_tokens, axis=0).astype(np.uint16)


SHIFT_AUG_RE = re.compile(r"_s[+-]\d+_[+-]\d+$")


def main():
    parser = argparse.ArgumentParser(description="Tokenize episodes with frozen FSQ")
    parser.add_argument("--episodes-dir", default="data/death_episodes")
    parser.add_argument("--checkpoint", default="checkpoints/fsq_best.pt")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--levels", type=int, nargs="+", default=[8, 5, 5, 5])
    parser.add_argument("--shifts", type=int, nargs="+", default=None,
                        help="Horizontal pixel shifts (e.g. -4 -2 0 2 4)")
    parser.add_argument("--shifts-v", type=int, nargs="+", default=None,
                        help="Vertical pixel shifts (e.g. -4 -2 0 2 4)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    from deepdash.fsq import FSQVAE
    model = FSQVAE(levels=args.levels).to(device)
    tokens_per_frame = 64

    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded FSQ from {args.checkpoint} (levels={args.levels})")

    episodes_dir = Path(args.episodes_dir)
    episodes = sorted(ep for ep in episodes_dir.glob("*")
                      if (ep / "frames.npy").exists()
                      and not SHIFT_AUG_RE.search(ep.name))
    print(f"Found {len(episodes)} base episodes (aug_dirs filtered)")

    shifts_h = args.shifts or [0]
    shifts_v = args.shifts_v or [0]
    shift_combos = [(dx, dy) for dx in shifts_h for dy in shifts_v]
    aug_shifts = [(dx, dy) for dx, dy in shift_combos if (dx, dy) != (0, 0)]

    if aug_shifts:
        print(f"Shift augmentation: {len(aug_shifts)} shifted variants per episode")
        print(f"  Horizontal: {shifts_h}, Vertical: {shifts_v}")

    def _tokens_valid(path):
        if not path.exists():
            return False
        try:
            np.load(path)
            return True
        except (EOFError, ValueError):
            path.unlink()
            print(f"  Deleted corrupt {path}")
            return False

    total_frames = 0
    skipped = 0
    aug_created = 0

    with torch.no_grad():
        for ep in episodes:
            frames = None
            if _tokens_valid(ep / "tokens.npy"):
                skipped += 1
            else:
                frames = np.load(ep / "frames.npy")
                total_frames += len(frames)
                tokens = _tokenize_frames(model, frames, args.batch_size,
                                          tokens_per_frame, device)
                np.save(ep / "tokens.npy", tokens)
                print(f"  {ep.name}: {len(frames)} frames -> {tokens.shape} tokens")

            for dx, dy in aug_shifts:
                aug_name = f"{ep.name}_s{dx:+d}_{dy:+d}"
                aug_dir = episodes_dir / aug_name
                if _tokens_valid(aug_dir / "tokens.npy"):
                    continue
                if frames is None:
                    frames = np.load(ep / "frames.npy")
                shifted = _shift_frames(frames, dx, dy)
                tokens = _tokenize_frames(model, shifted, args.batch_size,
                                          tokens_per_frame, device)
                aug_dir.mkdir(exist_ok=True)
                np.save(aug_dir / "tokens.npy", tokens)
                # Link actions.npy and frames.npy from original (symlink on
                # Linux, fall back to copy on Windows). frames.npy here is
                # the UNSHIFTED original by design - shifted pixels live
                # only in tokens.npy. Aug_dirs are filtered out of
                # train_fsq.py glob to prevent the V3-deploy duplication
                # bug from firing.
                for fname in ("actions.npy", "frames.npy"):
                    dst = aug_dir / fname
                    if not dst.exists():
                        src = (ep / fname).resolve()
                        try:
                            os.symlink(src, dst)
                        except OSError:
                            import shutil
                            shutil.copy2(src, dst)
                aug_created += 1
                print(f"  {aug_name}: shift ({dx:+d},{dy:+d}) -> {tokens.shape} tokens")

    if skipped:
        print(f"Skipped {skipped} already-tokenized episodes.")
    if aug_created:
        print(f"Created {aug_created} shifted augmentations.")
    print(f"\nDone. Tokenized {total_frames} new frames across "
          f"{len(episodes) - skipped} episodes.")


if __name__ == "__main__":
    main()
