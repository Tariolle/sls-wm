"""Play in the dream world - interactive testing of the world model.

The world model predicts frames in real-time based on your actions.
Tests whether a human can survive in the dream environment and
whether actions visibly affect outcomes.

Controls:
    SPACE / UP  = jump
    R           = restart (new episode)
    Q / ESC     = quit

Usage:
    python scripts/play_dream.py
    python scripts/play_dream.py --fps 7.5 --scale 6
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch

try:
    import pygame
except ImportError:
    print("pygame required: pip install pygame")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepdash.fsq import FSQVAE
from deepdash.world_model import WorldModel


class EpisodeLoader:
    """Lazily loads episodes: keeps current + prefetched next ready."""

    def __init__(self, episodes_dir, context_frames):
        shift_re = re.compile(r"_s[+-]\d+_[+-]\d+$")
        self.context_frames = context_frames
        self.dirs = []
        for ep in sorted(Path(episodes_dir).glob("*")):
            if shift_re.search(ep.name):
                continue
            if (ep / "tokens.npy").exists() and (ep / "actions.npy").exists():
                self.dirs.append(ep)
        self.rng = np.random.default_rng()
        self._next = None

    def __len__(self):
        return len(self.dirs)

    def _load(self, ep_dir):
        tokens = np.load(ep_dir / "tokens.npy").astype(np.int64)
        actions = np.load(ep_dir / "actions.npy").astype(np.int64)
        return tokens, actions, ep_dir.name

    def _pick(self):
        """Load a random valid episode."""
        order = self.rng.permutation(len(self.dirs))
        for idx in order:
            tokens, actions, name = self._load(self.dirs[idx])
            if len(tokens) >= self.context_frames + 1:
                return tokens, actions, name
        return None

    def get_next(self):
        """Return prefetched episode and start loading another."""
        if self._next is None:
            self._next = self._pick()
        current = self._next
        self._next = self._pick()
        return current


def decode_tokens(vae, tokens_np, device):
    """Decode 64 token IDs to (64, 64) uint8 grayscale image."""
    indices = torch.from_numpy(tokens_np.astype(np.int64)).reshape(1, 8, 8).to(device)
    with torch.no_grad():
        img = vae.decode_indices(indices)
    return (img[0, 0].cpu().numpy() * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Play in the dream world")
    parser.add_argument("--vae-checkpoint", default="checkpoints/fsq_best.pt")
    parser.add_argument("--transformer-checkpoint",
                        default="checkpoints/transformer_best.pt")
    parser.add_argument("--episodes-dir", default="data/death_episodes")
    parser.add_argument("--context-frames", type=int, default=4)
    parser.add_argument("--fps", type=float, default=15)
    parser.add_argument("--scale", type=int, default=6,
                        help="Display scale (6 = 384x384 window)")
    parser.add_argument("--start-pos", choices=["random", "beginning"],
                        default="random")
    # Model architecture
    parser.add_argument("--levels", type=int, nargs="+", default=[8, 5, 5, 5])
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--tokens-per-frame", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load FSQ-VAE decoder
    vae = FSQVAE(levels=args.levels).to(device)
    state = torch.load(args.vae_checkpoint, map_location=device,
                       weights_only=True)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    vae.load_state_dict(state)
    vae.eval()
    print("FSQ-VAE loaded")

    # Load world model
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

    # Episode loader (lazy: loads current + prefetches next)
    loader = EpisodeLoader(args.episodes_dir, args.context_frames)
    print(f"Found {len(loader)} episode directories")
    if not loader.dirs:
        return

    # Pygame setup
    W = 64 * args.scale
    H = 64 * args.scale
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("DeepDash Dream")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 20)
    big_font = pygame.font.SysFont("monospace", 32, bold=True)

    rng = np.random.default_rng()

    def new_episode():
        tokens, actions, name = loader.get_next()
        T = len(tokens)
        K = args.context_frames
        if args.start_pos == "beginning":
            start = 0
        else:
            start = rng.integers(0, max(1, T - K - 10))

        ctx_tok = tokens[start:start + K]
        ctx_act = actions[start:start + K]

        status = np.full((K, 1), wm.ALIVE_TOKEN, dtype=np.int64)
        ctx_with_status = np.concatenate([ctx_tok, status], axis=1)
        ct = torch.from_numpy(ctx_with_status[None]).to(device)
        ca = torch.from_numpy(ctx_act[None]).to(device)

        # Decode last context frame for initial display
        last_frame = decode_tokens(vae, ctx_tok[-1], device)

        print(f"  Episode: {name}, start frame: {start}")
        return ct, ca, last_frame, 0

    ctx_t, ctx_a, frame_img, steps = new_episode()
    dead = False
    death_prob_val = 0.0
    action = 0
    best_steps = 0

    print(f"\nControls: SPACE/UP=jump, R=restart, Q=quit")
    print(f"FPS: {args.fps}\n")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                if event.key == pygame.K_r:
                    ctx_t, ctx_a, frame_img, steps = new_episode()
                    dead = False
                    death_prob_val = 0.0

        if not running:
            break

        # Render current frame
        rgb = np.stack([frame_img, frame_img, frame_img], axis=-1)
        surf = pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))
        surf = pygame.transform.scale(surf, (W, H))
        screen.blit(surf, (0, 0))

        # HUD
        act_str = "JUMP" if action else "idle"
        dp_color = (255, 80, 80) if death_prob_val > 0.3 else (80, 255, 80)
        hud = font.render(
            f"step:{steps:3d}  {act_str:4s}  death:{death_prob_val:.2f}  "
            f"best:{best_steps}", True, dp_color)
        # Dark background for readability
        hud_bg = pygame.Surface((hud.get_width() + 10, hud.get_height() + 4))
        hud_bg.set_alpha(180)
        screen.blit(hud_bg, (0, 0))
        screen.blit(hud, (5, 2))

        if dead:
            overlay = pygame.Surface((W, H))
            overlay.set_alpha(120)
            screen.blit(overlay, (0, 0))
            txt1 = big_font.render(f"DEAD  step {steps}", True, (255, 50, 50))
            txt2 = font.render("R = restart   Q = quit", True, (255, 255, 255))
            screen.blit(txt1, (W // 2 - txt1.get_width() // 2, H // 2 - 30))
            screen.blit(txt2, (W // 2 - txt2.get_width() // 2, H // 2 + 20))
            pygame.display.flip()
            clock.tick(args.fps)
            continue

        pygame.display.flip()

        # Read action for NEXT step
        keys = pygame.key.get_pressed()
        action = 1 if keys[pygame.K_SPACE] or keys[pygame.K_UP] else 0

        # World model prediction
        with torch.no_grad():
            pred_tokens, death_prob = wm.predict_next_frame(
                ctx_t, ctx_a, temperature=0.0)

        death_prob_val = death_prob[0].item()
        pred_np = pred_tokens[0].cpu().numpy()
        frame_img = decode_tokens(vae, pred_np, device)
        steps += 1

        if death_prob_val > 0.5:
            dead = True
            best_steps = max(best_steps, steps)
            print(f"  DEAD at step {steps} (death_prob={death_prob_val:.3f})")
        else:
            # Shift context
            act_t = torch.tensor([[action]], dtype=torch.long, device=device)
            new_status = torch.full((1, 1), wm.ALIVE_TOKEN,
                                    dtype=torch.long, device=device)
            new_frame = torch.cat([pred_tokens, new_status],
                                  dim=1).unsqueeze(1)
            ctx_t = torch.cat([ctx_t[:, 1:], new_frame], dim=1)
            ctx_a = torch.cat([ctx_a[:, 1:], act_t], dim=1)

        clock.tick(args.fps)

    pygame.quit()
    print(f"\nBest: {best_steps} steps")


if __name__ == "__main__":
    main()
