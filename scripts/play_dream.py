"""Play in the dream world - interactive testing of the world model.

The world model predicts frames in real-time based on your actions.
Tests whether a human can survive in the dream environment and
whether actions visibly affect outcomes.

Controls:
    SPACE / UP  = jump
    R           = restart (new episode, human plays)
    Y           = restart (same episode, controller plays)
    T           = next episode
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
from deepdash.controller import MLPPolicy


def compute_val_set(death_dir, expert_dir):
    """Get global val set (shared across all models)."""
    from deepdash.data_split import get_val_episodes
    return get_val_episodes(death_dir, expert_dir)


class EpisodeLoader:
    """Lazily loads episodes: keeps current + prefetched next ready."""

    def __init__(self, episodes_dir, context_frames, split_filter="all",
                 val_set=None):
        shift_re = re.compile(r"_s[+-]\d+_[+-]\d+$")
        self.context_frames = context_frames
        self.val_set = val_set or set()
        self.dirs = []
        for ep in sorted(Path(episodes_dir).glob("*")):
            if shift_re.search(ep.name):
                continue
            if not ((ep / "tokens.npy").exists() and (ep / "actions.npy").exists()):
                continue
            if split_filter == "val" and ep.name not in self.val_set:
                continue
            if split_filter == "train" and ep.name in self.val_set:
                continue
            self.dirs.append(ep)
        self.rng = np.random.default_rng()
        self._next = None

    def add_dir(self, episodes_dir):
        """Add episodes from another directory."""
        shift_re = re.compile(r"_s[+-]\d+_[+-]\d+$")
        for ep in sorted(Path(episodes_dir).glob("*")):
            if shift_re.search(ep.name):
                continue
            if (ep / "tokens.npy").exists() and (ep / "actions.npy").exists():
                self.dirs.append(ep)

    def __len__(self):
        return len(self.dirs)

    def _load(self, ep_dir):
        from deepdash.data_split import is_val_episode
        tokens = np.load(ep_dir / "tokens.npy").astype(np.int64)
        actions = np.load(ep_dir / "actions.npy").astype(np.int64)
        split = "VAL" if is_val_episode(ep_dir.name, self.val_set) else "TRAIN"
        return tokens, actions, ep_dir.name, split

    def _pick(self):
        """Load a random valid episode."""
        order = self.rng.permutation(len(self.dirs))
        for idx in order:
            tokens, actions, name, split = self._load(self.dirs[idx])
            if len(tokens) >= self.context_frames * 3:
                return tokens, actions, name, split
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
    parser.add_argument("--controller-checkpoint",
                        default="checkpoints/controller_ppo_best.pt")
    parser.add_argument("--episodes-dir", default="data/death_episodes")
    parser.add_argument("--expert-episodes-dir", default="data/expert_episodes")
    parser.add_argument("--context-frames", type=int, default=4)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--scale", type=int, default=6,
                        help="Display scale (6 = 384x384 window)")
    parser.add_argument("--start-pos", choices=["uniform", "beginning"],
                        default="uniform")
    parser.add_argument("--filter", choices=["all", "train", "val"],
                        default="all",
                        help="Filter episodes by train/val split")
    # Model architecture (defaults from configs/v3.yaml)
    parser.add_argument("--config", default=None)
    parser.add_argument("--levels", type=int, nargs="+", default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--tokens-per-frame", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--max-dream-steps", type=int, default=None,
                        help="Max dream steps before auto-restart (0 = unlimited)")
    args = parser.parse_args()

    from deepdash.config import apply_config
    apply_config(args, section="controller_ppo")

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
        adaln=getattr(args, 'adaln', False),
        ffn_variant=getattr(args, 'ffn_variant', 'gelu'),
        ffn_hidden=getattr(args, 'ffn_hidden', None),
        multi_level_readout=getattr(args, 'multi_level_readout', False),
        readout_layers=getattr(args, 'readout_layers', None),
    ).to(device)
    state = torch.load(args.transformer_checkpoint, map_location=device,
                       weights_only=True)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    wm.load_state_dict(state)
    wm.eval()
    print("World model loaded")

    # Load controller (for Y = AI replay)
    controller = None
    ctrl_path = Path(args.controller_checkpoint)
    if ctrl_path.exists():
        controller = MLPPolicy(h_dim=args.embed_dim).to(device)
        state = torch.load(ctrl_path, map_location=device, weights_only=True)
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
        controller.load_state_dict(state)
        controller.eval()
        print("Controller loaded (Y = AI replay)")
    else:
        print(f"No controller at {ctrl_path} (Y disabled)")

    # Episode loader (lazy: loads current + prefetches next)
    val_set = compute_val_set(args.episodes_dir, args.expert_episodes_dir)
    loader = EpisodeLoader(args.episodes_dir, args.context_frames,
                            args.filter, val_set=val_set)
    loader.add_dir(args.expert_episodes_dir)
    print(f"Found {len(loader)} episode directories (filter: {args.filter})")
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

    # Current episode data (for replay)
    current_ep = {}

    def start_dream(tokens, actions, name, split, start=None):
        """Set up a dream from episode data."""
        T = len(tokens)
        K = args.context_frames
        if start is None:
            if args.start_pos == "beginning":
                start = 0
            else:
                # Uniform, excluding last 2*K frames (matches controller training)
                latest = T - K * 3
                start = rng.integers(0, latest + 1) if latest > 0 else 0

        current_ep.update(tokens=tokens, actions=actions,
                          name=name, split=split, start=start)

        ctx_tok = tokens[start:start + K]
        ctx_act = actions[start:start + K]

        status = np.full((K, 1), wm.ALIVE_TOKEN, dtype=np.int64)
        ctx_with_status = np.concatenate([ctx_tok, status], axis=1)
        ct = torch.from_numpy(ctx_with_status[None]).to(device)
        ca = torch.from_numpy(ctx_act[None]).to(device)

        last_frame = decode_tokens(vae, ctx_tok[-1], device)

        print(f"  Episode: {name} [{split}], start frame: {start}")
        return ct, ca, last_frame, 0, split

    def new_episode():
        tokens, actions, name, split = loader.get_next()
        return start_dream(tokens, actions, name, split)

    def replay_episode():
        return start_dream(current_ep['tokens'], current_ep['actions'],
                           current_ep['name'], current_ep['split'],
                           current_ep['start'])

    ctx_t, ctx_a, frame_img, steps, ep_split = new_episode()
    dead = False
    dead_by_death = False
    death_prob_val = 0.0
    action = 0
    best_steps = 0
    ai_mode = False
    print(f"\nControls: SPACE=jump, R=retry, Y=AI replay, T=next episode, Q=quit")
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
                    ctx_t, ctx_a, frame_img, steps, ep_split = replay_episode()
                    dead = False
                    death_prob_val = 0.0
                    ai_mode = False
                if event.key == pygame.K_y and controller is not None:
                    ctx_t, ctx_a, frame_img, steps, ep_split = replay_episode()
                    dead = False
                    death_prob_val = 0.0
                    ai_mode = True
                    print("  AI mode: controller playing")
                if event.key == pygame.K_t:
                    ctx_t, ctx_a, frame_img, steps, ep_split = new_episode()
                    dead = False
                    death_prob_val = 0.0
                    ai_mode = False

        if not running:
            break

        # Render current frame
        rgb = np.stack([frame_img, frame_img, frame_img], axis=-1)
        surf = pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))
        surf = pygame.transform.scale(surf, (W, H))
        screen.blit(surf, (0, 0))

        # HUD (two lines)
        act_str = "JUMP" if action else "idle"
        if ai_mode:
            act_str = "AI:" + act_str
        dp_color = (255, 80, 80) if death_prob_val > 0.3 else (80, 255, 80)
        actual_fps = clock.get_fps()
        line1 = font.render(
            f"{steps:3d}  {act_str:4s}  d:{death_prob_val:.2f}", True, dp_color)
        line2 = font.render(
            f"best:{best_steps}  {ep_split}  {actual_fps:.0f}fps",
            True, (180, 180, 180))
        hud_h = line1.get_height() + line2.get_height() + 6
        hud_w = max(line1.get_width(), line2.get_width()) + 10
        hud_bg = pygame.Surface((hud_w, hud_h))
        hud_bg.set_alpha(180)
        screen.blit(hud_bg, (0, 0))
        screen.blit(line1, (5, 2))
        screen.blit(line2, (5, line1.get_height() + 4))

        if dead:
            overlay = pygame.Surface((W, H))
            overlay.set_alpha(120)
            screen.blit(overlay, (0, 0))
            if dead_by_death:
                txt1 = big_font.render(f"DEAD  step {steps}", True, (255, 50, 50))
            else:
                txt1 = big_font.render(f"END  step {steps}", True, (200, 200, 200))
            txt2 = font.render("R = retry  Y = AI  T = next  Q = quit", True, (255, 255, 255))
            screen.blit(txt1, (W // 2 - txt1.get_width() // 2, H // 2 - 30))
            screen.blit(txt2, (W // 2 - txt2.get_width() // 2, H // 2 + 20))
            pygame.display.flip()
            clock.tick(args.fps)
            continue

        pygame.display.flip()

        # Read action for NEXT step
        if ai_mode:
            # Controller decides
            with torch.no_grad():
                h_t = wm.encode_context(ctx_t, ctx_a)  # (1, D)
                prob, _ = controller(h_t)
                action = 1 if prob[0].item() > 0.5 else 0
        else:
            keys = pygame.key.get_pressed()
            action = 1 if keys[pygame.K_SPACE] else 0

        # World model prediction.
        # bf16 instead of fp16: V5 SwiGLU overflows fp16 at inference, same
        # failure mode that killed training. bf16 is native on Ampere+.
        with torch.no_grad(), torch.amp.autocast(
                device.type, dtype=torch.bfloat16,
                enabled=device.type == "cuda"):
            pred_tokens, death_prob = wm.predict_next_frame(
                ctx_t, ctx_a, temperature=0.0)

        death_prob_val = death_prob[0].item()
        pred_np = pred_tokens[0].cpu().numpy()
        frame_img = decode_tokens(vae, pred_np, device)
        steps += 1

        actually_died = death_prob_val > 0.5
        reached_max = args.max_dream_steps > 0 and steps >= args.max_dream_steps
        if actually_died or reached_max:
            dead = True
            dead_by_death = actually_died
            best_steps = max(best_steps, steps)
            reason = f"death_prob={death_prob_val:.3f}" if actually_died else "max steps"
            label = "DEAD" if actually_died else "END"
            print(f"  {label} at step {steps} ({reason})")
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
