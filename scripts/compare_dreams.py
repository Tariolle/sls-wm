"""Sequential dream comparison: same warmup, free play on each model in turn.

Draws one episode and a start frame. Plays through each transformer
sequentially, all starting from the same K-frame warmup from the same
episode. You play freely each time. On death / end / E, the script
advances to the next model. After all models have been shown, you can
replay the same warmup or draw a new episode.

This is not action-deterministic: since each model dreams a different
world, the actions you take in one model cannot be meaningfully replayed
in another. The only thing held constant is the warmup.

All models must share the same architecture (same --config).

Controls (same during play and after a rollout ends):
    SPACE / UP = jump
    Q          = replay on the SAME model (retry, same warmup)
    W          = advance to next model (wraps around after the last)
    E          = draw a new episode
    R          = restart the cycle (back to model 0, same warmup)
    ESC        = quit

During play, Q/W/E/R abort the current rollout and immediately trigger the
corresponding navigation.

Usage:
    python scripts/compare_dreams.py \
        --config configs/e6.7-recon-cauchysls.yaml \
        --vae-checkpoint checkpoints_e6.7/fsq_best.pt \
        --transformer-checkpoint \
            checkpoints_e6.4/transformer_best.pt \
            checkpoints_e6.7/transformer_best.pt \
        --episodes-dir "C:/Users/Florent/sls-wm-data/death_episodes" \
        --expert-episodes-dir "C:/Users/Florent/sls-wm-data/expert_episodes"
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
    """Loads episodes lazily (current + prefetched next) and re-encodes
    frames through the currently-loaded FSQ."""

    def __init__(self, episodes_dir, context_frames, split_filter="all",
                 val_set=None, vae=None, device=None):
        shift_re = re.compile(r"_s[+-]\d+_[+-]\d+$")
        self.context_frames = context_frames
        self.val_set = val_set or set()
        self.vae = vae
        self.device = device
        self.dirs = []
        for ep in sorted(Path(episodes_dir).glob("*")):
            if shift_re.search(ep.name):
                continue
            if not ((ep / "frames.npy").exists() and (ep / "actions.npy").exists()):
                continue
            if split_filter == "val" and ep.name not in self.val_set:
                continue
            if split_filter == "train" and ep.name in self.val_set:
                continue
            self.dirs.append(ep)
        self.rng = np.random.default_rng()
        self._next = None

    def add_dir(self, episodes_dir):
        shift_re = re.compile(r"_s[+-]\d+_[+-]\d+$")
        for ep in sorted(Path(episodes_dir).glob("*")):
            if shift_re.search(ep.name):
                continue
            if (ep / "frames.npy").exists() and (ep / "actions.npy").exists():
                self.dirs.append(ep)

    def __len__(self):
        return len(self.dirs)

    def _load(self, ep_dir):
        from deepdash.data_split import is_val_episode
        frames = np.load(ep_dir / "frames.npy")
        actions = np.load(ep_dir / "actions.npy").astype(np.int64)
        with torch.no_grad():
            x = torch.from_numpy(frames).float().to(self.device) / 255.0
            x = x.unsqueeze(1)
            indices = self.vae.encode(x)
            tokens = indices.view(indices.size(0), -1).cpu().numpy().astype(np.int64)
        split = "VAL" if is_val_episode(ep_dir.name, self.val_set) else "TRAIN"
        return tokens, actions, ep_dir.name, split

    def _pick(self):
        order = self.rng.permutation(len(self.dirs))
        for idx in order:
            tokens, actions, name, split = self._load(self.dirs[idx])
            if len(tokens) >= self.context_frames * 3:
                return tokens, actions, name, split
        return None

    def get_next(self):
        if self._next is None:
            self._next = self._pick()
        current = self._next
        self._next = self._pick()
        return current


def decode_tokens(vae, tokens_np, device):
    indices = torch.from_numpy(tokens_np.astype(np.int64)).reshape(1, 8, 8).to(device)
    with torch.no_grad():
        img = vae.decode_indices(indices)
    return (img[0, 0].cpu().numpy() * 255).astype(np.uint8)


def load_transformer(args, ckpt_path, device):
    wm = WorldModel(
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim, n_heads=args.n_heads, n_layers=args.n_layers,
        context_frames=args.context_frames, dropout=args.dropout,
        tokens_per_frame=args.tokens_per_frame,
        adaln=getattr(args, "adaln", False),
        fsq_dim=len(args.levels) if getattr(args, "levels", None) else None,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    wm.load_state_dict(state, strict=False)
    wm.eval()
    return wm


def _default_label(path):
    p = Path(path).parent.name
    return p[len("checkpoints_"):] if p.startswith("checkpoints_") else p


def build_initial_context(tokens, actions, start, K, alive_token, device):
    ctx_tok = tokens[start:start + K]
    ctx_act = actions[start:start + K]
    status = np.full((K, 1), alive_token, dtype=np.int64)
    ctx_with_status = np.concatenate([ctx_tok, status], axis=1)
    ct = torch.from_numpy(ctx_with_status[None]).to(device)
    ca = torch.from_numpy(ctx_act[None]).to(device)
    return ct, ca, ctx_tok[-1]


def main():
    parser = argparse.ArgumentParser(description="Sequential dream comparison")
    parser.add_argument("--config", default=None)
    parser.add_argument("--vae-checkpoint", required=True)
    parser.add_argument("--transformer-checkpoint", nargs="+", required=True,
                        help="One or more transformer checkpoints to compare sequentially")
    parser.add_argument("--label", nargs="+", default=None,
                        help="Labels matching --transformer-checkpoint (default: parent dir name)")
    parser.add_argument("--episodes-dir", default="data/death_episodes")
    parser.add_argument("--expert-episodes-dir", default="data/expert_episodes")
    parser.add_argument("--context-frames", type=int, default=4)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--scale", type=int, default=6)
    parser.add_argument("--start-pos", choices=["uniform", "beginning"],
                        default="uniform")
    parser.add_argument("--filter", choices=["all", "train", "val"],
                        default="all")
    # Model architecture (filled from config; shared across all checkpoints)
    parser.add_argument("--levels", type=int, nargs="+", default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--tokens-per-frame", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--max-dream-steps", type=int, default=None,
                        help="Max dream steps per model (0 = unlimited)")
    args = parser.parse_args()

    from deepdash.config import apply_config
    apply_config(args, section="controller_ppo")

    if args.label is not None and len(args.label) != len(args.transformer_checkpoint):
        raise SystemExit("--label count must match --transformer-checkpoint")
    labels = args.label or [_default_label(p) for p in args.transformer_checkpoint]
    n_models = len(args.transformer_checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    vae = FSQVAE(levels=args.levels).to(device)
    state = torch.load(args.vae_checkpoint, map_location=device, weights_only=True)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    vae.load_state_dict(state)
    vae.eval()
    print("FSQ-VAE loaded")

    models = []
    for i, ckpt in enumerate(args.transformer_checkpoint):
        wm = load_transformer(args, ckpt, device)
        models.append(wm)
        print(f"  [{i}] {labels[i]}  <- {ckpt}")
    alive_token = models[0].ALIVE_TOKEN

    from deepdash.data_split import get_val_episodes
    val_set = get_val_episodes(args.episodes_dir, args.expert_episodes_dir)
    loader = EpisodeLoader(args.episodes_dir, args.context_frames,
                            args.filter, val_set=val_set,
                            vae=vae, device=device)
    loader.add_dir(args.expert_episodes_dir)
    print(f"Found {len(loader)} episode directories (filter: {args.filter})")
    if not loader.dirs:
        return

    W = 64 * args.scale
    H = 64 * args.scale
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("DeepDash Dream Comparison")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 18)
    label_font = pygame.font.SysFont("monospace", 20, bold=True)
    big_font = pygame.font.SysFont("monospace", 28, bold=True)

    current_ep = {}

    def pick_start(tokens_len, K):
        if args.start_pos == "beginning":
            return 0
        latest = tokens_len - K * 3
        rng = np.random.default_rng()
        return int(rng.integers(0, latest + 1)) if latest > 0 else 0

    def new_episode():
        tokens, actions, name, split = loader.get_next()
        start = pick_start(len(tokens), args.context_frames)
        current_ep.update(tokens=tokens, actions=actions, name=name,
                          split=split, start=start)
        print(f"Episode: {name} [{split}], start frame: {start}")

    def _render_hud(model_idx, steps, action, death_prob_val, split, hint,
                    dead=False, dead_by_death=False):
        dp_color = (255, 80, 80) if death_prob_val > 0.3 else (80, 255, 80)
        actual_fps = clock.get_fps()
        act_str = "JUMP" if action else "idle"
        label = labels[model_idx]
        line1 = label_font.render(
            f"[{model_idx + 1}/{n_models}] {label}",
            True, (255, 255, 255))
        line2 = font.render(
            f"step {steps:3d}  {act_str:4s}  d:{death_prob_val:.2f}",
            True, dp_color)
        line3 = font.render(f"{split}  {actual_fps:.0f}fps  {hint}",
                            True, (180, 180, 180))
        hud_w = max(line1.get_width(), line2.get_width(), line3.get_width()) + 10
        hud_h = line1.get_height() + line2.get_height() + line3.get_height() + 10
        hud_bg = pygame.Surface((hud_w, hud_h))
        hud_bg.set_alpha(180)
        screen.blit(hud_bg, (0, 0))
        screen.blit(line1, (5, 2))
        screen.blit(line2, (5, 2 + line1.get_height()))
        screen.blit(line3, (5, 2 + line1.get_height() + line2.get_height()))
        if dead:
            overlay = pygame.Surface((W, H))
            overlay.set_alpha(120)
            screen.blit(overlay, (0, 0))
            msg = f"DEAD step {steps}" if dead_by_death else f"END step {steps}"
            color = (255, 50, 50) if dead_by_death else (200, 200, 200)
            txt = big_font.render(msg, True, color)
            screen.blit(txt, (W // 2 - txt.get_width() // 2, H // 2 - 15))

    def _blit_frame(frame_img):
        rgb = np.stack([frame_img, frame_img, frame_img], axis=-1)
        surf = pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))
        surf = pygame.transform.scale(surf, (W, H))
        screen.blit(surf, (0, 0))

    NAV_KEYS = {
        pygame.K_q: "retry",
        pygame.K_w: "next",
        pygame.K_e: "new_episode",
        pygame.K_r: "restart_cycle",
    }
    HUD_HINT = "[SPACE=jump Q=retry W=next E=new R=restart ESC=quit]"

    def post_rollout_wait(model_idx, frame_img, steps, dead_by_death, split,
                          last_death_prob):
        """Freeze on the last frame and wait for a navigation key."""
        while True:
            _blit_frame(frame_img)
            _render_hud(model_idx, steps, 0, last_death_prob, split, HUD_HINT,
                        dead=True, dead_by_death=dead_by_death)
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "quit"
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return "quit"
                    if event.key in NAV_KEYS:
                        return NAV_KEYS[event.key]
            clock.tick(args.fps)

    def play_model(model_idx):
        """Free-play rollout on one model starting from the shared warmup.
        Returns one of: 'quit', 'retry', 'next', 'new_episode', 'restart_cycle'.
        """
        K = args.context_frames
        tokens = current_ep["tokens"]
        actions = current_ep["actions"]
        start = current_ep["start"]
        split = current_ep["split"]
        label = labels[model_idx]

        ct, ca, last_tokens = build_initial_context(
            tokens, actions, start, K, alive_token, device)
        frame_img = decode_tokens(vae, last_tokens, device)

        steps = 0
        action = 0
        death_prob_val = 0.0
        print(f"[{model_idx + 1}/{n_models}] {label}")

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "quit"
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return "quit"
                    if event.key in NAV_KEYS:
                        return NAV_KEYS[event.key]

            _blit_frame(frame_img)
            _render_hud(model_idx, steps, action, death_prob_val, split,
                        HUD_HINT)
            pygame.display.flip()

            # Read action for NEXT step
            keys = pygame.key.get_pressed()
            action = 1 if keys[pygame.K_SPACE] else 0

            with torch.no_grad():
                pred_tokens, death_prob = models[model_idx].predict_next_frame(
                    ct, ca, temperature=0.0)
            death_prob_val = float(death_prob[0].item())
            pred_np = pred_tokens[0].cpu().numpy()
            frame_img = decode_tokens(vae, pred_np, device)
            steps += 1

            actually_died = death_prob_val > 0.5
            reached_max = (args.max_dream_steps and args.max_dream_steps > 0
                           and steps >= args.max_dream_steps)
            if actually_died or reached_max:
                reason = (f"death_prob={death_prob_val:.3f}"
                          if actually_died else "max steps")
                kind = "DEAD" if actually_died else "END"
                print(f"  {kind} at step {steps} ({reason})")
                return post_rollout_wait(
                    model_idx, frame_img, steps, actually_died, split,
                    death_prob_val)

            act_t = torch.tensor([[action]], dtype=torch.long, device=device)
            new_status = torch.full((1, 1), alive_token,
                                    dtype=torch.long, device=device)
            new_frame = torch.cat([pred_tokens, new_status],
                                  dim=1).unsqueeze(1)
            ct = torch.cat([ct[:, 1:], new_frame], dim=1)
            ca = torch.cat([ca[:, 1:], act_t], dim=1)

            clock.tick(args.fps)

    new_episode()
    idx = 0
    while True:
        status = play_model(idx)
        if status == "quit":
            break
        if status == "retry":
            continue  # same idx, same warmup
        if status == "next":
            idx = (idx + 1) % n_models
            continue
        if status == "restart_cycle":
            idx = 0  # same warmup, back to model 0
            continue
        if status == "new_episode":
            new_episode()
            idx = 0
            continue

    pygame.quit()


if __name__ == "__main__":
    main()
