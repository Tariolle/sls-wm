"""Diagnostic: does the Transformer's death prediction respond to actions?

Runs the same near-death context with all-jump vs all-idle actions
and compares death probabilities at each dream step.

Usage:
    python scripts/diagnose_death_sensitivity.py
    python scripts/diagnose_death_sensitivity.py --visualize
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepdash.world_model import WorldModel


def decode_tokens(tokenizer, tokens, grid_size, device):
    """Decode flat token array to image."""
    idx = torch.from_numpy(tokens.astype(np.int64)).reshape(1, grid_size, grid_size).to(device)
    z_q = tokenizer.fsq.indices_to_codes(idx)
    img = tokenizer.decoder(z_q)[0, 0].cpu().numpy()
    return (img * 255).clip(0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", action="store_true",
                        help="Show decoded dream frames for jump vs idle")
    parser.add_argument("--n-tests", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load tokenizer for visualization
    tokenizer = None
    if args.visualize:
        from deepdash.fsq import FSQVAE
        tokenizer = FSQVAE(levels=[8, 5, 5, 5]).to(device)
        tok_state = torch.load("checkpoints/fsq_best.pt", map_location=device, weights_only=True)
        tok_state = {k.removeprefix("_orig_mod."): v for k, v in tok_state.items()}
        tokenizer.load_state_dict(tok_state)
        tokenizer.eval()

    model = WorldModel(
        vocab_size=1000, embed_dim=384, n_heads=8, n_layers=8,
        context_frames=4, tokens_per_frame=64,
    ).to(device)
    state = torch.load("checkpoints/transformer_best.pt",
                       map_location=device, weights_only=True)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    # Load episodes
    import re
    shift_re = re.compile(r"_s[+-]\d+_[+-]\d+$")
    episodes_dir = Path("data/death_episodes")
    episodes = []
    for ep in sorted(episodes_dir.glob("*")):
        if shift_re.search(ep.name):
            continue
        tp, ap = ep / "tokens.npy", ep / "actions.npy"
        if tp.exists() and ap.exists():
            tokens = np.load(tp).astype(np.int64)
            actions = np.load(ap).astype(np.int64)
            if len(tokens) >= 12:
                episodes.append((tokens, actions, ep.name))

    print(f"Loaded {len(episodes)} episodes")

    K = 4
    max_steps = 16
    rng = np.random.default_rng(args.seed)
    n_tests = args.n_tests

    results = []

    with torch.no_grad():
        for test_idx in range(n_tests):
            ep_idx = rng.integers(len(episodes))
            tokens, actions, name = episodes[ep_idx]
            T = len(tokens)

            # Start 8-16 frames before death
            start = max(0, T - rng.integers(8, 17) - K)
            ctx_tokens = tokens[start:start + K]

            # Add ALIVE status
            status = np.full((K, 1), model.ALIVE_TOKEN, dtype=np.int64)
            ctx_with_status = np.concatenate([ctx_tokens, status], axis=1)

            death_probs_jump = []
            death_probs_idle = []
            frames_jump = []
            frames_idle = []

            for action_label, action_val in [("jump", 1), ("idle", 0)]:
                ctx_t = torch.from_numpy(ctx_with_status.copy()).unsqueeze(0).to(device)
                ctx_a = torch.from_numpy(actions[start:start + K].copy()).unsqueeze(0).to(device)
                action_frames = []

                for step in range(max_steps):
                    pred_tokens, death_prob, h_t = model.predict_next_frame(
                        ctx_t, ctx_a, temperature=0.0, return_hidden=True)

                    dp = death_prob[0].item()
                    pred_np = pred_tokens[0].cpu().numpy()

                    if tokenizer is not None:
                        action_frames.append(decode_tokens(tokenizer, pred_np, 8, device))

                    if action_val == 1:
                        death_probs_jump.append(dp)
                    else:
                        death_probs_idle.append(dp)

                    if dp > 0.5:
                        remaining = max_steps - step - 1
                        if action_val == 1:
                            death_probs_jump.extend([1.0] * remaining)
                        else:
                            death_probs_idle.extend([1.0] * remaining)
                        break

                    chosen_action = torch.tensor([[action_val]], dtype=torch.long, device=device)
                    new_status = torch.full((1, 1), model.ALIVE_TOKEN, dtype=torch.long, device=device)
                    new_frame = torch.cat([pred_tokens, new_status], dim=1).unsqueeze(1)
                    ctx_t = torch.cat([ctx_t[:, 1:], new_frame], dim=1)
                    ctx_a = torch.cat([ctx_a[:, 1:], chosen_action], dim=1)

                if action_val == 1:
                    frames_jump = action_frames
                else:
                    frames_idle = action_frames

            # Visualize: animate jump vs idle side by side
            if args.visualize and frames_jump and frames_idle:
                n_frames = max(len(frames_jump), len(frames_idle))
                fig, (ax_jump, ax_idle) = plt.subplots(1, 2, figsize=(8, 4))
                fig.suptitle(f"{name} (start={start})")

                for f in range(n_frames):
                    ax_jump.clear()
                    ax_idle.clear()

                    if f < len(frames_jump):
                        ax_jump.imshow(frames_jump[f], cmap="gray", vmin=0, vmax=255)
                        dp_j = death_probs_jump[f] if f < len(death_probs_jump) else 1.0
                        color_j = "red" if dp_j > 0.5 else "black"
                        ax_jump.set_title(f"JUMP t+{f+1}  d={dp_j:.3f}", fontsize=10, color=color_j)
                    else:
                        ax_jump.imshow(np.zeros((64, 64), dtype=np.uint8), cmap="gray", vmin=0, vmax=255)
                        ax_jump.set_title("JUMP: DEAD", fontsize=10, color="red")

                    if f < len(frames_idle):
                        ax_idle.imshow(frames_idle[f], cmap="gray", vmin=0, vmax=255)
                        dp_i = death_probs_idle[f] if f < len(death_probs_idle) else 1.0
                        color_i = "red" if dp_i > 0.5 else "black"
                        ax_idle.set_title(f"IDLE t+{f+1}  d={dp_i:.3f}", fontsize=10, color=color_i)
                    else:
                        ax_idle.imshow(np.zeros((64, 64), dtype=np.uint8), cmap="gray", vmin=0, vmax=255)
                        ax_idle.set_title("IDLE: DEAD", fontsize=10, color="red")

                    ax_jump.axis("off")
                    ax_idle.axis("off")
                    plt.pause(1.0 / 7.5)  # ~8 FPS (quarter speed)

                plt.close(fig)

            # Find when each dies
            death_step_jump = next((i for i, p in enumerate(death_probs_jump) if p > 0.5), max_steps)
            death_step_idle = next((i for i, p in enumerate(death_probs_idle) if p > 0.5), max_steps)

            diff = death_step_jump - death_step_idle
            results.append((name, start, T, death_step_jump, death_step_idle, diff))

            print(f"  [{test_idx+1:2d}] {name} (start={start}, T={T}): "
                  f"jump dies at step {death_step_jump}, idle dies at step {death_step_idle}, "
                  f"diff={diff:+d}")

            # Show death probability trajectories for first 5
            if test_idx < 5:
                print(f"       jump probs: {['%.3f'%p for p in death_probs_jump[:10]]}")
                print(f"       idle probs: {['%.3f'%p for p in death_probs_idle[:10]]}")

    # Summary
    diffs = [r[5] for r in results]
    same = sum(1 for d in diffs if d == 0)
    jump_better = sum(1 for d in diffs if d > 0)
    idle_better = sum(1 for d in diffs if d < 0)
    both_survive = sum(1 for r in results if r[3] == max_steps and r[4] == max_steps)

    print(f"\n{'='*50}")
    print(f"Summary ({n_tests} tests):")
    print(f"  Jump survives longer: {jump_better}")
    print(f"  Idle survives longer: {idle_better}")
    print(f"  Same death step:      {same}")
    print(f"  Both survive all {max_steps} steps: {both_survive}")
    print(f"  Mean diff (jump - idle): {np.mean(diffs):+.1f} steps")

    if same == n_tests:
        print("\n!! Death prediction is completely action-invariant.")
        print("   The controller CANNOT influence survival.")
    elif abs(np.mean(diffs)) < 0.5:
        print("\n~  Death prediction is weakly action-sensitive.")
        print("   Jump and idle have similar outcomes on average.")
    else:
        print("\n✓  Death prediction is action-sensitive.")
        print("   Different actions lead to different survival outcomes.")


if __name__ == "__main__":
    main()
