"""Plot PPO controller training curves from CSV log.

Usage:
    python scripts/plot_ppo_training.py
    python scripts/plot_ppo_training.py --bc-checkpoint checkpoints/controller_bc_best.pt
    python scripts/plot_ppo_training.py --log checkpoints/controller_ppo_log.csv
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def moving_average(x, w):
    """Compute moving average, returning same-length array (NaN-padded start)."""
    if len(x) < w:
        return x
    cumsum = np.nancumsum(x)
    ma = np.full_like(x, np.nan)
    ma[w - 1:] = (cumsum[w - 1:] - np.concatenate([[0], cumsum[:-w]])) / w
    return ma


def eval_bc_baseline(args):
    """Load BC controller + world model and run eval on fixed contexts."""
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from deepdash.world_model import WorldModel
    from deepdash.controller import CNNPolicy

    # Reuse evaluate_fixed from training script
    from train_controller_ppo import (
        evaluate_fixed, load_episodes, sample_contexts)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating BC baseline on {device}...")

    model = WorldModel(
        vocab_size=args.vocab_size, embed_dim=args.embed_dim,
        n_heads=args.n_heads, n_layers=args.n_layers,
        context_frames=args.context_frames, dropout=args.dropout,
        tokens_per_frame=args.tokens_per_frame,
        adaln=getattr(args, 'adaln', False),
        fsq_dim=len(args.levels) if getattr(args, 'levels', None) else None,
    ).to(device)
    state = torch.load(args.transformer_checkpoint, map_location=device,
                       weights_only=True)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    controller = CNNPolicy(
        vocab_size=args.vocab_size,
        grid_size=int(args.tokens_per_frame ** 0.5),
        token_embed_dim=args.token_embed_dim,
        h_dim=args.embed_dim,
    ).to(device)
    state = torch.load(args.bc_checkpoint, map_location=device,
                       weights_only=True)
    controller.load_state_dict(state)
    controller.eval()

    episodes = load_episodes(args.episodes_dir, args.context_frames)
    expert_eps = load_episodes(args.expert_episodes_dir, args.context_frames)
    episodes.extend(expert_eps)

    rng = np.random.default_rng(args.seed)
    eval_tokens, eval_actions = sample_contexts(
        episodes, args.n_eval_episodes, args.context_frames, rng)

    with torch.no_grad():
        surv, jr = evaluate_fixed(
            model, controller, eval_tokens, eval_actions,
            args.max_dream_steps, args.death_threshold, device)

    print(f"BC baseline: survival={surv:.2f}, jump_ratio={jr:.2f}")
    return surv, jr


def main():
    parser = argparse.ArgumentParser(description="Plot PPO training curves")
    parser.add_argument("--log", default="checkpoints/controller_ppo_log.csv")
    parser.add_argument("--output", default="plots/controller_ppo.png",
                        help="Output path for plot")
    parser.add_argument("--bc-checkpoint", default=None,
                        help="BC controller checkpoint to evaluate as baseline")
    # Model/eval args (only used with --bc-checkpoint)
    parser.add_argument("--transformer-checkpoint",
                        default="checkpoints/transformer_best.pt")
    parser.add_argument("--episodes-dir", default="data/death_episodes")
    parser.add_argument("--expert-episodes-dir", default="data/expert_episodes")
    parser.add_argument("--config", default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--tokens-per-frame", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--token-embed-dim", type=int, default=None)
    parser.add_argument("--context-frames", type=int, default=None)
    parser.add_argument("--max-dream-steps", type=int, default=None)
    parser.add_argument("--death-threshold", type=float, default=None)
    parser.add_argument("--n-eval-episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from deepdash.config import apply_config
    apply_config(args, section="controller_ppo")

    df = pd.read_csv(args.log)

    # Get BC baseline: from CSV iteration 0, or by running eval
    eval_all = df[pd.to_numeric(df["eval_survival"], errors="coerce").notna()].copy()
    eval_all["eval_survival"] = eval_all["eval_survival"].astype(float)
    eval_all["jump_ratio"] = eval_all["jump_ratio"].astype(float)

    bc_row = eval_all[eval_all["iteration"] == 0]
    bc_surv = bc_row["eval_survival"].values[0] if len(bc_row) else None
    bc_jr = bc_row["jump_ratio"].values[0] if len(bc_row) else None

    if bc_surv is None and args.bc_checkpoint:
        bc_surv, bc_jr = eval_bc_baseline(args)

    eval_df = eval_all[eval_all["iteration"] > 0]
    train_df = df[df["iteration"] > 0]
    iters = train_df["iteration"].values

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle("PPO Controller Training (BC-pretrained)", fontsize=14, fontweight="bold")

    # Eval Survival
    ax = axes[0, 0]
    ax.plot(eval_df["iteration"], eval_df["eval_survival"],
            color="royalblue", alpha=0.3, linewidth=0.5)
    ax.plot(eval_df["iteration"],
            moving_average(eval_df["eval_survival"].values, 50),
            color="royalblue", linewidth=2, label="MA-50")
    if bc_surv is not None:
        ax.axhline(bc_surv, color="red", linestyle="--", linewidth=1.5,
                   label=f"BC baseline ({bc_surv:.1f})")
    ax.set_title("Eval Survival")
    ax.set_ylabel("Steps")
    ax.legend()
    ax.grid(alpha=0.2)

    # Train Survival
    ax = axes[0, 1]
    ax.plot(iters, train_df["mean_survival"], color="royalblue", alpha=0.15, linewidth=0.5)
    ax.plot(iters, moving_average(train_df["mean_survival"].values, 500),
            color="darkblue", linewidth=2, label="MA-500")
    ax.set_title("Train Survival")
    ax.set_ylabel("Steps")
    ax.legend()
    ax.grid(alpha=0.2)

    # Eval Jump Ratio
    ax = axes[1, 0]
    ax.plot(eval_df["iteration"], eval_df["jump_ratio"],
            color="orange", alpha=0.3, linewidth=0.5)
    ax.plot(eval_df["iteration"],
            moving_average(eval_df["jump_ratio"].values, 50),
            color="darkorange", linewidth=2, label="MA-50")
    if bc_jr is not None:
        ax.axhline(bc_jr, color="red", linestyle="--", linewidth=1.5,
                   label=f"BC baseline ({bc_jr:.2f})")
    ax.set_title("Eval Jump Ratio")
    ax.set_ylabel("Jump Ratio")
    ax.legend()
    ax.grid(alpha=0.2)

    # Train Entropy
    ax = axes[1, 1]
    ax.plot(iters, train_df["entropy"], color="green", alpha=0.15, linewidth=0.5)
    ax.plot(iters, moving_average(train_df["entropy"].values, 500),
            color="darkgreen", linewidth=2, label="MA-500")
    ax.set_title("Train Entropy")
    ax.set_ylabel("Entropy")
    ax.legend()
    ax.grid(alpha=0.2)

    # Learning Rate
    ax = axes[2, 0]
    ax.plot(iters, train_df["lr"].astype(float), color="crimson", linewidth=2)
    ax.set_title("Learning Rate")
    ax.set_ylabel("LR")
    ax.set_xlabel("Iteration")
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    ax.grid(alpha=0.2)

    # Loss
    ax = axes[2, 1]
    ax.plot(iters, train_df["loss"], color="purple", alpha=0.15, linewidth=0.5)
    ax.plot(iters, moving_average(train_df["loss"].values, 500),
            color="purple", linewidth=2, label="MA-500")
    ax.set_title("Train Loss")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Iteration")
    ax.legend()
    ax.grid(alpha=0.2)

    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
