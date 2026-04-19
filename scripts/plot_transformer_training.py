"""Plot transformer training curves from one or more CSV logs.

Compare multiple runs by passing multiple logs. Labels default to the parent
directory name of each log (e.g. `checkpoints_v5_fsq/transformer_log.csv`
is labelled `v5_baseline`).

Usage:
    # single run
    python scripts/plot_transformer_training.py --log checkpoints/transformer_log.csv

    # compare two runs
    python scripts/plot_transformer_training.py \
        --log checkpoints_v5_fsq/transformer_log.csv \
              checkpoints_v5_dimweights/transformer_log.csv \
        --output plots/v5_baseline_vs_dimweights.png

    # with explicit labels
    python scripts/plot_transformer_training.py \
        --log run_a/transformer_log.csv run_b/transformer_log.csv \
        --label "baseline" "dimweights"
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_log(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    data = {}
    for key in rows[0]:
        try:
            data[key] = np.array([float(r[key]) for r in rows])
        except ValueError:
            data[key] = [r[key] for r in rows]
    return data


def _default_label(log_path):
    """Use the log's parent dir name, stripped of a 'checkpoints_' prefix."""
    p = Path(log_path).resolve().parent.name
    return p[len("checkpoints_"):] if p.startswith("checkpoints_") else p


def _plot_train_val(ax, runs, train_key, val_key, title, ylabel=None,
                    ylog=False):
    """Plot train (dashed) + val (solid) from a series of runs on one axis."""
    for color, (label, d) in zip(_palette(len(runs)), runs):
        if train_key in d:
            ax.plot(d["epoch"], d[train_key],
                    color=color, linestyle="--", alpha=0.6,
                    label=f"{label} train")
        if val_key in d:
            ax.plot(d["epoch"], d[val_key],
                    color=color, linestyle="-", alpha=0.9,
                    label=f"{label} val")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    if ylabel:
        ax.set_ylabel(ylabel)
    if ylog:
        ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def _plot_val_only(ax, runs, key, title, ylabel=None):
    for color, (label, d) in zip(_palette(len(runs)), runs):
        if key in d:
            ax.plot(d["epoch"], d[key], color=color, alpha=0.9, label=label)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def _plot_death_pr(ax, runs, title="Val Death Precision / Recall"):
    """Precision (dashed) + Recall (solid) per run, val only."""
    for color, (label, d) in zip(_palette(len(runs)), runs):
        if "val_death_prec" in d:
            ax.plot(d["epoch"], d["val_death_prec"],
                    color=color, linestyle="--", alpha=0.6,
                    label=f"{label} P")
        if "val_death_rec" in d:
            ax.plot(d["epoch"], d["val_death_rec"],
                    color=color, linestyle="-", alpha=0.9,
                    label=f"{label} R")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def _palette(n):
    """Reasonable colors for up to ~8 runs; cycles through tab10 afterwards."""
    base = plt.get_cmap("tab10").colors
    return [base[i % len(base)] for i in range(n)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", nargs="+", required=True,
                        help="One or more transformer log CSVs to plot/compare")
    parser.add_argument("--label", nargs="+", default=None,
                        help="Optional labels matching --log (default: parent dir name)")
    parser.add_argument("--output", default="plots/transformer_training.png",
                        help="Output path for plot (use '' or 'show' to display instead of saving)")
    parser.add_argument("--title", default=None,
                        help="Figure title (default: 'Transformer Training' "
                             "or 'Transformer Training Comparison' for >1 run)")
    args = parser.parse_args()

    if args.label is not None and len(args.label) != len(args.log):
        raise SystemExit(
            f"--label count ({len(args.label)}) must match --log count ({len(args.log)})"
        )

    labels = args.label or [_default_label(p) for p in args.log]
    runs = [(lbl, load_log(p)) for lbl, p in zip(labels, args.log)]
    multi = len(runs) > 1

    fig, axes = plt.subplots(3, 3, figsize=(18, 13))
    title = args.title or (
        "Transformer Training Comparison" if multi else "Transformer Training")
    fig.suptitle(title, fontsize=14, fontweight="bold")

    _plot_train_val(axes[0, 0], runs, "train_total", "val_total",
                    "Total Loss", ylabel="Loss")
    _plot_train_val(axes[0, 1], runs, "train_loss", "val_loss",
                    "Token Loss", ylabel="CE")
    _plot_train_val(axes[0, 2], runs, "train_acc", "val_acc",
                    "Token Accuracy", ylabel="Accuracy")

    _plot_train_val(axes[1, 0], runs, "train_cpc", "val_cpc",
                    "CPC Loss", ylabel="CPC")
    _plot_train_val(axes[1, 1], runs, "train_death_f1", "val_death_f1",
                    "Death F1")
    _plot_death_pr(axes[1, 2], runs)

    # Overfitting gap (val - train)
    ax = axes[2, 0]
    for color, (label, d) in zip(_palette(len(runs)), runs):
        if "gap" in d:
            ax.plot(d["epoch"], d["gap"], color=color, alpha=0.9, label=label)
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_title("Overfitting Gap (val - train total)")
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # LR (log scale)
    _plot_val_only(axes[2, 1], runs, "lr", "Learning Rate", ylabel="LR")
    axes[2, 1].set_yscale("log")

    # Cumulative wall-clock time (hours)
    ax = axes[2, 2]
    for color, (label, d) in zip(_palette(len(runs)), runs):
        if "time_s" in d:
            cum_h = np.cumsum(d["time_s"]) / 3600.0
            ax.plot(d["epoch"], cum_h, color=color, alpha=0.9, label=label)
    ax.set_title("Cumulative Wall Clock")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Hours")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if args.output and args.output.lower() not in ("", "show", "none"):
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved to {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
