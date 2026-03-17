"""Plot transformer training curves from CSV log.

Usage:
    python scripts/plot_transformer_training.py
    python scripts/plot_transformer_training.py --log checkpoints/transformer_log.csv
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="checkpoints/transformer_log.csv")
    parser.add_argument("--output", default="plots/transformer_training.png",
                        help="Output path for plot")
    args = parser.parse_args()

    d = load_log(args.log)
    epochs = d["epoch"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Transformer Training", fontsize=14, fontweight="bold")

    # 1. Total loss (train vs val)
    ax = axes[0, 0]
    ax.plot(epochs, d["train_total"], label="Train", alpha=0.8)
    ax.plot(epochs, d["val_total"], label="Val", alpha=0.8)
    ax.set_title("Total Loss")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Token loss (train vs val)
    ax = axes[0, 1]
    ax.plot(epochs, d["train_loss"], label="Train", alpha=0.8)
    ax.plot(epochs, d["val_loss"], label="Val", alpha=0.8)
    ax.set_title("Token Loss")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Token accuracy (train vs val)
    ax = axes[0, 2]
    ax.plot(epochs, d["train_acc"], label="Train", alpha=0.8)
    ax.plot(epochs, d["val_acc"], label="Val", alpha=0.8)
    ax.set_title("Token Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Death metrics (precision, recall, F1) - val only
    ax = axes[1, 0]
    # Handle both old (death_acc) and new (death_prec/rec/f1) formats
    if "val_death_prec" in d:
        ax.plot(epochs, d["val_death_prec"], label="Precision", alpha=0.8)
        ax.plot(epochs, d["val_death_rec"], label="Recall", alpha=0.8)
        ax.plot(epochs, d["val_death_f1"], label="F1", alpha=0.8)
        ax.set_title("Val Death Prediction")
    elif "val_death_acc" in d:
        ax.plot(epochs, d["val_death_acc"], label="Death Acc", alpha=0.8)
        ax.set_title("Val Death Accuracy")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. CPC loss (train vs val)
    ax = axes[1, 1]
    ax.plot(epochs, d["train_cpc"], label="Train", alpha=0.8)
    ax.plot(epochs, d["val_cpc"], label="Val", alpha=0.8)
    ax.set_title("CPC Loss")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Overfitting gap + LR
    ax = axes[1, 2]
    ax.plot(epochs, d["gap"], label="Gap (val-train)", alpha=0.8, color="red")
    ax.set_ylabel("Gap", color="red")
    ax.tick_params(axis="y", labelcolor="red")
    ax.set_title("Overfitting Gap & LR")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(epochs, d["lr"], label="LR", alpha=0.6, color="gray",
             linestyle="--")
    ax2.set_ylabel("LR", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")

    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
