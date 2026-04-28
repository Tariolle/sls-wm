"""Plot beta-VAE vs FSQ training curves under identical conditions.

Reads:
  experiments/beta_vae_vs_fsq/fsq_v3deploy_log.csv  (frozen V3-deploy FSQ log)
  checkpoints_beta_vae/beta_vae_log.csv             (current run)

Writes:
  presentation/beta_vae_vs_fsq.png
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_log(path, recon_col="val_recon"):
    epochs, train_recon, val_recon = [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_recon.append(float(row["train_recon"]))
            val_recon.append(float(row[recon_col]))
    return epochs, train_recon, val_recon


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fsq-log", default="experiments/beta_vae_vs_fsq/fsq_v3deploy_log.csv")
    parser.add_argument("--beta-log", default="checkpoints_beta_vae/beta_vae_log.csv")
    parser.add_argument("--out", default="presentation/beta_vae_vs_fsq.png")
    parser.add_argument("--ylog", action="store_true", help="Log scale y-axis")
    args = parser.parse_args()

    e_fsq, tr_fsq, va_fsq = load_log(args.fsq_log)
    e_b, tr_b, va_b = load_log(args.beta_log)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

    ax = axes[0]
    ax.plot(e_b, tr_b, color="#d62728", lw=1.4, label=f"beta-VAE (final {tr_b[-1]:.2f})")
    ax.plot(e_fsq, tr_fsq, color="#1f77b4", lw=1.4, label=f"FSQ-VAE (final {tr_fsq[-1]:.2f})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reconstruction loss (MSE, sum/sample)")
    ax.set_title("Train")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.9)
    if args.ylog:
        ax.set_yscale("log")

    ax = axes[1]
    ax.plot(e_b, va_b, color="#d62728", lw=1.4, label=f"beta-VAE (best {min(va_b):.2f})")
    ax.plot(e_fsq, va_fsq, color="#1f77b4", lw=1.4, label=f"FSQ-VAE (best {min(va_fsq):.2f})")
    ax.set_xlabel("Epoch")
    ax.set_title("Validation")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.9)
    if args.ylog:
        ax.set_yscale("log")

    fig.suptitle("Beta-VAE vs FSQ-VAE under identical V3-deploy training conditions",
                 fontsize=12)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    print(f"Final: beta-VAE val recon {va_b[-1]:.3f}, FSQ val recon {va_fsq[-1]:.3f}")
    print(f"Best:  beta-VAE val recon {min(va_b):.3f}, FSQ val recon {min(va_fsq):.3f}")


if __name__ == "__main__":
    main()
