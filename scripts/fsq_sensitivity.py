"""FSQ codebook sensitivity analysis.

Measures how much reconstruction degrades when perturbing quantized codes
by 1 or 2 steps in various FSQ dimensions. Used to calibrate the structured
label smoothing sigma for the transformer.

Usage:
    python scripts/fsq_sensitivity.py
    python scripts/fsq_sensitivity.py --checkpoint checkpoints/fsq_best.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepdash.fsq import FSQVAE
from deepdash.data_split import get_val_episodes, is_val_episode


def load_val_frames(episodes_dir, expert_episodes_dir, max_frames=10000, seed=42):
    """Load a sample of validation frames."""
    val_set = get_val_episodes(episodes_dir, expert_episodes_dir)
    all_frames = []
    for ep_dir in [episodes_dir, expert_episodes_dir]:
        p = Path(ep_dir)
        if not p.exists():
            continue
        for ep in sorted(p.glob("*")):
            fp = ep / "frames.npy"
            if not fp.exists():
                continue
            if is_val_episode(ep.name, val_set):
                all_frames.append(np.load(fp))
    data = np.concatenate(all_frames, axis=0)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(data), size=min(max_frames, len(data)), replace=False)
    return data[indices]


def main():
    parser = argparse.ArgumentParser(description="FSQ codebook sensitivity analysis")
    parser.add_argument("--checkpoint", default="checkpoints/fsq_best.pt")
    parser.add_argument("--levels", type=int, nargs="+", default=[8, 5, 5, 5])
    parser.add_argument("--episodes-dir", default="data/death_episodes")
    parser.add_argument("--expert-episodes-dir", default="data/expert_episodes")
    parser.add_argument("--max-frames", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = FSQVAE(levels=args.levels)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Load data
    print("Loading validation frames...")
    frames = load_val_frames(
        args.episodes_dir, args.expert_episodes_dir, args.max_frames
    )
    print(f"Loaded {len(frames)} frames")

    # Encode all frames to get quantized codes and pre-quantization latents
    all_z_q = []  # (B, D, 8, 8) quantized codes
    all_z_e = []  # (B, D, 8, 8) pre-quantization (continuous)
    all_imgs = []  # (B, 1, 64, 64) original images

    with torch.no_grad():
        for i in range(0, len(frames), args.batch_size):
            batch = torch.from_numpy(
                frames[i:i + args.batch_size].astype(np.float32) / 255.0
            ).unsqueeze(1).to(device)
            z_e = model.encoder(batch)
            z_q, _ = model.fsq(z_e)
            # Get bounded continuous values (before rounding)
            fsq = model.fsq
            half = fsq.half_levels.view(1, -1, 1, 1)
            z_bounded = torch.tanh(z_e) * half
            all_z_e.append(z_bounded.cpu())
            all_z_q.append(z_q.cpu())
            all_imgs.append(batch.cpu())

    all_z_q = torch.cat(all_z_q)  # (N, D, 8, 8)
    all_z_e = torch.cat(all_z_e)  # (N, D, 8, 8)
    all_imgs = torch.cat(all_imgs)  # (N, 1, 64, 64)

    n_dims = len(args.levels)
    half_levels = torch.tensor([L // 2 for L in args.levels], dtype=torch.float32)

    # Baseline reconstruction MSE
    with torch.no_grad():
        baseline_mse = 0.0
        for i in range(0, len(all_z_q), args.batch_size):
            z_q = all_z_q[i:i + args.batch_size].to(device)
            recon = model.decoder(z_q)
            orig = all_imgs[i:i + args.batch_size].to(device)
            baseline_mse += ((recon - orig) ** 2).sum().item()
        baseline_mse /= len(all_z_q)

    print(f"\nBaseline reconstruction MSE: {baseline_mse:.6f}")
    print()

    # Define perturbation experiments: all non-empty subsets of dims, +1 step
    from itertools import combinations

    experiments = []

    # Single dim +1 and +2
    for d in range(n_dims):
        for step in [1, 2]:
            experiments.append((f"dim{d}(L={args.levels[d]}) +{step}", [(d, step)]))

    # All multi-dim combos at +1 step (2-dim, 3-dim, 4-dim)
    for k in range(2, n_dims + 1):
        for combo in combinations(range(n_dims), k):
            name = "+".join(f"d{d}" for d in combo) + " +1"
            experiments.append((name, [(d, 1) for d in combo]))

    def run_perturbation(all_z_q, all_imgs, perturbations):
        """Apply perturbations and measure MSE."""
        perturbed_z_q = all_z_q.clone()
        for d, step in perturbations:
            half = half_levels[d]
            max_val = half - (1.0 - args.levels[d] % 2)
            min_val = -half
            vals = perturbed_z_q[:, d]
            can_up = vals + step <= max_val
            can_down = vals - step >= min_val
            rand_up = torch.rand_like(vals) > 0.5
            go_up = torch.where(can_up & can_down, rand_up,
                                torch.where(can_up, torch.ones_like(rand_up),
                                            torch.zeros_like(rand_up)))
            can_any = can_up | can_down
            delta = torch.where(go_up, step, -step) * can_any.float()
            perturbed_z_q[:, d] = vals + delta

        with torch.no_grad():
            mse = 0.0
            for i in range(0, len(perturbed_z_q), args.batch_size):
                z_q = perturbed_z_q[i:i + args.batch_size].to(device)
                recon = model.decoder(z_q)
                orig = all_imgs[i:i + args.batch_size].to(device)
                mse += ((recon - orig) ** 2).sum().item()
            mse /= len(perturbed_z_q)
        return mse

    print(f"{'Perturbation':<30} {'MSE':>10} {'Delta':>10} {'Ratio':>8}")
    print("-" * 62)

    results = {}
    for name, perturbations in experiments:
        mse = run_perturbation(all_z_q, all_imgs, perturbations)
        delta = mse - baseline_mse
        ratio = mse / baseline_mse
        results[name] = (mse, delta, ratio)
        print(f"{name:<30} {mse:10.6f} {delta:+10.6f} {ratio:8.2f}x")

    # Compounding analysis: compare measured multi-dim ratios against
    # additive (sum of deltas) and multiplicative (product of ratios) predictions
    print()
    print("=" * 72)
    print("Compounding analysis (+1 step combos)")
    print(f"{'Combo':<20} {'Measured':>8} {'Additive':>10} {'Multiplic':>10} {'Best fit':>10}")
    print("-" * 62)

    # Collect single-dim +1 results
    single = {}
    for d in range(n_dims):
        key = f"dim{d}(L={args.levels[d]}) +1"
        single[d] = results[key]  # (mse, delta, ratio)

    add_errors = []
    mul_errors = []

    for k in range(2, n_dims + 1):
        for combo in combinations(range(n_dims), k):
            key = "+".join(f"d{d}" for d in combo) + " +1"
            measured_ratio = results[key][2]

            # Additive: sum of deltas / baseline + 1
            add_pred = sum(single[d][1] for d in combo) / baseline_mse + 1.0
            # Multiplicative: product of ratios
            mul_pred = 1.0
            for d in combo:
                mul_pred *= single[d][2]

            add_err = abs(measured_ratio - add_pred) / measured_ratio
            mul_err = abs(measured_ratio - mul_pred) / measured_ratio
            add_errors.append(add_err)
            mul_errors.append(mul_err)

            best = "additive" if add_err < mul_err else "multiplic"
            print(f"{key:<20} {measured_ratio:8.2f}x {add_pred:9.2f}x {mul_pred:9.2f}x {best:>10}")

    print()
    avg_add = sum(add_errors) / len(add_errors) * 100
    avg_mul = sum(mul_errors) / len(mul_errors) * 100
    print(f"Mean relative error:  additive={avg_add:.1f}%  multiplicative={avg_mul:.1f}%")
    if avg_add < avg_mul:
        print("=> Perturbations compound ADDITIVELY (use L1/Manhattan distance)")
    else:
        print("=> Perturbations compound MULTIPLICATIVELY (use L2/Euclidean distance)")

    # Sigma sweep: find optimal sigma for weighted L2 Gaussian kernel
    # Ground truth: 1/ratio = reconstruction similarity (higher = more similar)
    # Gaussian prediction: exp(-weighted_dist^2 / (2*sigma^2))
    # We want the sigma where kernel weights best correlate with 1/ratio
    print()
    print("=" * 72)
    print("Sigma sweep (weighted L2 distance)")

    # Measured sensitivities as dim weights (normalized so mean=1)
    single_ratios = [single[d][2] for d in range(n_dims)]
    mean_ratio = sum(single_ratios) / n_dims
    dim_weights = [r / mean_ratio for r in single_ratios]
    print(f"Dim weights (from sensitivity): {[f'{w:.3f}' for w in dim_weights]}")

    # Collect all +1 step combos with their measured ratios and weighted L2 distances
    combo_data = []
    for k in range(1, n_dims + 1):
        for combo in combinations(range(n_dims), k):
            if k == 1:
                key = f"dim{combo[0]}(L={args.levels[combo[0]]}) +1"
            else:
                key = "+".join(f"d{d}" for d in combo) + " +1"
            measured_ratio = results[key][2]
            # Weighted L2 distance: sqrt(sum(w_d^2)) for +1 step in each dim
            w_sq_dist = sum(dim_weights[d] ** 2 for d in combo)
            similarity = 1.0 / measured_ratio
            combo_data.append((key, w_sq_dist, similarity, measured_ratio))

    print()
    print(f"{'Sigma':>6}  {'Corr':>7}  {'MSE':>10}  Notes")
    print("-" * 42)

    best_sigma = 0.0
    best_mse = float("inf")
    for sigma_test in [x * 0.1 for x in range(1, 31)]:
        # Predict similarity as Gaussian kernel value
        predictions = []
        targets = []
        for _, w_sq_dist, similarity, _ in combo_data:
            pred = np.exp(-w_sq_dist / (2 * sigma_test ** 2))
            predictions.append(pred)
            targets.append(similarity)
        # Normalize both to [0, 1] for fair comparison
        pred_arr = np.array(predictions)
        tgt_arr = np.array(targets)
        # Pearson correlation
        if pred_arr.std() > 1e-8:
            corr = np.corrcoef(pred_arr, tgt_arr)[0, 1]
        else:
            corr = 0.0
        # MSE between normalized predictions and targets
        pred_norm = pred_arr / pred_arr.sum()
        tgt_norm = tgt_arr / tgt_arr.sum()
        mse_fit = ((pred_norm - tgt_norm) ** 2).mean()
        marker = ""
        if mse_fit < best_mse:
            best_mse = mse_fit
            best_sigma = sigma_test
            marker = " <-- best"
        print(f"{sigma_test:6.1f}  {corr:7.4f}  {mse_fit:10.6f}{marker}")

    print()
    print(f"Optimal sigma: {best_sigma:.1f}")
    print(f"Recommended: --fsq-sigma {best_sigma:.1f} "
          f"--fsq-dim-weights {' '.join(f'{w:.2f}' for w in dim_weights)}")

    # Quantization margin analysis: estimate natural label smoothing epsilon
    # The FSQ rounds z_bounded to the nearest integer. The "margin" is how
    # close z_bounded was to rounding differently. Small margin = ambiguous token.
    print()
    print("=" * 72)
    print("Quantization margin analysis (natural label smoothing epsilon)")

    # z_bounded is continuous, z_q is rounded. Margin = |z_bounded - z_q|
    # per dimension per position. Range is [0, 0.5] (0 = exactly on boundary,
    # 0.5 = dead center of a level).
    margin = (all_z_e - all_z_q).abs()  # (N, D, 8, 8)

    # A token is "ambiguous" if ANY dimension has margin < threshold
    # (it could have rounded to a different code)
    print(f"\n{'Threshold':<12} {'Ambiguous%':>10} {'Per-dim margins':>40}")
    print("-" * 65)

    per_dim_means = [margin[:, d].mean().item() for d in range(n_dims)]
    print(f"{'(mean)':>12} {'':>10} "
          f"{' '.join(f'd{d}={m:.3f}' for d, m in enumerate(per_dim_means)):>40}")

    for threshold in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        # Token is ambiguous if any dim has margin below threshold
        close_any_dim = (margin < threshold).any(dim=1)  # (N, 8, 8)
        ambiguous_frac = close_any_dim.float().mean().item()
        # Per-dim breakdown
        per_dim = [(margin[:, d] < threshold).float().mean().item() for d in range(n_dims)]
        per_dim_str = " ".join(f"d{d}={v:.1%}" for d, v in enumerate(per_dim))
        print(f"{threshold:<12.2f} {ambiguous_frac:>9.1%} {per_dim_str:>40}")

    # Recommended epsilon: fraction of tokens within margin 0.25
    # (halfway to the next level = coin flip territory)
    ambig_025 = (margin < 0.25).any(dim=1).float().mean().item()
    print(f"\nRecommended --label-smoothing {ambig_025:.2f}")
    print(f"  (fraction of tokens with any dim within 0.25 of a boundary)")

    # Full recommendation
    print()
    print("=" * 72)
    print("Full recommendation:")
    print(f"  --fsq-sigma {best_sigma:.1f} "
          f"--fsq-dim-weights {' '.join(f'{w:.2f}' for w in dim_weights)} "
          f"--label-smoothing {ambig_025:.2f}")


if __name__ == "__main__":
    main()
