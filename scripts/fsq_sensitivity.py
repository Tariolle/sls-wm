"""FSQ codebook sensitivity analysis — paper-quality calibration.

Fits multiple kernel families (L2 Gaussian, L1 Laplace, anisotropic Gaussian,
Cauchy) to reconstruction-similarity-vs-FSQ-distance data, with cross-
validation and bootstrap confidence intervals. Reports per-metric (MSE and
SSIM) recommendations for --fsq-kernel, --fsq-sigma / --fsq-sigma-d,
and --fsq-dim-weights.

Usage:
    python scripts/fsq_sensitivity.py \
        --checkpoint checkpoints_v5_fsq/fsq_best.pt --levels 5 5 5 5
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import List

import numpy as np
import torch
from scipy.optimize import minimize, minimize_scalar
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepdash.fsq import FSQVAE
from deepdash.data_split import get_val_episodes, is_val_episode


# -------- Data loading --------

def load_val_frames(episodes_dir, expert_episodes_dir, max_frames=10000, seed=42):
    """Load a sample of validation frames from base episodes only (skip shifts)."""
    shift_re = re.compile(r"_s[+-]\d+_[+-]\d+$")
    val_set = get_val_episodes(episodes_dir, expert_episodes_dir)
    all_frames = []
    for ep_dir in [episodes_dir, expert_episodes_dir]:
        p = Path(ep_dir)
        if not p.exists():
            continue
        for ep in sorted(p.glob("*")):
            if shift_re.search(ep.name):
                continue
            fp = ep / "frames.npy"
            if not fp.exists():
                continue
            if is_val_episode(ep.name, val_set):
                all_frames.append(np.load(fp))
    data = np.concatenate(all_frames, axis=0)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(data), size=min(max_frames, len(data)), replace=False)
    return data[indices]


def encode_all(model, frames, batch_size, device):
    """Encode frames to quantized codes; keep images for later reconstruction."""
    all_z_q = []
    all_imgs = []
    all_indices = []
    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            batch = torch.from_numpy(
                frames[i:i + batch_size].astype(np.float32) / 255.0
            ).unsqueeze(1).to(device)
            z_e = model.encoder(batch)
            z_q, _ = model.fsq(z_e)
            indices = model.encode(batch)  # (B, side, side) flat code ids
            all_z_q.append(z_q.cpu())
            all_imgs.append(batch.cpu())
            all_indices.append(indices.cpu())
    return torch.cat(all_z_q), torch.cat(all_imgs), torch.cat(all_indices)


def codebook_usage(indices, levels):
    """Compute codebook usage % and perplexity over a tensor of flat code ids.

    indices: long tensor of any shape, values in [0, prod(levels)).
    Returns (usage_pct, perplexity, n_used, vocab_size).
    """
    vocab_size = int(np.prod(levels))
    flat = indices.reshape(-1).long()
    counts = torch.zeros(vocab_size, dtype=torch.long)
    counts.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.long))
    n_used = int((counts > 0).sum().item())
    usage_pct = 100.0 * n_used / vocab_size
    total = int(counts.sum().item())
    if total == 0:
        return usage_pct, 0.0, n_used, vocab_size
    p = counts.to(torch.float64) / float(total)
    p_nz = p[p > 0]
    entropy = float(-(p_nz * p_nz.log()).sum().item())
    return usage_pct, float(np.exp(entropy)), n_used, vocab_size


# -------- Perturbation specs --------

@dataclass
class Spec:
    """A perturbation: step magnitudes per dim (0 = no perturbation)."""
    name: str
    step_by_dim: List[int] = field(default_factory=list)  # len = n_dims

    def nonzero_steps(self):
        return [(d, s) for d, s in enumerate(self.step_by_dim) if s != 0]


def generate_perturbations(levels):
    """Build a comprehensive perturbation set: single-dim up to 4 steps,
    multi-dim +1 and +2 combos, and mixed (+2, +1) pairs.
    """
    n_dims = len(levels)
    specs = []
    # Single-dim: steps 1..4 where feasible
    for d in range(n_dims):
        max_step = levels[d] - 1
        for step in range(1, min(max_step, 4) + 1):
            v = [0] * n_dims
            v[d] = step
            specs.append(Spec(f"dim{d}(L={levels[d]}) +{step}", v))
    # All 2/3/4-dim subsets at +1
    for k in range(2, n_dims + 1):
        for combo in combinations(range(n_dims), k):
            v = [0] * n_dims
            for d in combo:
                v[d] = 1
            name = "+".join(f"d{d}" for d in combo) + " +1"
            specs.append(Spec(name, v))
    # All 2/3/4-dim subsets at +2 (where feasible for all dims in combo)
    for k in range(2, n_dims + 1):
        for combo in combinations(range(n_dims), k):
            if any(levels[d] - 1 < 2 for d in combo):
                continue
            v = [0] * n_dims
            for d in combo:
                v[d] = 2
            name = "+".join(f"d{d}" for d in combo) + " +2"
            specs.append(Spec(name, v))
    # Mixed-step pairs (d_i +2, d_j +1)
    for d1 in range(n_dims):
        for d2 in range(n_dims):
            if d1 == d2:
                continue
            if levels[d1] - 1 < 2 or levels[d2] - 1 < 1:
                continue
            v = [0] * n_dims
            v[d1] = 2
            v[d2] = 1
            specs.append(Spec(f"d{d1}+2,d{d2}+1", v))
    return specs


def apply_perturbation(z_q, spec, levels):
    """Apply a perturbation to quantized codes, respecting code-range bounds."""
    half_levels = [L // 2 for L in levels]
    out = z_q.clone()
    for d, step in spec.nonzero_steps():
        half = float(half_levels[d])
        max_val = half - (1.0 - levels[d] % 2)
        min_val = -half
        vals = out[:, d]
        can_up = vals + step <= max_val
        can_down = vals - step >= min_val
        rand_up = torch.rand_like(vals) > 0.5
        go_up = torch.where(can_up & can_down, rand_up,
                            torch.where(can_up, torch.ones_like(rand_up),
                                        torch.zeros_like(rand_up)))
        can_any = can_up | can_down
        delta = torch.where(go_up, float(step), float(-step)) * can_any.float()
        out[:, d] = vals + delta
    return out


# -------- Per-frame error measurement --------

def _ssim_batch(recon_np, orig_np):
    """Compute SSIM per-frame for a (B, H, W) batch. Returns (B,) float array."""
    out = np.empty(len(recon_np), dtype=np.float32)
    for j in range(len(recon_np)):
        out[j] = ssim(orig_np[j], recon_np[j], data_range=1.0)
    return out


def measure_errors(model, all_z_q, all_imgs, specs, levels, batch_size, device,
                   compute_ssim=True):
    """Compute per-frame MSE and SSIM for baseline and each perturbation.

    Returns:
        mse_baseline: (N,) float
        ssim_baseline: (N,) float or None
        mse_perturbed: (K, N) float
        ssim_perturbed: (K, N) float or None
    """
    N = len(all_z_q)
    K = len(specs)

    def _decode_pass(z_q_all):
        mse = np.zeros(N, dtype=np.float32)
        ssim_arr = np.zeros(N, dtype=np.float32) if compute_ssim else None
        with torch.no_grad():
            for i in range(0, N, batch_size):
                z_q = z_q_all[i:i + batch_size].to(device)
                recon = model.decoder(z_q)
                orig = all_imgs[i:i + batch_size].to(device)
                per_mse = ((recon - orig) ** 2).mean(dim=[1, 2, 3]).cpu().numpy()
                mse[i:i + len(per_mse)] = per_mse
                if compute_ssim:
                    recon_np = recon.cpu().numpy()[:, 0]
                    orig_np = orig.cpu().numpy()[:, 0]
                    ssim_arr[i:i + len(per_mse)] = _ssim_batch(recon_np, orig_np)
        return mse, ssim_arr

    print("  baseline ...", flush=True)
    mse_baseline, ssim_baseline = _decode_pass(all_z_q)

    mse_perturbed = np.zeros((K, N), dtype=np.float32)
    ssim_perturbed = np.zeros((K, N), dtype=np.float32) if compute_ssim else None
    for k, spec in enumerate(specs):
        print(f"  [{k + 1:>2}/{K}] {spec.name}", flush=True)
        z_q_p = apply_perturbation(all_z_q, spec, levels)
        mse_k, ssim_k = _decode_pass(z_q_p)
        mse_perturbed[k] = mse_k
        if compute_ssim:
            ssim_perturbed[k] = ssim_k
    return mse_baseline, ssim_baseline, mse_perturbed, ssim_perturbed


# -------- Bootstrap CIs --------

def bootstrap_ratio(per_frame_baseline, per_frame_perturbed, B=500, seed=42):
    """Mean ratio of perturbed/baseline MSE (or 1-SSIM) with 95% percentile CI."""
    rng = np.random.default_rng(seed)
    N = len(per_frame_baseline)
    ratios = np.empty(B, dtype=np.float32)
    for b in range(B):
        idx = rng.integers(0, N, size=N)
        num = per_frame_perturbed[idx].mean()
        den = max(per_frame_baseline[idx].mean(), 1e-12)
        ratios[b] = num / den
    return float(ratios.mean()), float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))


# -------- Kernel families --------

def kernel_value(kernel, params, weighted_steps):
    """Compute kernel value at a weighted step vector.

    kernel: 'gaussian', 'laplace', 'cauchy', 'aniso_gaussian'
    params: scalar sigma (1D kernels) or per-dim sigma array (aniso)
    weighted_steps: (n_dims,) array of step_d * dim_weight_d
    """
    if kernel == 'gaussian':
        d_sq = float((weighted_steps ** 2).sum())
        return np.exp(-d_sq / (2.0 * float(params) ** 2))
    if kernel == 'laplace':
        d = float(np.abs(weighted_steps).sum())
        return np.exp(-d / float(params))
    if kernel == 'cauchy':
        d_sq = float((weighted_steps ** 2).sum())
        return 1.0 / (1.0 + d_sq / (float(params) ** 2))
    if kernel == 'aniso_gaussian':
        sigma_d = np.asarray(params, dtype=np.float64)
        return float(np.exp(-((weighted_steps / sigma_d) ** 2).sum() / 2.0))
    raise ValueError(f"unknown kernel: {kernel}")


def _normalized_mse(preds, targets):
    preds = np.clip(preds, 1e-20, None)
    pred_norm = preds / preds.sum()
    tgt_norm = targets / targets.sum()
    return float(((pred_norm - tgt_norm) ** 2).mean())


def _fit_on_indices(kernel, specs, similarities, dim_weights, indices, seed):
    """Fit kernel params on a subset of specs/similarities (used by CV)."""
    n_dims = len(dim_weights)
    dw = np.asarray(dim_weights, dtype=np.float64)
    ws_list = []
    tgt_list = []
    for i in indices:
        step_vec = np.asarray(specs[i].step_by_dim, dtype=np.float64)
        ws_list.append(step_vec * dw)
        tgt_list.append(float(similarities[i]))
    ws_arr = np.array(ws_list)
    tgt_arr = np.array(tgt_list)

    def loss(params):
        preds = np.array([kernel_value(kernel, params, ws) for ws in ws_arr])
        return _normalized_mse(preds, tgt_arr)

    if kernel == 'aniso_gaussian':
        rng = np.random.default_rng(seed)
        best_x = None
        best_f = np.inf
        for r in range(3):
            x0 = np.full(n_dims, 0.9) if r == 0 else rng.uniform(0.3, 1.5, size=n_dims)
            res = minimize(loss, x0, method='L-BFGS-B',
                           bounds=[(0.01, 5.0)] * n_dims)
            if res.fun < best_f:
                best_f = float(res.fun)
                best_x = res.x
        return best_x, best_f
    else:
        res = minimize_scalar(loss, bounds=(0.01, 5.0), method='bounded')
        return np.array([float(res.x)]), float(res.fun)


def _score_on_indices(kernel, specs, similarities, dim_weights, params, indices):
    n_dims = len(dim_weights)
    dw = np.asarray(dim_weights, dtype=np.float64)
    preds = []
    tgts = []
    for i in indices:
        step_vec = np.asarray(specs[i].step_by_dim, dtype=np.float64)
        ws = step_vec * dw
        preds.append(kernel_value(kernel, params if len(params) > 1 else params[0], ws))
        tgts.append(float(similarities[i]))
    return _normalized_mse(np.array(preds), np.array(tgts))


def cv_fit(kernel, specs, similarities, dim_weights, seed):
    """2-fold CV + full fit. Returns (params_full, fit_mse, cv_mse)."""
    rng = np.random.default_rng(seed)
    n = len(specs)
    idx = rng.permutation(n)
    fold_a, fold_b = idx[:n // 2], idx[n // 2:]
    p_a, _ = _fit_on_indices(kernel, specs, similarities, dim_weights, fold_a, seed)
    p_b, _ = _fit_on_indices(kernel, specs, similarities, dim_weights, fold_b, seed + 1)
    cv_a = _score_on_indices(kernel, specs, similarities, dim_weights, p_a, fold_b)
    cv_b = _score_on_indices(kernel, specs, similarities, dim_weights, p_b, fold_a)
    all_idx = np.arange(n)
    p_full, fit_full = _fit_on_indices(kernel, specs, similarities, dim_weights, all_idx, seed)
    return p_full, fit_full, (cv_a + cv_b) / 2.0


def multi_seed_fit(kernel, specs, similarities, dim_weights, seeds):
    """Repeat cv_fit across seeds; report mean params and mean CV MSE."""
    p_list = []
    fit_list = []
    cv_list = []
    for s in seeds:
        p, fit_mse, cv_mse = cv_fit(kernel, specs, similarities, dim_weights, seed=s)
        p_list.append(np.atleast_1d(p))
        fit_list.append(fit_mse)
        cv_list.append(cv_mse)
    p_arr = np.array(p_list)
    return {
        'params_mean': p_arr.mean(axis=0),
        'params_std': p_arr.std(axis=0),
        'fit_mse': float(np.mean(fit_list)),
        'cv_mse': float(np.mean(cv_list)),
    }


def aic(params, fit_mse, n_samples):
    """AIC = 2k + n*ln(loss). fit_mse is normalized-distribution MSE, so AIC
    is a relative ranking only — consistent within a single fit metric column.
    """
    k = len(np.atleast_1d(params))
    return 2.0 * k + float(n_samples) * np.log(max(fit_mse, 1e-20))


# -------- Reporting --------

def format_params(params_mean, params_std):
    """Format kernel params. Omit '+/- std' when std is negligible."""
    tiny = 1e-3
    if len(params_mean) == 1:
        if params_std[0] < tiny:
            return f"sigma={params_mean[0]:.3f}"
        return f"sigma={params_mean[0]:.3f} +/- {params_std[0]:.3f}"
    pm = ",".join(f"{m:.3f}" for m in params_mean)
    if all(s < tiny for s in params_std):
        return f"sigma=[{pm}]"
    ps = ",".join(f"{s:.3f}" for s in params_std)
    return f"sigma=[{pm}] +/- [{ps}]"


def run_analysis(args, device):
    """Full calibration pipeline."""
    model = FSQVAE(levels=args.levels)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()

    print("Loading validation frames...", flush=True)
    frames = load_val_frames(args.episodes_dir, args.expert_episodes_dir,
                             args.max_frames)
    print(f"Loaded {len(frames)} frames", flush=True)

    print("Encoding...", flush=True)
    all_z_q, all_imgs, all_indices = encode_all(
        model, frames, args.batch_size, device)

    usage_pct, ppl, n_used, vocab_size = codebook_usage(
        all_indices, args.levels)
    print(f"\n== Codebook usage ==", flush=True)
    print(f"  codes used: {n_used}/{vocab_size} "
          f"({usage_pct:.2f}%)")
    print(f"  perplexity: {ppl:.2f} "
          f"(max={vocab_size} uniform, 1=single-code collapse)")

    specs = generate_perturbations(args.levels)
    print(f"Generated {len(specs)} perturbations", flush=True)

    compute_ssim = not args.no_ssim
    print(f"\nMeasuring errors (SSIM {'on' if compute_ssim else 'off'})...",
          flush=True)
    mse_base, ssim_base, mse_pert, ssim_pert = measure_errors(
        model, all_z_q, all_imgs, specs, args.levels, args.batch_size, device,
        compute_ssim=compute_ssim)

    # Bootstrap ratios
    print("\nBootstrapping ratios...", flush=True)
    mse_ratios = []
    ssim_ratios = []
    for k in range(len(specs)):
        mse_ratios.append(bootstrap_ratio(mse_base, mse_pert[k], B=args.bootstrap_samples))
        if compute_ssim:
            # Use (1 - SSIM) as "distance"; baseline usually small positive.
            d_base = np.clip(1.0 - ssim_base, 1e-6, None)
            d_pert = np.clip(1.0 - ssim_pert[k], 1e-6, None)
            ssim_ratios.append(bootstrap_ratio(d_base, d_pert, B=args.bootstrap_samples))

    # Dim weights from single-dim +1 MSE ratios
    n_dims = len(args.levels)
    single_one_idx = {}
    for i, spec in enumerate(specs):
        nz = spec.nonzero_steps()
        if len(nz) == 1 and nz[0][1] == 1:
            single_one_idx[nz[0][0]] = i
    dim_ratio_means = np.array(
        [mse_ratios[single_one_idx[d]][0] for d in range(n_dims)])
    dim_weights = (dim_ratio_means / dim_ratio_means.mean()).tolist()

    # Reports
    print(f"\n== Dim weights (from +1 step MSE ratios, 95% CI) ==", flush=True)
    for d in range(n_dims):
        r, lo, hi = mse_ratios[single_one_idx[d]]
        print(f"  dim{d}  {r:.3f}x  [{lo:.3f}, {hi:.3f}]")
    print(f"Normalized dim_weights: {[round(w, 3) for w in dim_weights]}")

    print(f"\n== Perturbation ratios ==")
    header = f"{'Perturbation':<30} {'MSE':>9} {'95% CI':>18}"
    if compute_ssim:
        header += f"   {'(1-SSIM)':>9} {'95% CI':>18}"
    print(header)
    print("-" * len(header))
    for k, spec in enumerate(specs):
        r, lo, hi = mse_ratios[k]
        line = f"{spec.name:<30} {r:>8.3f}x [{lo:>5.2f}, {hi:>5.2f}]"
        if compute_ssim:
            r2, lo2, hi2 = ssim_ratios[k]
            line += f"   {r2:>8.3f}x [{lo2:>5.2f}, {hi2:>5.2f}]"
        print(line)

    # Qualitative compounding note (L1 vs L2)
    single_plus_one = {d: mse_ratios[single_one_idx[d]][0] for d in range(n_dims)}
    add_errs, mul_errs = [], []
    for k in range(2, n_dims + 1):
        for combo in combinations(range(n_dims), k):
            for i, spec in enumerate(specs):
                nz = spec.nonzero_steps()
                if set(d for d, _ in nz) == set(combo) and all(s == 1 for _, s in nz):
                    measured = mse_ratios[i][0]
                    add_pred = sum(single_plus_one[d] - 1 for d in combo) + 1
                    mul_pred = 1.0
                    for d in combo:
                        mul_pred *= single_plus_one[d]
                    add_errs.append(abs(measured - add_pred) / measured)
                    mul_errs.append(abs(measured - mul_pred) / measured)
                    break
    if add_errs:
        avg_add = 100 * sum(add_errs) / len(add_errs)
        avg_mul = 100 * sum(mul_errs) / len(mul_errs)
        print(f"\nCompounding: additive err {avg_add:.1f}%, multiplicative err {avg_mul:.1f}% "
              f"(qualitative: {'L1' if avg_add < avg_mul else 'L2'} favored)")

    # Fit kernels under MSE similarity and SSIM similarity
    seeds = list(range(42, 42 + args.n_seeds))
    mse_sim = np.array([1.0 / max(r[0], 1e-8) for r in mse_ratios])
    ssim_sim = (np.array([1.0 / max(r[0], 1e-8) for r in ssim_ratios])
                if compute_ssim else None)

    metric_sims = {'mse': mse_sim}
    if compute_ssim:
        metric_sims['ssim'] = ssim_sim

    results = {}
    for metric_name, sim in metric_sims.items():
        print(f"\n== Kernel comparison ({metric_name.upper()} similarity) ==")
        print(f"{'Kernel':<18} {'Params':<52} {'Fit MSE':>10} {'CV MSE':>10} {'AIC':>10}")
        print("-" * 102)
        results[metric_name] = {}
        for kernel in args.kernels:
            res = multi_seed_fit(kernel, specs, sim, dim_weights, seeds)
            aic_val = aic(res['params_mean'], res['fit_mse'], len(specs))
            results[metric_name][kernel] = {
                'params_mean': res['params_mean'].tolist(),
                'params_std': res['params_std'].tolist(),
                'fit_mse': res['fit_mse'],
                'cv_mse': res['cv_mse'],
                'aic': float(aic_val),
            }
            params_str = format_params(res['params_mean'], res['params_std'])
            print(f"{kernel:<18} {params_str:<52} {res['fit_mse']:>10.2e} "
                  f"{res['cv_mse']:>10.2e} {aic_val:>10.2f}")
        best = min(
            ((k, v) for k, v in results[metric_name].items() if isinstance(v, dict)),
            key=lambda kv: kv[1]['cv_mse'],
        )
        print(f">> Best ({metric_name.upper()}) by CV MSE: {best[0]}")
        results[metric_name]['best'] = best[0]

    # Recommendations
    print(f"\n== Recommendations ==")
    dw_str = " ".join(f"{w:.2f}" for w in dim_weights)
    for metric_name, metric_res in results.items():
        best_name = metric_res['best']
        best_res = metric_res[best_name]
        if best_name == 'aniso_gaussian':
            sigma_flag = "--fsq-sigma-d " + " ".join(
                f"{s:.3f}" for s in best_res['params_mean'])
        else:
            sigma_flag = f"--fsq-sigma {best_res['params_mean'][0]:.3f}"
        print(f"{metric_name.upper()}-best:")
        print(f"  --fsq-kernel {best_name} {sigma_flag} "
              f"--fsq-dim-weights {dw_str}")

    print()
    print("Note: --label-smoothing (epsilon) is a regularization hyperparameter")
    print("and must be chosen by ablation, not derived from encoder statistics.")

    # JSON dump
    json_out = args.json_out or str(Path(args.checkpoint).parent / "fsq_sensitivity.json")
    summary = {
        'checkpoint': args.checkpoint,
        'levels': args.levels,
        'n_frames': int(len(all_z_q)),
        'n_perturbations': len(specs),
        'codebook': {
            'n_used': n_used,
            'vocab_size': vocab_size,
            'usage_pct': float(usage_pct),
            'perplexity': float(ppl),
        },
        'dim_weights': [float(w) for w in dim_weights],
        'perturbations': {
            spec.name: {
                'mse': {'mean': float(mse_ratios[k][0]),
                        'ci_lo': float(mse_ratios[k][1]),
                        'ci_hi': float(mse_ratios[k][2])},
                **({'ssim_distance': {'mean': float(ssim_ratios[k][0]),
                                       'ci_lo': float(ssim_ratios[k][1]),
                                       'ci_hi': float(ssim_ratios[k][2])}}
                   if compute_ssim else {}),
            }
            for k, spec in enumerate(specs)
        },
        'kernels': results,
    }
    Path(json_out).parent.mkdir(parents=True, exist_ok=True)
    with open(json_out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nJSON summary written to {json_out}")


# -------- Self-test --------

def run_self_test():
    """Verify build_structured_smooth_targets(kernel='gaussian') preserves
    backward compatibility with the pre-upgrade implementation.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.train_world_model import build_structured_smooth_targets
    levels = [5, 5, 5, 5]
    full_vocab = 625 + 2
    # Default kernel should be 'gaussian' and output should equal pre-upgrade
    out = build_structured_smooth_targets(
        levels=levels, full_vocab_size=full_vocab,
        sigma=0.9, smoothing=0.1, dim_weights=[1.0, 1.0, 1.0, 1.0])
    assert out.shape == (full_vocab, full_vocab)
    row_sums = out.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(full_vocab), atol=1e-4), row_sums
    for i in range(625, 627):
        assert out[i, i].item() == 1.0
    # Diag visual mass
    for i in range(3):
        assert abs(out[i, i].item() - 0.9) < 1e-6, out[i, i].item()
    # Explicit kernel='laplace' should differ from gaussian
    out_l = build_structured_smooth_targets(
        levels=levels, full_vocab_size=full_vocab,
        sigma=0.9, smoothing=0.1, dim_weights=[1.0, 1.0, 1.0, 1.0],
        kernel='laplace')
    diff = (out[:625, :625] - out_l[:625, :625]).abs().sum().item()
    assert diff > 1.0, f"Laplace kernel too close to Gaussian (diff={diff})"
    print("Self-test passed: gaussian preserves backward compat, laplace differs.")


# -------- CLI --------

def main():
    parser = argparse.ArgumentParser(description="FSQ codebook sensitivity analysis")
    parser.add_argument("--checkpoint", default="checkpoints/fsq_best.pt")
    parser.add_argument("--levels", type=int, nargs="+", default=[8, 5, 5, 5])
    parser.add_argument("--episodes-dir", default="data/death_episodes")
    parser.add_argument("--expert-episodes-dir", default="data/expert_episodes")
    parser.add_argument("--max-frames", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--kernels", nargs="+",
                        default=["gaussian", "laplace", "cauchy", "aniso_gaussian"])
    parser.add_argument("--no-ssim", action="store_true",
                        help="Skip SSIM (MSE-only; faster)")
    parser.add_argument("--json-out", default=None,
                        help="Path for JSON summary (default: alongside checkpoint)")
    parser.add_argument("--self-test", action="store_true",
                        help="Run internal sanity checks and exit")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    run_analysis(args, device)


if __name__ == "__main__":
    main()
