"""SIGReg: Sketched Isotropic Gaussian Regularizer.

Anti-collapse regularizer for latent world models. Computes the Epps-Pulley
test statistic against an isotropic N(0, I_D) target along M random 1D
projections, averaged across (time, spatial position) pairs.

Matches the reference implementation in LeWorldModel
(https://github.com/lucas-maes/le-wm, module.py) verbatim, just extended to
our (time x spatial) axis structure instead of a single CLS-token axis.

Reference:
    Maes et al., LeWorldModel: Stable End-to-End Joint-Embedding Predictive
    Architecture from Pixels. arXiv:2603.19312. SIGReg definition in App. A;
    authoritative behaviour is in the code (paper knot range says [0.2, 4.0],
    code uses [0, 3]).

Key detail: the per-(projection, position) statistic is multiplied by the
per-call sample count B. This is the standard Epps-Pulley normalization
(T = N * integral(w(t) * |ECF - phi_0|^2 dt)) that makes the statistic
converge to a chi-squared limit under the null and, practically, sets the
raw loss magnitude to be comparable to the prediction loss so lambda can
stay in the paper's 0.01-0.2 range. Without this multiplier the SIGReg
loss is ~B times smaller than it should be.
"""

from __future__ import annotations

import torch


def sigreg_per_position(
    z_e: torch.Tensor,
    n_projections: int = 1024,
    n_knots: int = 17,
    knot_max: float = 3.0,
    position_chunk: int = 16,
) -> torch.Tensor:
    """Per-(time, position) SIGReg against isotropic N(0, I_D).

    Args:
        z_e: (B, D, H, W) or (B, T, D, H, W). Treats each (t, h, w) triple
            as an independent distribution of B samples in R^D.
        n_projections: M, random unit directions per (t, h, w).
        n_knots: K, trapezoid quadrature knots.
        knot_max: integration upper bound; LeWM code uses 3.0 (paper says 4.0).
        position_chunk: positions processed per chunk (memory trade).

    Returns:
        Scalar loss. Mean across (T*H*W) triples of the Epps-Pulley
        statistic (which itself averages M projections and integrates K
        knots), multiplied by B per the standard Epps-Pulley normalization.
    """
    if z_e.ndim == 4:
        z_e = z_e.unsqueeze(1)  # (B, 1, D, H, W)
    B, T, D, H, W = z_e.shape
    P = T * H * W
    device = z_e.device
    dtype = z_e.dtype

    # (B, T, D, H, W) -> (T, H, W, B, D) -> (P, B, D).
    proj_src = z_e.permute(1, 3, 4, 0, 2).reshape(P, B, D).contiguous()

    # LeWM's quadrature: 17 knots in [0, 3], weights [dt, 2dt, ..., 2dt, dt]
    # (2x standard trapezoid; constant scale factor) pre-multiplied with
    # the w(t) = exp(-t^2/2) window.
    t = torch.linspace(0.0, knot_max, n_knots, device=device, dtype=dtype)
    dt = knot_max / (n_knots - 1)
    weights_trap = torch.full((n_knots,), 2 * dt, device=device, dtype=dtype)
    weights_trap[0] = dt
    weights_trap[-1] = dt
    window = torch.exp(-0.5 * t * t)
    # Target characteristic function phi_0(t) for N(0,1) is real: exp(-t^2/2).
    target_cf = window
    weights = weights_trap * window  # (K,) — combined trapezoid * window.

    # Fresh random unit-norm directions per call. Shape (D, M) matches
    # LeWM's `A` convention so we can `proj @ A` directly.
    A = torch.randn(D, n_projections, device=device, dtype=dtype)
    A = A / A.norm(p=2, dim=0, keepdim=True).clamp_min(1e-8)

    pieces = []
    for p_start in range(0, P, position_chunk):
        p_end = min(p_start + position_chunk, P)
        pc = proj_src[p_start:p_end]                         # (p, B, D)
        # (p, B, D) @ (D, M) -> (p, B, M); unsqueeze K axis; * t -> (p, B, M, K).
        x_t = (pc @ A).unsqueeze(-1) * t
        # Empirical characteristic function: mean over samples (dim -3 = B).
        err = (x_t.cos().mean(dim=-3) - target_cf).square() \
            + x_t.sin().mean(dim=-3).square()                # (p, M, K)
        # Integrate over K via precombined trapezoid*window weights, then
        # apply the Epps-Pulley N-multiplier (sample count per position).
        statistic = (err @ weights) * B                      # (p, M)
        pieces.append(statistic)

    all_stats = torch.cat(pieces, dim=0)  # (P, M)
    return all_stats.mean()
