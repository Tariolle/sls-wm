"""SIGReg: Sketched Isotropic Gaussian Regularizer.

Anti-collapse regularizer for latent world models. For each spatial position
of the pre-quantization encoder output, treats the batch of D-dim features as
samples of a random variable and pushes its distribution toward isotropic
N(0, I_D) via Cramer-Wold (M random 1D projections) + univariate Epps-Pulley
normality test per projection.

Reference:
    Maes et al., LeWorldModel: Stable End-to-End Joint-Embedding Predictive
    Architecture from Pixels. arXiv:2603.19312. SIGReg definition in App. A.

Applied per-position (not per-frame-flat) because our FSQ quantizes each of
the 64 spatial positions independently, and flattening to a 256-d vector
would force inter-position independence that fights natural spatial
correlations in the game frames.
"""

from __future__ import annotations

import torch


def sigreg_per_position(
    z_e: torch.Tensor,
    n_projections: int = 1024,
    n_knots: int = 17,
    knot_min: float = 0.2,
    knot_max: float = 4.0,
    position_chunk: int = 8,
) -> torch.Tensor:
    """Per-position SIGReg against isotropic N(0, I_D).

    Args:
        z_e: (B, D, H, W) pre-quantization continuous features. B is the
            effective sample count per position (batch x context frames).
        n_projections: M, random unit directions per position.
        n_knots: K, trapezoid quadrature knots for the Epps-Pulley integral.
        knot_min, knot_max: integration bounds (frequency t of the empirical
            characteristic function). LeWM default [0.2, 4.0].
        position_chunk: positions processed per tensor op; memory trade.
            Peak intermediate shape is (position_chunk, B, M, K).

    Returns:
        Scalar loss. Mean across H*W positions of the per-position SIGReg
        statistic, which itself averages M per-direction Epps-Pulley values.
    """
    B, D, H, W = z_e.shape
    P = H * W
    device = z_e.device
    dtype = z_e.dtype

    # (B, D, H, W) -> (P, B, D): per-position sample cloud.
    z = z_e.permute(2, 3, 0, 1).reshape(P, B, D)

    # Fresh random directions per call (no persistent buffer; matches LeWM).
    u = torch.randn(n_projections, D, device=device, dtype=dtype)
    u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-8)  # (M, D)

    t = torch.linspace(knot_min, knot_max, n_knots, device=device, dtype=dtype)
    # N(0,1) characteristic function is real-valued: phi_0(t) = exp(-t^2/2).
    target_cf = torch.exp(-0.5 * t * t)  # (K,)
    # Paper App. A example weighting: w(t) = exp(-t^2 / (2 lambda^2)); take lambda=1.
    weight = target_cf  # (K,)
    dt = (knot_max - knot_min) / (n_knots - 1)

    pieces = []
    for p_start in range(0, P, position_chunk):
        p_end = min(p_start + position_chunk, P)
        zc = z[p_start:p_end]                            # (p, B, D)
        proj = torch.einsum("pbd,md->pbm", zc, u)        # (p, B, M)
        outer = proj.unsqueeze(-1) * t.view(1, 1, 1, -1) # (p, B, M, K)
        cos_mean = outer.cos().mean(dim=1)               # (p, M, K)
        sin_mean = outer.sin().mean(dim=1)               # (p, M, K)
        sq = (cos_mean - target_cf.view(1, 1, -1)).pow(2) + sin_mean.pow(2)
        integrand = sq * weight.view(1, 1, -1)           # (p, M, K)
        # Trapezoid over K.
        T_pm = (
            integrand[..., 0] * 0.5
            + integrand[..., -1] * 0.5
            + integrand[..., 1:-1].sum(dim=-1)
        ) * dt                                           # (p, M)
        pieces.append(T_pm)

    all_pm = torch.cat(pieces, dim=0)  # (P, M)
    return all_pm.mean()
