"""Finite Scalar Quantization VAE for Geometry Dash frame compression.

Replaces VQ-VAE's codebook lookup with simple rounding to fixed levels.
No codebook, no EMA, no dead entries, no commitment loss tuning.
8x8 spatial grid (64 tokens/frame), padding=1 encoder.

FSQ levels [8,5,5,5] → 1000 implicit codes with 4d latent per spatial position.

References:
    - Mentzer et al., 2024: Finite Scalar Quantization (ICLR 2024)
    - Huh et al., 2023: GRWM — geometric regularization for world models
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_LEVELS = [8, 5, 5, 5]


class FSQQuantizer(nn.Module):
    def __init__(self, levels=None):
        super().__init__()
        levels = levels or DEFAULT_LEVELS
        self.register_buffer("levels", torch.tensor(levels, dtype=torch.long))
        self.dim = len(levels)
        self.codebook_size = math.prod(levels)

        # Precompute divisors for mixed-radix index conversion
        divs = []
        acc = 1
        for L in reversed(levels):
            divs.append(acc)
            acc *= L
        divs.reverse()
        self.register_buffer("divisors", torch.tensor(divs, dtype=torch.long))

        # Half-levels for centering: e.g. level=7 → half=3 → values in {-3..3}
        self.register_buffer("half_levels",
                             torch.tensor([L // 2 for L in levels], dtype=torch.float32))

    def bound(self, z_e):
        """Bound raw encoder output to [-half_d, +half_d] per channel.

        Exposed for regularizers that need the pre-round bounded representation.
        """
        half = self.half_levels.view(1, -1, 1, 1)
        return (2.0 * torch.sigmoid(1.6 * z_e) - 1.0) * half

    def forward(self, z_e):
        """Quantize encoder output via iFSQ bounding + round.

        Uses 2*sigmoid(1.6*z)-1 instead of tanh(z) for near-uniform bin
        utilization (~100% vs ~83% with tanh). See iFSQ (arXiv 2601.17124).

        Args:
            z_e: (B, D, H, W) float — raw encoder output, D = len(levels).

        Returns:
            z_q: (B, D, H, W) float — quantized (straight-through gradient).
            indices: (B, H, W) long — flat codebook indices.
        """
        # Bound and round per channel.
        # Odd L:  values in {-floor(L/2), ..., +floor(L/2)}  (symmetric)
        # Even L: values in {-L/2+1, ..., L/2}  then shift by -0.5 to center
        # Following the FSQ paper's approach for even levels.
        half = self.half_levels.view(1, -1, 1, 1)  # (1, D, 1, 1)
        z_bounded = self.bound(z_e)
        z_q = z_bounded.round()
        # Clamp to valid range to handle even levels correctly
        levels = self.levels.float().view(1, -1, 1, 1)
        z_q = z_q.clamp(-half, half - (1.0 - levels % 2))

        # Straight-through estimator: gradient flows through sigmoid, not round
        z_q = z_bounded + (z_q - z_bounded).detach()

        indices = self._codes_to_indices(z_q)
        return z_q, indices

    def _codes_to_indices(self, z_q):
        """Convert quantized codes to flat indices via mixed-radix encoding."""
        # z_q: (B, D, H, W), values in {-half..+half} (odd) or {-half..+half-1} (even)
        # Shift to non-negative: code_d + half_d → {0..L-1}
        half = self.half_levels.view(1, -1, 1, 1)
        codes_shifted = (z_q + half).long()  # (B, D, H, W) in {0..L-1}
        divs = self.divisors.view(1, -1, 1, 1)
        return (codes_shifted * divs).sum(dim=1)  # (B, H, W)

    def indices_to_codes(self, indices):
        """Convert flat indices back to quantized code vectors.

        Args:
            indices: (B, H, W) long — flat codebook indices.

        Returns:
            z_q: (B, D, H, W) float — quantized codes.
        """
        B, H, W = indices.shape
        codes = []
        remainder = indices
        for d in range(self.dim):
            codes.append(remainder // self.divisors[d])
            remainder = remainder % self.divisors[d]
        # (B, D, H, W), shift back to centered
        z_q = torch.stack(codes, dim=1).float()
        half = self.half_levels.view(1, -1, 1, 1)
        return z_q - half


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return F.silu(x + self.block(x))


class Encoder(nn.Module):
    def __init__(self, img_channels=1, latent_dim=5):
        super().__init__()
        # Padded convs: 64 -> 32 -> 16 -> 8
        self.conv1 = nn.Conv2d(img_channels, 32, 4, stride=2, padding=1)
        self.res1 = nn.Sequential(ResBlock(32), ResBlock(32))
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.res2 = nn.Sequential(ResBlock(64), ResBlock(64))
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.res3 = nn.Sequential(ResBlock(128), ResBlock(128))
        self.proj = nn.Conv2d(128, latent_dim, 1)

    def forward(self, x):
        x = F.silu(self.conv1(x))
        x = self.res1(x)
        x = F.silu(self.conv2(x))
        x = self.res2(x)
        x = F.silu(self.conv3(x))
        x = self.res3(x)
        return self.proj(x)  # (B, latent_dim, 8, 8)


class Decoder(nn.Module):
    def __init__(self, img_channels=1, latent_dim=5):
        super().__init__()
        self.proj = nn.Conv2d(latent_dim, 128, 1)
        self.res1 = nn.Sequential(ResBlock(128), ResBlock(128))
        # 8 -> 16 -> 32 -> 64
        self.deconv1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.res2 = nn.Sequential(ResBlock(64), ResBlock(64))
        self.deconv2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.res3 = nn.Sequential(ResBlock(32), ResBlock(32))
        self.deconv3 = nn.ConvTranspose2d(32, img_channels, 4, stride=2, padding=1)

    def forward(self, z_q):
        x = F.silu(self.proj(z_q))
        x = self.res1(x)
        x = F.silu(self.deconv1(x))
        x = self.res2(x)
        x = F.silu(self.deconv2(x))
        x = self.res3(x)
        return torch.sigmoid(self.deconv3(x))


class FSQVAE(nn.Module):
    def __init__(self, img_channels=1, levels=None):
        super().__init__()
        levels = levels or DEFAULT_LEVELS
        latent_dim = len(levels)
        self.encoder = Encoder(img_channels, latent_dim)
        self.fsq = FSQQuantizer(levels)
        self.decoder = Decoder(img_channels, latent_dim)

    @property
    def codebook_size(self):
        return self.fsq.codebook_size

    def forward(self, x):
        z_e = self.encoder(x)       # (B, D, 8, 8)
        z_q, indices = self.fsq(z_e)  # z_q: (B, D, 8, 8), indices: (B, 8, 8)
        recon = self.decoder(z_q)    # (B, 1, 64, 64)
        return recon, z_e, indices

    def encode(self, x):
        """Encode to flat token indices (B, 8, 8)."""
        z_e = self.encoder(x)
        _, indices = self.fsq(z_e)
        return indices

    def decode_indices(self, indices):
        """Decode flat indices back to images."""
        z_q = self.fsq.indices_to_codes(indices)
        return self.decoder(z_q)


def fsqvae_loss(recon_x, x, loss_type='mse'):
    """Reconstruction loss only. FSQ has no codebook loss.

    loss_type='mse': sum of squared errors per sample (legacy default).
    loss_type='l1': sum of absolute errors per sample. Sharper output
        under uncertainty (median vs L2's mean), but ~30x larger absolute
        magnitude at convergence -- recon_weight must be calibrated.
    """
    if loss_type == 'l1':
        return F.l1_loss(recon_x, x, reduction='sum') / x.size(0)
    return F.mse_loss(recon_x, x, reduction='sum') / x.size(0)


def grwm_slowness(z_e_t, z_e_t1):
    """Temporal slowness: penalize large latent changes between consecutive frames."""
    return (z_e_t - z_e_t1).pow(2).mean()


def grwm_uniformity(z_e, t=2.0):
    """Uniformity loss: prevent encoder collapse to a single point.

    Wang & Isola (2020): L_uniform = log(mean(exp(-t * ||z_i - z_j||^2))).
    Computed on flattened spatial features averaged per sample.
    """
    # z_e: (B, D, H, W) -> (B, D*H*W)
    z_flat = z_e.flatten(1)
    z_flat = F.normalize(z_flat, dim=1)
    # Pairwise squared distances
    sq_dists = torch.cdist(z_flat, z_flat).pow(2)  # (B, B)
    return sq_dists.mul(-t).exp().mean().log()


def fsq_marginal_uniform_reg(z_bounded, half_levels):
    """Per-dim Cramér-von Mises distance from Uniform[-half_d, +half_d].

    FSQ dims are factorised by construction, so each 1D marginal can be
    tested directly — no Cramér-Wold sketching needed. CvM is used
    instead of KS for smoother gradients.

    Accepts (B, D, H, W) [single batch] or (B, T, D, H, W) [T independent
    frames share the regulariser]. In the 5D case each frame contributes
    its own CvM-per-channel over B*H*W samples; the returned scalar is
    the mean across T and D. Single fused sort replaces the per-frame
    Python loop that was here before.
    """
    half = half_levels.to(z_bounded.device, dtype=z_bounded.dtype)
    if z_bounded.ndim == 4:
        B, D, H, W = z_bounded.shape
        z_flat = (z_bounded.permute(1, 0, 2, 3)
                  .reshape(D, B * H * W)
                  .contiguous())
        M = z_flat.size(1)
        h_sorted, _ = z_flat.sort(dim=1)
        i_emp = (torch.arange(M, device=z_flat.device, dtype=z_flat.dtype) + 0.5) / M
        cdf = ((h_sorted + half.view(D, 1)) / (2.0 * half.view(D, 1))).clamp(0.0, 1.0)
        return (cdf - i_emp.view(1, M)).pow(2).mean()
    if z_bounded.ndim == 5:
        B, T, D, H, W = z_bounded.shape
        z_flat = (z_bounded.permute(1, 2, 0, 3, 4)
                  .reshape(T, D, B * H * W)
                  .contiguous())
        M = z_flat.size(-1)
        h_sorted, _ = z_flat.sort(dim=-1)
        i_emp = (torch.arange(M, device=z_flat.device, dtype=z_flat.dtype) + 0.5) / M
        cdf = ((h_sorted + half.view(1, D, 1))
               / (2.0 * half.view(1, D, 1))).clamp(0.0, 1.0)
        return (cdf - i_emp.view(1, 1, M)).pow(2).mean()
    raise ValueError(f"expected 4D or 5D input, got {z_bounded.ndim}D")
