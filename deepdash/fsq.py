"""Finite Scalar Quantization VAE for Geometry Dash frame compression.

Replaces VQ-VAE's codebook lookup with simple rounding to fixed levels.
No codebook, no EMA, no dead entries, no commitment loss tuning.
8x8 spatial grid (64 tokens/frame), padding=1 encoder.

FSQ levels [8,5,5,5] → 1000 implicit codes with 4d latent per spatial position.

References:
    - Mentzer et al., 2023: Finite Scalar Quantization
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

    def forward(self, z_e):
        """Quantize encoder output via tanh + round.

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
        z_bounded = torch.tanh(z_e) * half
        z_q = z_bounded.round()
        # Clamp to valid range to handle even levels correctly
        levels = self.levels.float().view(1, -1, 1, 1)
        z_q = z_q.clamp(-half, half - (1.0 - levels % 2))

        # Straight-through estimator: gradient flows through tanh, not round
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
    def __init__(self, img_channels=1, latent_dim=5, n_downsample=3):
        super().__init__()
        # n_downsample=3: 64->32->16->8, n_downsample=2: 64->32->16
        channels = [32, 64, 128][:n_downsample]
        layers = []
        in_ch = img_channels
        for ch in channels:
            layers.append(nn.Conv2d(in_ch, ch, 4, stride=2, padding=1))
            layers.append(nn.SiLU(inplace=True))
            layers.append(ResBlock(ch))
            layers.append(ResBlock(ch))
            in_ch = ch
        self.layers = nn.Sequential(*layers)
        self.proj = nn.Conv2d(in_ch, latent_dim, 1)

    def forward(self, x):
        return self.proj(self.layers(x))


class Decoder(nn.Module):
    def __init__(self, img_channels=1, latent_dim=5, n_upsample=3):
        super().__init__()
        # Mirror encoder: n_upsample=3: 8->16->32->64, n_upsample=2: 16->32->64
        channels = [128, 64, 32][-n_upsample:]
        self.proj = nn.Conv2d(latent_dim, channels[0], 1)
        self.proj_act = nn.SiLU(inplace=True)
        self.res_first = nn.Sequential(ResBlock(channels[0]), ResBlock(channels[0]))
        layers = []
        for i in range(len(channels) - 1):
            layers.append(nn.ConvTranspose2d(channels[i], channels[i + 1], 4, stride=2, padding=1))
            layers.append(nn.SiLU(inplace=True))
            layers.append(ResBlock(channels[i + 1]))
            layers.append(ResBlock(channels[i + 1]))
        self.layers = nn.Sequential(*layers)
        self.final = nn.ConvTranspose2d(channels[-1], img_channels, 4, stride=2, padding=1)

    def forward(self, z_q):
        x = self.proj_act(self.proj(z_q))
        x = self.res_first(x)
        x = self.layers(x)
        return torch.sigmoid(self.final(x))


class FSQVAE(nn.Module):
    def __init__(self, img_channels=1, levels=None, grid_size=8):
        super().__init__()
        levels = levels or DEFAULT_LEVELS
        latent_dim = len(levels)
        # 64 / grid_size = downsampling factor, log2 gives number of stride-2 convs
        n_downsample = int(math.log2(64 // grid_size))
        self.encoder = Encoder(img_channels, latent_dim, n_downsample)
        self.fsq = FSQQuantizer(levels)
        self.decoder = Decoder(img_channels, latent_dim, n_downsample)
        self.grid_size = grid_size

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


def fsqvae_loss(recon_x, x):
    """MSE reconstruction loss only. FSQ has no codebook loss."""
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
    return recon_loss


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
