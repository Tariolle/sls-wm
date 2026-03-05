"""U-Net Variational Autoencoder for Geometry Dash edge-map compression.

Adapted from World Models (Ha & Schmidhuber, 2018) for 176x96 input.
Encoder: 4 stride-2 conv layers (176x96 -> 11x6 spatial)
Decoder: mirror with transposed convolutions + skip connections from encoder
Latent: 256-dimensional Gaussian (mu, logvar)
"""

import torch
import torch.nn as nn


LATENT_DIM = 256
IMG_CHANNELS = 1
# Spatial dims after encoder (H, W): 96/16=6, 176/16=11
ENCODER_SPATIAL = (6, 11)
ENCODER_OUT_CHANNELS = 256
FLATTEN_DIM = ENCODER_OUT_CHANNELS * ENCODER_SPATIAL[0] * ENCODER_SPATIAL[1]  # 16896


class Encoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(IMG_CHANNELS, 32, 4, stride=2, padding=1), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(128, ENCODER_OUT_CHANNELS, 4, stride=2, padding=1), nn.ReLU())
        self.fc_mu = nn.Linear(FLATTEN_DIM, latent_dim)
        self.fc_logvar = nn.Linear(FLATTEN_DIM, latent_dim)

    def forward(self, x):
        s1 = self.conv1(x)    # (B, 32, 48, 88)
        s2 = self.conv2(s1)   # (B, 64, 24, 44)
        s3 = self.conv3(s2)   # (B, 128, 12, 22)
        h = self.conv4(s3)    # (B, 256, 6, 11)
        flat = h.view(h.size(0), -1)
        return self.fc_mu(flat), self.fc_logvar(flat), [s1, s2, s3]


SKIP_DROPOUT = 0.7


class Decoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.fc = nn.Linear(latent_dim, FLATTEN_DIM)
        self.skip_drop = nn.Dropout2d(SKIP_DROPOUT)
        # Each deconv input doubles channels to account for skip concatenation
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(ENCODER_OUT_CHANNELS, 128, 4, stride=2, padding=1), nn.ReLU())
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(128 + 128, 64, 4, stride=2, padding=1), nn.ReLU())
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(64 + 64, 32, 4, stride=2, padding=1), nn.ReLU())
        self.deconv4 = nn.Sequential(nn.ConvTranspose2d(32 + 32, IMG_CHANNELS, 4, stride=2, padding=1), nn.Sigmoid())

    def forward(self, z, skips):
        s1, s2, s3 = skips
        h = self.fc(z)
        h = h.view(h.size(0), ENCODER_OUT_CHANNELS, *ENCODER_SPATIAL)
        h = self.deconv1(h)
        h = self.deconv2(torch.cat([h, self.skip_drop(s3)], dim=1))
        h = self.deconv3(torch.cat([h, self.skip_drop(s2)], dim=1))
        h = self.deconv4(torch.cat([h, self.skip_drop(s1)], dim=1))
        return h


class VAE(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x):
        mu, logvar, skips = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, skips)
        return recon, mu, logvar

    def encode(self, x):
        mu, logvar, _ = self.encoder(x)
        return self.reparameterize(mu, logvar)


def vae_loss(recon_x, x, mu, logvar, beta=1.0, pos_weight=20.0):
    """Weighted BCE reconstruction + beta-weighted KL divergence.

    pos_weight compensates for sparse edge maps (~5% white pixels)
    by penalizing missed edges much more than false positives.
    """
    weight = torch.where(x > 0.1, pos_weight, 1.0)
    recon_loss = nn.functional.binary_cross_entropy(recon_x, x, weight=weight, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss
