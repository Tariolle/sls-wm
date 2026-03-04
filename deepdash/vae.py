"""Variational Autoencoder for Geometry Dash frame compression.

Adapted from World Models (Ha & Schmidhuber, 2018) for 176x96 input.
Encoder: 4 stride-2 conv layers (176x96 -> 11x6 spatial)
Decoder: mirror with transposed convolutions
Latent: 32-dimensional Gaussian (mu, logvar)
"""

import torch
import torch.nn as nn


LATENT_DIM = 32
IMG_CHANNELS = 3
# Spatial dims after encoder (H, W): 96/16=6, 176/16=11
ENCODER_SPATIAL = (6, 11)
ENCODER_OUT_CHANNELS = 256
FLATTEN_DIM = ENCODER_OUT_CHANNELS * ENCODER_SPATIAL[0] * ENCODER_SPATIAL[1]  # 16896


class Encoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(IMG_CHANNELS, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, ENCODER_OUT_CHANNELS, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(FLATTEN_DIM, latent_dim)
        self.fc_logvar = nn.Linear(FLATTEN_DIM, latent_dim)

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.fc = nn.Linear(latent_dim, FLATTEN_DIM)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(ENCODER_OUT_CHANNELS, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, IMG_CHANNELS, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(h.size(0), ENCODER_OUT_CHANNELS, *ENCODER_SPATIAL)
        return self.deconv(h)


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
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def encode(self, x):
        mu, logvar = self.encoder(x)
        return self.reparameterize(mu, logvar)


def vae_loss(recon_x, x, mu, logvar):
    """L1 reconstruction + KL divergence."""
    recon_loss = nn.functional.l1_loss(recon_x, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss, recon_loss, kl_loss
