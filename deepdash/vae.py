"""Variational Autoencoder for Geometry Dash frame compression.

Matches the World Models paper (Ha & Schmidhuber, 2018) and ctallec/world-models PyTorch reimplementation:
  - Input: 64x64x3 RGB
  - Encoder: 4 no-padding stride-2 conv layers (64→31→14→6→2), flatten to 1024
  - Decoder: asymmetric — 1×1×1024 upsampled with 5×5/6×6 transposed convs
  - Latent: 32-dimensional Gaussian (mu, logvar)
  - Loss: MSE + KL with tolerance floor (0.5 nats/dim)
"""

import torch
import torch.nn as nn


LATENT_DIM = 32
IMG_CHANNELS = 1
FLATTEN_DIM = 2 * 2 * 256  # 1024 — no padding encoder: 64→31→14→6→2


class Encoder(nn.Module):
    def __init__(self, img_channels=IMG_CHANNELS, latent_dim=LATENT_DIM):
        super().__init__()
        self.conv1 = nn.Conv2d(img_channels, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
        self.fc_mu = nn.Linear(FLATTEN_DIM, latent_dim)
        self.fc_logvar = nn.Linear(FLATTEN_DIM, latent_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)


class Decoder(nn.Module):
    def __init__(self, img_channels=IMG_CHANNELS, latent_dim=LATENT_DIM):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 1024)
        self.deconv1 = nn.ConvTranspose2d(1024, 128, 5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, img_channels, 6, stride=2)

    def forward(self, z):
        x = torch.relu(self.fc(z))
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = torch.relu(self.deconv1(x))
        x = torch.relu(self.deconv2(x))
        x = torch.relu(self.deconv3(x))
        return torch.sigmoid(self.deconv4(x))


class VAE(nn.Module):
    def __init__(self, img_channels=IMG_CHANNELS, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = Encoder(img_channels, latent_dim)
        self.decoder = Decoder(img_channels, latent_dim)

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


_sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
_sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)


def _sobel_magnitude(x):
    """Compute Sobel edge magnitude map (no grad, same device)."""
    kern_x = _sobel_x.to(x.device)
    kern_y = _sobel_y.to(x.device)
    gx = torch.nn.functional.conv2d(x, kern_x, padding=1)
    gy = torch.nn.functional.conv2d(x, kern_y, padding=1)
    return torch.sqrt(gx ** 2 + gy ** 2)


def vae_loss(recon_x, x, mu, logvar, kl_tolerance=0.5, edge_weight=5.0):
    """L1 reconstruction with 5x edge weighting + KL with tolerance floor."""
    with torch.no_grad():
        edges = _sobel_magnitude(x)
        edges = edges / (edges.max() + 1e-8)  # normalize to [0, 1]
        weight = 1.0 + edge_weight * edges
    recon_loss = torch.sum(torch.abs(recon_x - x) * weight, dim=[1, 2, 3]).mean()
    # KL disabled (pure autoencoder) to diagnose reconstruction capacity
    kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_loss = torch.tensor(0.0, device=recon_x.device)
    return recon_loss + kl_loss, recon_loss, kl_loss
