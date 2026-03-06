"""Vector Quantized VAE for Geometry Dash frame compression.

Replaces the Gaussian latent with a discrete codebook.
No KL divergence = no blurriness from posterior averaging.
Spatial latent: 6x6 grid of codebook indices (36 tokens per frame).
Same encoder backbone as the working VAE, minus the last conv layer.
"""

import torch
import torch.nn as nn


NUM_EMBEDDINGS = 512
EMBEDDING_DIM = 64
IMG_CHANNELS = 1


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=NUM_EMBEDDINGS, embedding_dim=EMBEDDING_DIM,
                 commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z_e):
        # z_e: (B, D, H, W)
        B, D, H, W = z_e.shape
        z_e_flat = z_e.permute(0, 2, 3, 1).reshape(-1, D)

        # Squared distances to codebook vectors
        distances = (
            z_e_flat.pow(2).sum(1, keepdim=True)
            - 2 * z_e_flat @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(1, keepdim=True).t()
        )

        # Nearest codebook entry
        indices = distances.argmin(dim=1)
        z_q_flat = self.embedding(indices)

        # Codebook loss: move codebook vectors towards encoder outputs
        vq_loss = (z_q_flat - z_e_flat.detach()).pow(2).mean()
        # Commitment loss: encourage encoder to commit to codebook entries
        commit_loss = (z_e_flat - z_q_flat.detach()).pow(2).mean()

        # Straight-through estimator
        z_q = z_e + (z_q_flat.reshape(B, H, W, D).permute(0, 3, 1, 2) - z_e).detach()

        return z_q, vq_loss + self.commitment_cost * commit_loss, indices.reshape(B, H, W)


class Encoder(nn.Module):
    def __init__(self, img_channels=IMG_CHANNELS, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        # Same no-padding convs as VAE, minus the last layer
        # 64 -> 31 -> 14 -> 6 (stops here instead of going to 2)
        self.conv1 = nn.Conv2d(img_channels, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.proj = nn.Conv2d(128, embedding_dim, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        return self.proj(x)  # (B, embedding_dim, 6, 6)


class Decoder(nn.Module):
    def __init__(self, img_channels=IMG_CHANNELS, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        self.proj = nn.Conv2d(embedding_dim, 128, 1)
        # 6 -> 14 -> 31 -> 64 (mirrors the encoder)
        self.deconv1 = nn.ConvTranspose2d(128, 64, 4, stride=2)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(32, img_channels, 4, stride=2)

    def forward(self, z_q):
        x = torch.relu(self.proj(z_q))
        x = torch.relu(self.deconv1(x))
        x = torch.relu(self.deconv2(x))
        return torch.sigmoid(self.deconv3(x))


class VQVAE(nn.Module):
    def __init__(self, img_channels=IMG_CHANNELS, num_embeddings=NUM_EMBEDDINGS,
                 embedding_dim=EMBEDDING_DIM, commitment_cost=0.25):
        super().__init__()
        self.encoder = Encoder(img_channels, embedding_dim)
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(img_channels, embedding_dim)

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss, indices = self.vq(z_e)
        recon = self.decoder(z_q)
        return recon, vq_loss, indices

    def encode(self, x):
        z_e = self.encoder(x)
        _, _, indices = self.vq(z_e)
        return indices

    def decode_indices(self, indices):
        z_q = self.vq.embedding(indices).permute(0, 3, 1, 2)
        return self.decoder(z_q)


def vqvae_loss(recon_x, x, vq_loss):
    """L1 reconstruction + VQ commitment loss."""
    recon_loss = torch.nn.functional.l1_loss(recon_x, x, reduction='sum') / x.size(0)
    return recon_loss + vq_loss, recon_loss, vq_loss
