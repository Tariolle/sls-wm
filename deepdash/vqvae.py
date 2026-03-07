"""Vector Quantized VAE for Geometry Dash frame compression.

Replaces the Gaussian latent with a discrete codebook.
No KL divergence = no blurriness from posterior averaging.
Spatial latent: 6x6 grid of codebook indices (36 tokens per frame).
No-padding encoder: 64 -> 31 -> 14 -> 6. MSE reconstruction loss.
"""

import torch
import torch.nn as nn


NUM_EMBEDDINGS = 1024
EMBEDDING_DIM = 8
IMG_CHANNELS = 1


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=NUM_EMBEDDINGS, embedding_dim=EMBEDDING_DIM,
                 commitment_cost=0.25, ema_decay=0.99):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)
        self.embedding.weight.requires_grad_(False)

        self.register_buffer('ema_cluster_size', torch.ones(num_embeddings))
        self.register_buffer('ema_embed_sum', self.embedding.weight.data.clone())
        self.register_buffer('forward_count', torch.tensor(0))

    @torch.no_grad()
    def _kmeans_init(self, z_e_flat, n_iter=10):
        idx = torch.randperm(z_e_flat.size(0), device=z_e_flat.device)[:self.num_embeddings]
        centroids = z_e_flat[idx].detach().clone()
        for _ in range(n_iter):
            dists = (z_e_flat.pow(2).sum(1, keepdim=True)
                     - 2 * z_e_flat @ centroids.t()
                     + centroids.pow(2).sum(1, keepdim=True).t())
            assignments = dists.argmin(dim=1)
            encodings = torch.zeros(z_e_flat.size(0), self.num_embeddings, device=z_e_flat.device)
            encodings.scatter_(1, assignments.unsqueeze(1), 1)
            cluster_size = encodings.sum(0)
            new_centroids = encodings.t() @ z_e_flat
            mask = cluster_size > 0
            centroids[mask] = new_centroids[mask] / cluster_size[mask].unsqueeze(1)
        self.embedding.weight.data.copy_(centroids)
        self.ema_embed_sum.copy_(centroids)
        self.ema_cluster_size.fill_(1)

    def _reset_dead_entries(self, z_e_flat):
        dead = self.ema_cluster_size < 1
        n_dead = dead.sum().item()
        if n_dead == 0:
            return
        rand_idx = torch.randint(0, z_e_flat.size(0), (n_dead,), device=z_e_flat.device)
        new_embeds = z_e_flat[rand_idx].detach()
        self.embedding.weight.data[dead] = new_embeds
        self.ema_embed_sum[dead] = new_embeds
        self.ema_cluster_size[dead] = 1

    def forward(self, z_e):
        # z_e: (B, D, H, W)
        B, D, H, W = z_e.shape
        z_e_flat = z_e.permute(0, 2, 3, 1).reshape(-1, D)

        # K-means init on first training forward pass
        if self.training and self.forward_count == 0:
            self._kmeans_init(z_e_flat)

        # Squared distances to codebook vectors
        distances = (
            z_e_flat.pow(2).sum(1, keepdim=True)
            - 2 * z_e_flat @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(1, keepdim=True).t()
        )

        # Nearest codebook entry
        indices = distances.argmin(dim=1)
        z_q_flat = self.embedding(indices)

        # EMA codebook update during training
        if self.training:
            self.forward_count += 1
            encodings = torch.zeros(indices.size(0), self.num_embeddings, device=z_e.device)
            encodings.scatter_(1, indices.unsqueeze(1), 1)

            self.ema_cluster_size.mul_(self.ema_decay).add_(encodings.sum(0), alpha=1 - self.ema_decay)
            self.ema_embed_sum.mul_(self.ema_decay).add_(encodings.t() @ z_e_flat.detach(), alpha=1 - self.ema_decay)

            # Laplace smoothing to avoid division by zero
            n = self.ema_cluster_size.sum()
            cluster_size = (self.ema_cluster_size + 1e-5) / (n + self.num_embeddings * 1e-5) * n
            self.embedding.weight.data = self.ema_embed_sum / cluster_size.unsqueeze(1)

            if self.forward_count > 100:
                self._reset_dead_entries(z_e_flat)

        # Commitment loss only (codebook updated via EMA, not gradients)
        commit_loss = (z_e_flat - z_q_flat.detach()).pow(2).mean()

        # Straight-through estimator
        z_q = z_e + (z_q_flat.reshape(B, H, W, D).permute(0, 3, 1, 2) - z_e).detach()

        return z_q, self.commitment_cost * commit_loss, indices.reshape(B, H, W)


class Encoder(nn.Module):
    def __init__(self, img_channels=IMG_CHANNELS, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        # No-padding convs: 64 -> 31 -> 14 -> 6
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
    """MSE reconstruction + VQ commitment loss."""
    recon_loss = torch.nn.functional.mse_loss(recon_x, x, reduction='sum') / x.size(0)
    return recon_loss + vq_loss, recon_loss, vq_loss
