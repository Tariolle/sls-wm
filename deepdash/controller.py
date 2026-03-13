"""Linear controller for CMA-ES optimization.

Maps the Transformer's hidden state h_t to a binary action (jump/idle).
    action = sigmoid(W @ h_t + b) > 0.5

Total parameters: hidden_dim + 1 (257 for embed_dim=256).
"""

import numpy as np
import torch


class Controller:
    def __init__(self, hidden_dim=256):
        self.hidden_dim = hidden_dim
        self.n_params = hidden_dim + 1  # W + b

    def set_params(self, flat_params):
        """Set W and b from a flat numpy array."""
        self.W = torch.from_numpy(flat_params[:self.hidden_dim].copy()).float()
        self.b = float(flat_params[self.hidden_dim])

    def act(self, h_t):
        """h_t: (B, hidden_dim) → actions: (B,) long {0=idle, 1=jump}."""
        logits = h_t @ self.W.to(h_t.device) + self.b
        return (logits.sigmoid() > 0.5).long()

    def save(self, path):
        flat = np.concatenate([self.W.numpy(), [self.b]])
        np.save(path, flat)

    @classmethod
    def load(cls, path, hidden_dim=256):
        flat = np.load(path)
        c = cls(hidden_dim)
        c.set_params(flat)
        return c
