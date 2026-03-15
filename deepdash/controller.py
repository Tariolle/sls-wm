"""Controllers for World Model agent.

CMA-ES Controller: numpy-based, for evolutionary optimization.
PolicyController: nn.Module, for Reinforce policy gradient training.
"""

import numpy as np
import torch
import torch.nn as nn


class Controller:
    def __init__(self, hidden_dim=256, mlp_hidden=0):
        """
        Args:
            hidden_dim: dimension of h_t input.
            mlp_hidden: if > 0, use a 2-layer MLP with this hidden size.
                        if 0, use a linear controller.
        """
        self.hidden_dim = hidden_dim
        self.mlp_hidden = mlp_hidden

        if mlp_hidden > 0:
            # W1: (hidden_dim, mlp_hidden), b1: (mlp_hidden,)
            # W2: (mlp_hidden,), b2: scalar
            self.n_params = hidden_dim * mlp_hidden + mlp_hidden + mlp_hidden + 1
        else:
            self.n_params = hidden_dim + 1  # W + b

    def set_params(self, flat_params):
        """Set weights from a flat numpy array."""
        if self.mlp_hidden > 0:
            d, h = self.hidden_dim, self.mlp_hidden
            offset = 0
            self.W1 = torch.from_numpy(flat_params[offset:offset + d * h].copy()).float().reshape(d, h)
            offset += d * h
            self.b1 = torch.from_numpy(flat_params[offset:offset + h].copy()).float()
            offset += h
            self.W2 = torch.from_numpy(flat_params[offset:offset + h].copy()).float()
            offset += h
            self.b2 = float(flat_params[offset])
        else:
            self.W = torch.from_numpy(flat_params[:self.hidden_dim].copy()).float()
            self.b = float(flat_params[self.hidden_dim])

    def act(self, h_t):
        """h_t: (B, hidden_dim) → actions: (B,) long {0=idle, 1=jump}."""
        dev = h_t.device
        if self.mlp_hidden > 0:
            hidden = torch.relu(h_t @ self.W1.to(dev) + self.b1.to(dev))
            logits = hidden @ self.W2.to(dev) + self.b2
        else:
            logits = h_t @ self.W.to(dev) + self.b
        return (logits.sigmoid() > 0.5).long()

    def save(self, path):
        if self.mlp_hidden > 0:
            flat = np.concatenate([
                self.W1.numpy().ravel(), self.b1.numpy(),
                self.W2.numpy(), [self.b2]
            ])
        else:
            flat = np.concatenate([self.W.numpy(), [self.b]])
        np.save(path, flat)

    @classmethod
    def load(cls, path, hidden_dim=256, mlp_hidden=0):
        flat = np.load(path)
        c = cls(hidden_dim, mlp_hidden)
        c.set_params(flat)
        return c


class PolicyController(nn.Module):
    """MLP policy for Reinforce training.

    Maps h_t to a jump probability via LayerNorm → Linear → ReLU → Linear → sigmoid.
    Stochastic action sampling for training, deterministic for evaluation.
    """

    def __init__(self, hidden_dim=256, mlp_hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1),
        )
        # Small init on output layer for ~50/50 starting policy
        nn.init.uniform_(self.net[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h_t):
        """h_t: (B, hidden_dim) → jump probability: (B,)"""
        return self.net(h_t).squeeze(-1).sigmoid()

    def act(self, h_t):
        """Sample action from Bernoulli policy.

        Returns:
            action: (B,) long {0=idle, 1=jump}
            log_prob: (B,) log probability of the sampled action
            entropy: (B,) policy entropy
        """
        prob = self.forward(h_t)
        dist = torch.distributions.Bernoulli(probs=prob)
        action = dist.sample()
        return action.long(), dist.log_prob(action), dist.entropy()

    def act_deterministic(self, h_t):
        """Greedy action for evaluation."""
        return (self.forward(h_t) > 0.5).long()
