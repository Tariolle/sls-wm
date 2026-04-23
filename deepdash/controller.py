"""Controllers for World Model agent.

CMA-ES Controller: numpy-based, for evolutionary optimization.
PolicyController: nn.Module MLP, for Reinforce policy gradient training.
TransformerPolicy: ViT encoder (DART-style), sees individual tokens with positions.
MLPPolicy: MLP on h_t (TWISTER/DreamerV3-style), actor-critic.
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

    def __init__(self, hidden_dim=256, mlp_hidden=64, extra_features=0):
        super().__init__()
        input_dim = hidden_dim + extra_features
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, mlp_hidden),
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


class TransformerPolicy(nn.Module):
    """Transformer encoder actor-critic policy (DART-style).

    Processes individual observation tokens with learnable positional encoding,
    enabling spatial attention for timing-critical decisions. h_t from the
    world model is injected as a context token.

    Input sequence: [CLS, obs_1, ..., obs_N, h_t_proj]
    Output: CLS representation -> actor (jump probability) + critic (value)

    Reference: DART (Agarwal et al., ICML 2024)
    """

    def __init__(self, wm_embed_dim=256, n_tokens=64, embed_dim=128,
                 n_heads=4, n_layers=3, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_tokens = n_tokens
        seq_len = n_tokens + 2  # CLS + tokens + h_t

        # Project WM token embeddings to policy dim
        self.token_proj = nn.Linear(wm_embed_dim, embed_dim)

        # Project h_t to policy dim
        self.h_proj = nn.Linear(wm_embed_dim, embed_dim)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Learnable positional encoding
        self.pos_embed = nn.Parameter(
            torch.randn(1, seq_len, embed_dim) * 0.02)

        # Transformer encoder (pre-norm for stability)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=embed_dim * 4, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers)

        self.ln_f = nn.LayerNorm(embed_dim)

        # Actor head (from CLS)
        self.actor = nn.Linear(embed_dim, 1)
        nn.init.uniform_(self.actor.weight, -0.01, 0.01)
        nn.init.zeros_(self.actor.bias)

        # Critic head (from CLS, separate)
        self.critic = nn.Linear(embed_dim, 1)

    def _encode(self, token_embeds, h_t):
        """Shared encoder forward, returns CLS representation.

        Args:
            token_embeds: (B, N, wm_embed_dim) from world model's token_embed.
            h_t: (B, wm_embed_dim) hidden state from world model.
        Returns:
            cls_out: (B, embed_dim)
        """
        B = token_embeds.shape[0]

        tokens = self.token_proj(token_embeds)        # (B, N, D)
        h = self.h_proj(h_t).unsqueeze(1)             # (B, 1, D)
        cls = self.cls_token.expand(B, -1, -1)        # (B, 1, D)

        x = torch.cat([cls, tokens, h], dim=1)        # (B, N+2, D)
        x = x + self.pos_embed

        x = self.encoder(x)
        x = self.ln_f(x)

        return x[:, 0]

    def forward(self, token_embeds, h_t):
        """Jump probability and value estimate.

        Returns:
            prob: (B,) jump probability
            value: (B,) state value estimate
        """
        cls_out = self._encode(token_embeds, h_t)
        prob = self.actor(cls_out).squeeze(-1).sigmoid()
        value = self.critic(cls_out).squeeze(-1)
        return prob, value

    def act(self, token_embeds, h_t):
        """Sample action from Bernoulli policy.

        Returns:
            action: (B,) long {0=idle, 1=jump}
            log_prob: (B,) log probability of sampled action
            entropy: (B,) policy entropy
            value: (B,) critic value estimate
        """
        prob, value = self.forward(token_embeds, h_t)
        dist = torch.distributions.Bernoulli(probs=prob)
        action = dist.sample()
        return action.long(), dist.log_prob(action), dist.entropy(), value

    def act_deterministic(self, token_embeds, h_t):
        """Greedy action for evaluation."""
        prob, _ = self.forward(token_embeds, h_t)
        return (prob > 0.5).long()


class MLPPolicy(nn.Module):
    """MLP actor-critic on world model hidden state.

    Pure MLP on h_t (TWISTER/DreamerV3-style). The transformer's hidden
    state already encodes spatial and temporal information after 8 layers
    of block-causal attention -- no CNN needed.

    Architecture:
        h_t (512d) -> Linear(512) + LayerNorm + SiLU
        Actor: Linear(512, 1)   (zero-init)
        Critic: Linear(512, 1)  (zero-init)
    """

    def __init__(self, h_dim=384, mlp_hidden=512, dropout=0.0, mlp_layers=1):
        super().__init__()
        layers = [
            nn.Linear(h_dim, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.SiLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        for _ in range(mlp_layers - 1):
            layers += [nn.Linear(mlp_hidden, mlp_hidden), nn.LayerNorm(mlp_hidden), nn.SiLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.trunk = nn.Sequential(*layers)

        # Actor-critic heads (zero-init like IRIS/DIAMOND)
        self.actor = nn.Linear(mlp_hidden, 1)
        self.critic = nn.Linear(mlp_hidden, 1)
        nn.init.zeros_(self.actor.weight)
        nn.init.zeros_(self.actor.bias)
        nn.init.zeros_(self.critic.weight)
        nn.init.zeros_(self.critic.bias)

    def _encode(self, h_t):
        """Encode h_t into shared representation.

        Args:
            h_t: (B, h_dim) float, world model hidden state.
        Returns:
            features: (B, mlp_hidden)
        """
        return self.trunk(h_t)

    def forward(self, h_t):
        """Jump probability and value estimate.

        Args:
            h_t: (B, h_dim) float.
        Returns:
            prob: (B,) jump probability
            value: (B,) state value estimate
        """
        features = self._encode(h_t)
        prob = self.actor(features).squeeze(-1).sigmoid()
        value = self.critic(features).squeeze(-1)
        return prob, value

    def act(self, h_t):
        """Sample action from Bernoulli policy.

        Returns:
            action: (B,) long {0=idle, 1=jump}
            log_prob: (B,)
            entropy: (B,)
            value: (B,)
        """
        prob, value = self.forward(h_t)
        dist = torch.distributions.Bernoulli(probs=prob)
        action = dist.sample()
        return action.long(), dist.log_prob(action), dist.entropy(), value

    def act_deterministic(self, h_t):
        """Greedy action for evaluation."""
        prob, _ = self.forward(h_t)
        return (prob > 0.5).long()
