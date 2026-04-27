"""Controllers for World Model agent.

CMA-ES Controller: numpy-based, for evolutionary optimization.
PolicyController: nn.Module MLP, for Reinforce policy gradient training.
TransformerPolicy: ViT encoder (DART-style), sees individual tokens with positions.
MLPPolicy: MLP on h_t (TWISTER/DreamerV3-style), actor-critic.
CNNPolicy: CNN on 8x8 token grid + h_t (IRIS/DIAMOND-style), actor-critic.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class CNNPolicy(nn.Module):
    """CNN actor-critic on z_t (spatial) + compressed h_t (temporal).

    Factorization of concerns:
      - z_t path: FSQ token grid -> learnable embedding -> 2 strided convs
        -> (64, 2, 2) = 256d, LayerNorm. Carries spatial info.
      - h_t path: LayerNorm -> Linear(h_dim, temporal_dim=32) -> SiLU.
        Bottleneck forces a compact temporal state (velocity, jump arc,
        mode). Dim-imbalanced with spatial (256) >> temporal (32) so the
        head is biased to read spatial features.

    Init: orthogonal (Engstrom et al. "Implementation Matters").
    """

    def __init__(self, vocab_size=625, grid_size=8, token_embed_dim=16,
                 h_dim=512, temporal_dim=32):
        super().__init__()
        self.grid_size = grid_size
        self.temporal_dim = temporal_dim

        self.token_embed = nn.Embedding(vocab_size, token_embed_dim)
        self.conv1 = nn.Conv2d(token_embed_dim, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)

        cnn_out = 64 * (grid_size // 4) ** 2  # 256 for 8x8
        self.spatial_norm = nn.LayerNorm(cnn_out)

        self.h_norm = nn.LayerNorm(h_dim)
        self.h_proj = nn.Linear(h_dim, temporal_dim)

        head_input = cnn_out + temporal_dim
        self.actor = nn.Linear(head_input, 1)
        self.critic = nn.Linear(head_input, 1)

        self._init_weights()

    def _init_weights(self):
        gain_hidden = 2 ** 0.5  # sqrt(2), SiLU/ReLU effective gain
        for m in (self.conv1, self.conv2, self.h_proj):
            nn.init.orthogonal_(m.weight, gain=gain_hidden)
            nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def _encode(self, token_ids, h_t):
        B = token_ids.shape[0]
        G = self.grid_size
        x = self.token_embed(token_ids)              # (B, 64, E)
        x = x.permute(0, 2, 1).reshape(B, -1, G, G)  # (B, E, 8, 8)
        x = F.silu(self.conv1(x))                    # (B, 32, 4, 4)
        x = F.silu(self.conv2(x))                    # (B, 64, 2, 2)
        x = x.flatten(1)                              # (B, 256)
        x = self.spatial_norm(x)

        t = F.silu(self.h_proj(self.h_norm(h_t)))    # (B, temporal_dim)
        return torch.cat([x, t], dim=1)               # (B, 256 + temporal_dim)

    def forward(self, token_ids, h_t):
        features = self._encode(token_ids, h_t)
        prob = self.actor(features).squeeze(-1).sigmoid()
        value = self.critic(features).squeeze(-1)
        return prob, value

    def act(self, token_ids, h_t):
        prob, value = self.forward(token_ids, h_t)
        dist = torch.distributions.Bernoulli(probs=prob)
        action = dist.sample()
        return action.long(), dist.log_prob(action), dist.entropy(), value

    def act_deterministic(self, token_ids, h_t):
        prob, _ = self.forward(token_ids, h_t)
        return (prob > 0.5).long()


class V3CNNPolicy(nn.Module):
    """V3-deploy faithful CNN actor-critic. NOT the same as CNNPolicy.

    Verbatim port of V3-deploy's controller (commit 75fe40a). Differences
    from CNNPolicy:
      - stride=1 convs + MaxPool(2x2) instead of stride=2 convs.
      - ReLU activation instead of SiLU.
      - h_t is concatenated DIRECTLY (no LayerNorm, no projection, no
        compression). head_input = 256 + h_dim.
      - Heads zero-init (IRIS/DIAMOND), no orthogonal scaling.
      - MTP head: predicts mtp_steps=8 future action probabilities for the
        auxiliary loss in PPO.
    Use only with the V3-style training pipeline (frozen FSQ + transformer
    with embed_dim=h_dim=384, vocab_size=1000).
    """

    def __init__(self, vocab_size=1000, grid_size=8, token_embed_dim=16,
                 h_dim=384, mtp_steps=8):
        super().__init__()
        self.grid_size = grid_size
        self.mtp_steps = mtp_steps

        self.token_embed = nn.Embedding(vocab_size, token_embed_dim)
        self.conv1 = nn.Conv2d(token_embed_dim, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)

        cnn_out = 64 * (grid_size // 4) ** 2  # 256 for 8x8 with two 2x2 pools
        head_input = cnn_out + h_dim

        self.actor = nn.Linear(head_input, 1)
        self.critic = nn.Linear(head_input, 1)
        self.mtp_head = nn.Linear(head_input, mtp_steps)
        for layer in (self.actor, self.critic, self.mtp_head):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _encode(self, token_ids, h_t):
        B = token_ids.shape[0]
        G = self.grid_size
        x = self.token_embed(token_ids)              # (B, 64, E)
        x = x.permute(0, 2, 1).reshape(B, -1, G, G)  # (B, E, 8, 8)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))   # (B, 32, 4, 4)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))   # (B, 64, 2, 2)
        x = x.flatten(1)                              # (B, 256)
        return torch.cat([x, h_t], dim=1)             # (B, 256 + h_dim)

    def forward(self, token_ids, h_t):
        features = self._encode(token_ids, h_t)
        prob = self.actor(features).squeeze(-1).sigmoid()
        value = self.critic(features).squeeze(-1)
        return prob, value

    def predict_future_action_logits(self, token_ids, h_t):
        features = self._encode(token_ids, h_t)
        return self.mtp_head(features)

    def predict_future_actions(self, token_ids, h_t):
        return self.predict_future_action_logits(token_ids, h_t).sigmoid()

    def act(self, token_ids, h_t):
        prob, value = self.forward(token_ids, h_t)
        dist = torch.distributions.Bernoulli(probs=prob)
        action = dist.sample()
        return action.long(), dist.log_prob(action), dist.entropy(), value

    def act_deterministic(self, token_ids, h_t):
        prob, _ = self.forward(token_ids, h_t)
        return (prob > 0.5).long()
