"""Transformer world model V2 — block-causal + Block Teacher Forcing (BTF).

Predicts next-frame tokens given past frames + actions.

Sequence format (K context frames):
    [f0 (36 tokens)] [a0 (1 token)] ... [fK-1 (36)] [aK-1 (1)] [target (36)]

Block-causal attention: bidirectional within each frame/action block, causal across.
BTF: target tokens are predicted in parallel — they can see all context but NOT each
other. A learnable mask token replaces real target embeddings to prevent leaking.

References:
    - IRIS (Micheli et al., ICLR 2023): block-causal attention on VQ tokens
    - TWM Improved (ICML 2025): Block Teacher Forcing for parallel prediction
"""

import torch
import torch.nn as nn

# Tokens per frame (6x6 VQ-VAE grid)
TOKENS_PER_FRAME = 36


class WorldModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 1024,
        n_actions: int = 2,
        embed_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 8,
        context_frames: int = 8,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_actions = n_actions
        self.embed_dim = embed_dim
        self.context_frames = context_frames
        self.tokens_per_frame = TOKENS_PER_FRAME

        # Sequence length: K * (36 + 1) + 36
        self.seq_len = context_frames * (TOKENS_PER_FRAME + 1) + TOKENS_PER_FRAME

        # Embeddings
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.action_embed = nn.Embedding(n_actions, embed_dim)
        self.pos_embed = nn.Embedding(self.seq_len, embed_dim)

        # Learnable mask token for target positions (BTF)
        self.mask_token = nn.Parameter(torch.randn(embed_dim) * 0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Attention mask: block-causal context + BTF target
        self.register_buffer("attn_mask", self._build_mask())

        self._init_weights()

    def _build_mask(self):
        """Build block-causal + BTF attention mask.

        Context region: bidirectional within frame/action blocks, causal across.
        Target region: can see all context, cannot see any target position.

        Returns:
            mask: (seq_len, seq_len) bool — True = blocked.
        """
        S = self.seq_len
        K = self.context_frames
        TPF = TOKENS_PER_FRAME
        ctx_end = K * (TPF + 1)  # start of target block

        # Assign block index to each position
        block_idx = torch.zeros(S, dtype=torch.long)
        pos = 0
        for i in range(K):
            block_idx[pos:pos + TPF] = 2 * i        # frame i
            pos += TPF
            block_idx[pos] = 2 * i + 1               # action i
            pos += 1
        block_idx[pos:] = 2 * K                      # target frame

        # Block-causal: query can attend to same or earlier blocks
        # blocked if query_block < key_block
        mask = block_idx.unsqueeze(1) < block_idx.unsqueeze(0)

        # BTF override: target positions cannot see each other
        mask[ctx_end:, ctx_end:] = True

        return mask

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, frame_tokens, actions):
        """Forward pass.

        Args:
            frame_tokens: (B, K+1, 36) long — K context frames + 1 target frame.
                          Target frame is ignored (BTF uses mask tokens).
            actions: (B, K) long — action for each context frame.

        Returns:
            logits: (B, 36, vocab_size) — predictions for the target frame tokens.
        """
        B = frame_tokens.size(0)
        K = self.context_frames

        # Build context sequence
        parts = []
        for i in range(K):
            parts.append(self.token_embed(frame_tokens[:, i]))  # (B, 36, D)
            act = self.action_embed(actions[:, i])               # (B, D)
            parts.append(act.unsqueeze(1))                       # (B, 1, D)

        # Target positions: learnable mask token (BTF — no target info leakage)
        parts.append(self.mask_token.expand(B, TOKENS_PER_FRAME, -1))

        x = torch.cat(parts, dim=1)  # (B, seq_len, D)

        # Add positional embeddings
        x = x + self.pos_embed(torch.arange(self.seq_len, device=x.device))

        # Transformer blocks
        for block in self.blocks:
            x = block(x, self.attn_mask)
        x = self.ln_f(x)

        # Logits at target positions (last 36) — direct prediction (no shift needed)
        logits = self.head(x[:, -TOKENS_PER_FRAME:])  # (B, 36, vocab_size)

        return logits

    @torch.no_grad()
    def predict_next_frame(self, frame_tokens, actions):
        """Predict next frame tokens in a single forward pass (BTF).

        Args:
            frame_tokens: (B, K, 36) long — K context frames.
            actions: (B, K) long — actions for context frames.

        Returns:
            predicted: (B, 36) long — predicted next frame tokens.
        """
        B = frame_tokens.size(0)
        # Dummy target frame (ignored — mask token used instead)
        dummy = torch.zeros(B, 1, TOKENS_PER_FRAME, dtype=torch.long,
                            device=frame_tokens.device)
        full_frames = torch.cat([frame_tokens, dummy], dim=1)
        logits = self.forward(full_frames, actions)
        return logits.argmax(dim=-1)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask):
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask)
        x = x + h
        x = x + self.mlp(self.ln2(x))
        return x
