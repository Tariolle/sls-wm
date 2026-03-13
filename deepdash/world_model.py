"""Transformer world model V5 — block-causal + 3D-RoPE + death token + AC-CPC.

Predicts next-frame tokens given past frames + actions.
Death is represented as a token (not a separate head) — the 65th position
of each frame block predicts ALIVE or DEATH via the same cross-entropy loss.

Sequence format (K context frames, T tokens/frame):
    [f0 (T visual + 1 status)] [a0] ... [fK-1 (T+1)] [aK-1] [target (T+1)]

Block-causal attention: bidirectional within each frame/action block, causal across.
Target frame uses causal masking within. GPT-shift: position t-1 predicts token t.

3D-RoPE: Each token gets (row, col, frame_idx) coordinates. Head dimensions are
split into three bands with separate frequency bases — lower theta for spatial
axes (positions 0–7) to give meaningful angular variation at small distances.
Inspired by V-JEPA 2 (Bardes et al., 2025).

References:
    - IRIS (Micheli et al., ICLR 2023): block-causal attention on VQ tokens
    - TWISTER (Burchert et al., ICLR 2025): AC-CPC contrastive loss
    - Su et al., 2021: RoFormer / Rotary Position Embeddings
    - Bardes et al., 2025: V-JEPA 2 — 3D factored RoPE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_rope(x, cos, sin):
    """Apply rotary position embedding.

    Args:
        x: (B, n_heads, T, head_dim)
        cos: (T, head_dim // 2)
        sin: (T, head_dim // 2)
    Returns:
        Rotated tensor, same shape as x.
    """
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, d)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class WorldModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 1000,
        n_actions: int = 2,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 6,
        context_frames: int = 4,
        dropout: float = 0.1,
        cpc_dim: int = 64,
        tokens_per_frame: int = 64,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_actions = n_actions
        self.embed_dim = embed_dim
        self.context_frames = context_frames
        self.tokens_per_frame = tokens_per_frame

        # Death token indices (appended as 65th position per frame)
        self.ALIVE_TOKEN = vocab_size      # index for alive status
        self.DEATH_TOKEN = vocab_size + 1  # index for death status
        self.full_vocab_size = vocab_size + 2  # visual tokens + ALIVE + DEATH

        # Tokens per frame block: visual tokens + 1 status token
        self.block_size = tokens_per_frame + 1

        # Sequence length: K * (block_size + 1 action) + block_size target
        self.seq_len = context_frames * (self.block_size + 1) + self.block_size

        # Embeddings (no absolute positional — using RoPE)
        self.token_embed = nn.Embedding(self.full_vocab_size, embed_dim)
        self.action_embed = nn.Embedding(n_actions, embed_dim)

        # 3D-RoPE precomputed frequencies
        head_dim = embed_dim // n_heads
        self.grid_size = int(tokens_per_frame ** 0.5)
        rope_cos, rope_sin = self._precompute_rope_3d(head_dim)
        self.register_buffer("rope_cos", rope_cos)
        self.register_buffer("rope_sin", rope_sin)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.embed_drop = nn.Dropout(dropout)
        self.head = nn.Linear(embed_dim, self.full_vocab_size, bias=False)
        # Weight tying: share embedding and output projection weights
        self.head.weight = self.token_embed.weight

        # AC-CPC: contrastive prediction of future hidden states
        self.cpc_dim = cpc_dim
        self.cpc_target_proj = nn.Linear(embed_dim, cpc_dim)
        self.cpc_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim + embed_dim, cpc_dim),
                nn.GELU(),
                nn.Linear(cpc_dim, cpc_dim),
            )
            for _ in range(context_frames)
        ])

        # Block-causal attention mask
        self.register_buffer("attn_mask", self._build_mask())

        self._init_weights()

    def _build_position_ids(self):
        """Build 3D position IDs (row, col, frame) for each sequence position.

        Visual tokens get their grid coordinates. Status and action tokens
        get the grid center as their spatial position (they summarise the
        whole frame, so no single spatial location is preferred).
        """
        S = self.seq_len
        K = self.context_frames
        BS = self.block_size
        G = self.grid_size
        center = (G - 1) / 2.0  # 3.5 for 8×8

        rows = torch.zeros(S)
        cols = torch.zeros(S)
        frames = torch.zeros(S)

        pos = 0
        for i in range(K):
            # Visual tokens (tokens_per_frame positions)
            for j in range(self.tokens_per_frame):
                rows[pos + j] = j // G
                cols[pos + j] = j % G
                frames[pos + j] = i
            # Status token
            rows[pos + self.tokens_per_frame] = center
            cols[pos + self.tokens_per_frame] = center
            frames[pos + self.tokens_per_frame] = i
            pos += BS
            # Action token
            rows[pos] = center
            cols[pos] = center
            frames[pos] = i
            pos += 1

        # Target frame
        for j in range(self.tokens_per_frame):
            rows[pos + j] = j // G
            cols[pos + j] = j % G
            frames[pos + j] = K
        rows[pos + self.tokens_per_frame] = center
        cols[pos + self.tokens_per_frame] = center
        frames[pos + self.tokens_per_frame] = K

        return rows, cols, frames

    def _precompute_rope_3d(self, head_dim, theta_spatial=10.0,
                            theta_temporal=10000.0):
        """Precompute 3D-RoPE cos/sin tables.

        Splits head_dim into three bands (row, col, time) and uses a
        lower base frequency for spatial axes so that positions 0–7
        produce meaningful angular variation.

        Args:
            head_dim: Dimension per attention head.
            theta_spatial: Base frequency for row/col axes (default 10).
            theta_temporal: Base frequency for frame axis (default 10000).

        Returns:
            cos, sin: (seq_len, head_dim // 2) each.
        """
        # Split head_dim into three even-sized bands
        d_row = (head_dim // 3) & ~1  # round down to nearest even
        d_col = d_row
        d_time = head_dim - d_row - d_col
        assert d_time > 0 and d_time % 2 == 0, (
            f"head_dim={head_dim} does not split cleanly into 3 even bands"
        )

        row_freqs = 1.0 / (theta_spatial ** (
            torch.arange(0, d_row, 2, dtype=torch.float32) / d_row))
        col_freqs = 1.0 / (theta_spatial ** (
            torch.arange(0, d_col, 2, dtype=torch.float32) / d_col))
        time_freqs = 1.0 / (theta_temporal ** (
            torch.arange(0, d_time, 2, dtype=torch.float32) / d_time))

        rows, cols, frames = self._build_position_ids()

        row_angles = torch.outer(rows, row_freqs)      # (S, d_row/2)
        col_angles = torch.outer(cols, col_freqs)       # (S, d_col/2)
        time_angles = torch.outer(frames, time_freqs)   # (S, d_time/2)

        angles = torch.cat([row_angles, col_angles, time_angles], dim=-1)
        return torch.cos(angles), torch.sin(angles)

    def _build_mask(self):
        """Build hybrid attention mask.

        Context frames: block-causal (bidirectional within frame block, causal across).
        Target frame: causal within the block.

        Returns:
            mask: (seq_len, seq_len) bool — True = blocked.
        """
        S = self.seq_len
        K = self.context_frames
        BS = self.block_size  # tokens_per_frame + 1 (status token)

        # Assign block index to each position
        block_idx = torch.zeros(S, dtype=torch.long)
        pos = 0
        for i in range(K):
            block_idx[pos:pos + BS] = 2 * i       # frame block i (visual + status)
            pos += BS
            block_idx[pos] = 2 * i + 1             # action i
            pos += 1
        block_idx[pos:] = 2 * K                    # target frame block

        # Block-causal: query can attend to same or earlier blocks
        mask = block_idx.unsqueeze(1) < block_idx.unsqueeze(0)

        # Override target block: causal within (not bidirectional)
        ctx_end = K * (BS + 1)
        target_size = BS
        for i in range(target_size):
            for j in range(i + 1, target_size):
                mask[ctx_end + i, ctx_end + j] = True

        return mask

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def _compute_cpc_loss(self, x, actions, temperature=0.1):
        """Compute AC-CPC contrastive loss over context frame hidden states."""
        K = self.context_frames
        BS = self.block_size
        B = x.size(0)

        if B < 2:
            return torch.tensor(0.0, device=x.device)

        # Action positions: after each frame block (block_size tokens)
        action_positions = [i * (BS + 1) + BS for i in range(K)]

        # Hidden states at action positions + target summary (last pos)
        h_steps = [x[:, pos] for pos in action_positions]
        h_steps.append(x[:, -1])

        z_targets = [F.normalize(self.cpc_target_proj(h.detach()), dim=-1)
                     for h in h_steps]

        act_embeds = self.action_embed(actions)  # (B, K, D)

        total_loss = 0.0
        n_pairs = 0

        for step_idx, k in enumerate(range(1, K + 1)):
            predictor = self.cpc_predictors[step_idx]
            for t in range(K + 1 - k):
                h_src = h_steps[t]
                end = min(t + k, K)
                if end <= t:
                    continue
                act_ctx = act_embeds[:, t:end].mean(dim=1)
                z_pred = predictor(torch.cat([h_src, act_ctx], dim=-1))
                z_pred = F.normalize(z_pred, dim=-1)
                z_pos = z_targets[t + k]
                sim = torch.mm(z_pred, z_pos.t()) / temperature
                labels = torch.arange(B, device=sim.device)
                total_loss += F.cross_entropy(sim, labels)
                n_pairs += 1

        return total_loss / max(n_pairs, 1)

    def forward(self, frame_tokens, actions):
        """Forward pass.

        Args:
            frame_tokens: (B, K+1, tokens_per_frame+1) long — K context + 1 target.
                          Last column is the status token (ALIVE or DEATH).
            actions: (B, K) long — action for each context frame.

        Returns:
            logits: (B, tokens_per_frame+1, full_vocab_size) — predictions for target.
            cpc_loss: scalar — AC-CPC contrastive loss.
        """
        B = frame_tokens.size(0)
        K = self.context_frames

        # Build interleaved sequence: [f0(65) a0 f1(65) a1 ... fK-1(65) aK-1 fK(65)]
        parts = []
        for i in range(K):
            parts.append(self.token_embed(frame_tokens[:, i]))    # (B, 65, D)
            act = self.action_embed(actions[:, i])
            parts.append(act.unsqueeze(1))                        # (B, 1, D)
        parts.append(self.token_embed(frame_tokens[:, K]))        # (B, 65, D) target

        x = torch.cat(parts, dim=1)  # (B, seq_len, D)
        x = self.embed_drop(x)

        for block in self.blocks:
            x, _ = block(x, self.attn_mask, self.rope_cos, self.rope_sin)
        x = self.ln_f(x)

        # GPT-style shift: predict target token t from position t-1
        predict_positions = x[:, -(self.block_size + 1):-1]  # (B, 65, D)
        logits = self.head(predict_positions)  # (B, 65, full_vocab_size)

        cpc_loss = self._compute_cpc_loss(x, actions)

        return logits, cpc_loss

    @staticmethod
    def _sample_token(logits, temperature, top_k, top_p):
        """Sample a token from logits with temperature, top-k, and top-p.

        Args:
            logits: (B, vocab) raw logits.
            temperature: Scaling factor. 0 = greedy (argmax).
            top_k: Keep only top-k logits. 0 = disabled.
            top_p: Nucleus sampling threshold. 0 = disabled.

        Returns:
            (B,) sampled token indices.
        """
        if temperature <= 0:
            return logits.argmax(dim=-1)

        logits = logits / temperature

        if top_k > 0:
            top_values = logits.topk(top_k, dim=-1).values
            logits[logits < top_values[..., -1:]] = float('-inf')

        if 0 < top_p < 1.0:
            sorted_logits, sorted_indices = logits.sort(dim=-1, descending=True)
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            # Remove tokens with cumulative probability above the threshold
            # (keep at least one token)
            remove = cumulative_probs - sorted_logits.softmax(dim=-1) >= top_p
            sorted_logits[remove] = float('-inf')
            logits = sorted_logits.scatter(-1, sorted_indices, sorted_logits)

        probs = logits.softmax(dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    @torch.no_grad()
    def predict_next_frame(self, frame_tokens, actions,
                           temperature=0.0, top_k=0, top_p=0.0,
                           return_hidden=False):
        """Predict next frame tokens + death autoregressively (KV-cached).

        Uses KV cache: one prefill pass over context, then single-token
        decode steps for the target frame.

        Args:
            frame_tokens: (B, K, tokens_per_frame+1) long — K context frames
                          with status tokens.
            actions: (B, K) long — actions for context frames.
            temperature: Sampling temperature. 0 = greedy (default).
            top_k: Keep only top-k logits. 0 = disabled.
            top_p: Nucleus sampling threshold. 0 = disabled.
            return_hidden: If True, also return the hidden state h_t (128d)
                          at the last context position (post layer-norm).

        Returns:
            predicted: (B, tokens_per_frame) long — predicted visual tokens.
            death_prob: (B,) float — probability of death.
            h_t: (B, embed_dim) float — only if return_hidden=True.
        """
        B = frame_tokens.size(0)
        TPF = self.tokens_per_frame
        K = self.context_frames
        device = frame_tokens.device

        # --- Phase 1: Prefill context ---
        # Build interleaved context: [f0(65) a0 f1(65) a1 ... fK-1(65) aK-1]
        parts = []
        for i in range(K):
            parts.append(self.token_embed(frame_tokens[:, i]))    # (B, 65, D)
            act = self.action_embed(actions[:, i])
            parts.append(act.unsqueeze(1))                        # (B, 1, D)
        x = torch.cat(parts, dim=1)  # (B, ctx_len, D)

        ctx_len = K * (self.block_size + 1)
        ctx_mask = self.attn_mask[:ctx_len, :ctx_len]
        rope_cos_ctx = self.rope_cos[:ctx_len]
        rope_sin_ctx = self.rope_sin[:ctx_len]

        past_kvs = []
        for block in self.blocks:
            x, kv = block(x, ctx_mask, rope_cos_ctx, rope_sin_ctx,
                          use_cache=True)
            past_kvs.append(kv)
        x = self.ln_f(x)
        h_t = x[:, -1] if return_hidden else None

        # First target token: predicted from last context position (GPT shift)
        logits = self.head(x[:, -1])  # (B, full_vocab_size)
        predicted = torch.zeros(B, self.block_size, dtype=torch.long,
                                device=device)
        predicted[:, 0] = self._sample_token(logits, temperature, top_k, top_p)

        # --- Phase 2: Decode target tokens 1..64 ---
        for t in range(self.block_size - 1):
            tok = self.token_embed(predicted[:, t:t + 1])  # (B, 1, D)

            pos = ctx_len + t
            rope_cos_t = self.rope_cos[pos:pos + 1]
            rope_sin_t = self.rope_sin[pos:pos + 1]

            h = tok
            new_kvs = []
            for i, block in enumerate(self.blocks):
                h, kv = block(h, None, rope_cos_t, rope_sin_t,
                              past_kv=past_kvs[i], use_cache=True)
                new_kvs.append(kv)
            past_kvs = new_kvs

            h = self.ln_f(h)
            logits = self.head(h[:, 0])  # (B, full_vocab_size)
            predicted[:, t + 1] = self._sample_token(
                logits, temperature, top_k, top_p)

        # Death probability from the final logits (status token prediction)
        death_prob = F.softmax(logits, dim=-1)[:, self.DEATH_TOKEN]

        if return_hidden:
            return predicted[:, :TPF], death_prob, h_t
        return predicted[:, :TPF], death_prob


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        self.ln1 = nn.LayerNorm(embed_dim)
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask, rope_cos, rope_sin,
                past_kv=None, use_cache=False):
        B, T, D = x.shape
        h = self.ln1(x)

        qkv = self.qkv(h).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_kv = (k, v) if use_cache else None

        # SDPA expects bool mask where True = attend (opposite of ours)
        if attn_mask is not None:
            sdpa_mask = ~attn_mask
        else:
            sdpa_mask = None

        drop_p = self.attn_drop.p if self.training else 0.0
        h = F.scaled_dot_product_attention(
            q, k, v, attn_mask=sdpa_mask, dropout_p=drop_p,
        ).transpose(1, 2).reshape(B, T, D)
        h = self.out_proj(h)

        x = x + self.resid_drop(h)
        x = x + self.mlp(self.ln2(x))
        return x, present_kv
