"""Transformer world model — block-causal + 3D-RoPE + death token + AC-CPC.

Predicts all next-frame tokens in parallel given past frames + actions.
All target positions use a learnable query embedding (no ground truth leakage).
Death is represented as a token (not a separate head) — the 65th position
of each frame block predicts ALIVE or DEATH via the same cross-entropy loss.

Sequence format (K context frames, T tokens/frame):
    [f0 (T visual + 1 status)] [a0] ... [fK-1 (T+1)] [aK-1] [target (T+1)]

Block-causal attention: bidirectional within each frame/action block, causal across.
Target frame is bidirectional within (all target tokens see each other).

3D-RoPE: Each token gets (row, col, frame_idx) coordinates. Head dimensions are
split into three bands with separate frequency bases — lower theta for spatial
axes (positions 0–7) to give meaningful angular variation at small distances.
Inspired by V-JEPA 2 (Bardes et al., 2025).

References:
    - IRIS (Micheli et al., ICLR 2023): block-causal attention on VQ tokens
    - TWISTER (Burchi & Timofte, ICLR 2025): AC-CPC contrastive loss
    - Su et al., 2021: RoFormer / Rotary Position Embeddings
    - Bardes et al., 2025: V-JEPA 2 — 3D factored RoPE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm).to(x.dtype) * self.weight


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
        embed_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 8,
        context_frames: int = 4,
        dropout: float = 0.1,
        cpc_dim: int = 64,
        tokens_per_frame: int = 64,
        adaln: bool = False,
        fsq_dim: int | None = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_actions = n_actions
        self.embed_dim = embed_dim
        self.context_frames = context_frames
        self.tokens_per_frame = tokens_per_frame
        self.adaln = adaln

        # Death token indices (appended as 65th position per frame)
        self.ALIVE_TOKEN = vocab_size      # index for alive status
        self.DEATH_TOKEN = vocab_size + 1  # index for death status
        self.full_vocab_size = vocab_size + 2  # visual tokens + ALIVE + DEATH

        # Tokens per frame block: visual tokens + 1 status token
        self.block_size = tokens_per_frame + 1

        if adaln:
            # No action tokens in sequence — actions injected via AdaLN
            self.seq_len = (context_frames + 1) * self.block_size
        else:
            # Sequence length: K * (block_size + 1 action) + block_size target
            self.seq_len = context_frames * (self.block_size + 1) + self.block_size

        # Embeddings (no absolute positional — using RoPE)
        self.token_embed = nn.Embedding(self.full_vocab_size, embed_dim)
        if not adaln:
            self.action_embed = nn.Embedding(n_actions, embed_dim)

        # Joint training gradient conduit (E6.1+): a learnable Linear that
        # contributes zero to the forward value but routes gradient from
        # transformer losses through STE back to the FSQ encoder. See
        # forward() for the zero-sum STE correction pattern. Only created
        # when fsq_dim is provided; None in V5 behavior.
        self.fsq_dim = fsq_dim
        if fsq_dim is not None:
            self.fsq_grad_proj = nn.Linear(fsq_dim, embed_dim)
        else:
            self.fsq_grad_proj = None

        # AdaLN: action conditioning MLP
        if adaln:
            self.action_cond = nn.Sequential(
                nn.Linear(n_actions * context_frames, embed_dim),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim),
            )

        # 3D-RoPE precomputed frequencies
        head_dim = embed_dim // n_heads
        self.grid_size = int(tokens_per_frame ** 0.5)
        rope_cos, rope_sin = self._precompute_rope_3d(head_dim)
        self.register_buffer("rope_cos", rope_cos)
        self.register_buffer("rope_sin", rope_sin)

        # Transformer blocks
        BlockClass = AdaLNTransformerBlock if adaln else TransformerBlock
        self.blocks = nn.ModuleList([
            BlockClass(embed_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.embed_drop = nn.Dropout(dropout)
        self.head = nn.Linear(embed_dim, self.full_vocab_size, bias=False)
        # Weight tying: share embedding and output projection weights
        self.head.weight = self.token_embed.weight

        # Learnable [MASK] embedding (not in vocab -- head never predicts it)
        self.mask_embed = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # AC-CPC: contrastive prediction of future hidden states
        self.cpc_dim = cpc_dim
        self.cpc_target_proj = nn.Linear(embed_dim, cpc_dim)
        self.cpc_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim + (k + 1) * n_actions, cpc_dim),
                nn.GELU(),
                nn.Linear(cpc_dim, cpc_dim),
            )
            for k in range(context_frames)
        ])

        # Block-causal attention mask
        self.register_buffer("attn_mask", self._build_mask())

        self._init_weights()

        # Zero-init AdaLN projections AFTER _init_weights
        # Weights=0, scale/shift biases=0: model starts as standard transformer.
        # Gate biases=1: full residual flow from the start (not dead blocks).
        if adaln:
            for block in self.blocks:
                linear = block.adaln_proj[-1]
                nn.init.zeros_(linear.weight)
                nn.init.zeros_(linear.bias)
                D = linear.out_features // 6
                nn.init.constant_(linear.bias[2*D:3*D], 1.0)  # gate1
                nn.init.constant_(linear.bias[5*D:6*D], 1.0)  # gate2

    def _backbone_forward(self, x, cond=None):
        """Run transformer blocks + final layernorm (compile-friendly hot path)."""
        for block in self.blocks:
            x, _ = block(x, self.attn_mask, self.rope_cos, self.rope_sin,
                         cond=cond)
        return self.ln_f(x)

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
            if not self.adaln:
                # Action token (not present in AdaLN mode)
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
        if self.adaln:
            for i in range(K):
                block_idx[pos:pos + BS] = i
                pos += BS
            block_idx[pos:] = K
        else:
            for i in range(K):
                block_idx[pos:pos + BS] = 2 * i       # frame block i
                pos += BS
                block_idx[pos] = 2 * i + 1             # action i
                pos += 1
            block_idx[pos:] = 2 * K                    # target frame block

        # Block-causal: query can attend to same or earlier blocks
        mask = block_idx.unsqueeze(1) < block_idx.unsqueeze(0)

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

        if self.adaln:
            # AdaLN: use status token (last position of each frame block)
            anchor_positions = [(i + 1) * BS - 1 for i in range(K)]
        else:
            # Standard: use action token positions
            anchor_positions = [i * (BS + 1) + BS for i in range(K)]

        # Hidden states at anchor positions + target summary (last pos)
        h_steps = [x[:, pos] for pos in anchor_positions]
        h_steps.append(x[:, -1])

        z_targets = [F.normalize(self.cpc_target_proj(h.detach()), dim=-1)
                     for h in h_steps]

        act_onehot = F.one_hot(actions, self.n_actions).float()  # (B, K, n_actions)

        total_loss = 0.0
        n_pairs = 0

        for step_idx, k in enumerate(range(1, K + 1)):
            predictor = self.cpc_predictors[step_idx]
            for t in range(K + 1 - k):
                h_src = h_steps[t]
                end = min(t + k, K)
                if end <= t:
                    continue
                act_ctx = act_onehot[:, t:end].reshape(B, -1)  # (B, k * n_actions)
                z_pred = predictor(torch.cat([h_src, act_ctx], dim=-1))
                z_pred = F.normalize(z_pred, dim=-1)
                z_pos = z_targets[t + k]
                sim = torch.mm(z_pred, z_pos.t()) / temperature
                labels = torch.arange(B, device=sim.device)
                total_loss += F.cross_entropy(sim, labels)
                n_pairs += 1

        return total_loss / max(n_pairs, 1)

    def forward(self, frame_tokens, actions, z_q_ste_context=None):
        """Forward pass: predict all target tokens from context.

        All target positions are replaced with mask_embed (no ground truth
        leakage). This matches inference exactly.

        Args:
            frame_tokens: (B, K+1, tokens_per_frame+1) long — K context + 1 target.
                          Last column is the status token (ALIVE or DEATH).
            actions: (B, K) long — action for each context frame.
            z_q_ste_context: optional (B, K, tokens_per_frame, fsq_dim) continuous
                STE-routed FSQ codes for context frames' VISUAL positions only.
                When provided (joint training), a zero-sum correction term
                `fsq_grad_proj(z_q_ste) - fsq_grad_proj(z_q_ste).detach()` is
                added to each context frame's visual-token embedding; the
                forward value is unchanged (term cancels to 0), but gradient
                flows from downstream losses through fsq_grad_proj and STE
                back to the FSQ encoder. Status position (index
                tokens_per_frame) receives no correction. When None, V5
                behavior is byte-identical.

        Returns:
            logits: (B, block_size, full_vocab_size) — predictions for all target positions.
            cpc_loss: scalar — AC-CPC contrastive loss.
        """
        B = frame_tokens.size(0)
        K = self.context_frames
        use_ste = z_q_ste_context is not None
        if use_ste and self.fsq_grad_proj is None:
            raise RuntimeError(
                "z_q_ste_context passed but fsq_grad_proj not initialized; "
                "construct WorldModel with fsq_dim=<int> for joint training."
            )

        def embed_ctx_frame(i):
            """Embed context frame i with optional zero-sum STE correction
            on visual positions. Status (last) position is unchanged."""
            hard = self.token_embed(frame_tokens[:, i])  # (B, TPF+1, D)
            if not use_ste:
                return hard
            z = z_q_ste_context[:, i]                     # (B, TPF, fsq_dim)
            corr_vis = self.fsq_grad_proj(z)              # (B, TPF, D)
            # Zero-sum: forward value = 0, backward grad routes through z
            corr_vis = corr_vis - corr_vis.detach()
            # Pad the status position with zeros (no z for that position)
            pad = torch.zeros(B, 1, self.embed_dim,
                              device=hard.device, dtype=hard.dtype)
            corr = torch.cat([corr_vis, pad], dim=1)      # (B, TPF+1, D)
            return hard + corr

        parts = []
        cond = None

        if self.adaln:
            # AdaLN: actions injected via conditioning, not in sequence
            act_onehot = F.one_hot(actions, self.n_actions).float()
            cond = self.action_cond(act_onehot.reshape(B, -1))  # (B, D)
            for i in range(K):
                parts.append(embed_ctx_frame(i))
            ctx_end = K * self.block_size
        else:
            # Standard: interleaved [f0(65) a0 f1(65) a1 ... fK-1(65) aK-1]
            for i in range(K):
                parts.append(embed_ctx_frame(i))
                act = self.action_embed(actions[:, i])
                parts.append(act.unsqueeze(1))
            ctx_end = K * (self.block_size + 1)

        # All target positions use mask_embed (no peeking at ground truth)
        target_embed = self.mask_embed.expand(B, self.block_size, -1)
        parts.append(target_embed)
        x = torch.cat(parts, dim=1)  # (B, seq_len, D)
        x = self.embed_drop(x)

        x = self._backbone_forward(x, cond=cond)

        # No GPT-shift: each target position predicts its own token
        predict_positions = x[:, ctx_end:ctx_end + self.block_size]  # (B, 65, D)
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
    def encode_context(self, frame_tokens, actions):
        """Encode context frames and return hidden state h_t.

        Runs only the prefill phase (no prediction). Used at inference
        when we only need h_t for the controller, not predicted tokens.

        Args:
            frame_tokens: (B, K, tokens_per_frame+1) long -- K context frames
                          with status tokens.
            actions: (B, K) long -- actions for context frames.

        Returns:
            h_t: (B, embed_dim) float -- hidden state at last context position.
        """
        K = self.context_frames
        parts = []
        cond = None

        if self.adaln:
            act_onehot = F.one_hot(actions, self.n_actions).float()
            cond = self.action_cond(act_onehot.reshape(actions.size(0), -1))
            for i in range(K):
                parts.append(self.token_embed(frame_tokens[:, i]))
            ctx_len = K * self.block_size
        else:
            for i in range(K):
                parts.append(self.token_embed(frame_tokens[:, i]))
                act = self.action_embed(actions[:, i])
                parts.append(act.unsqueeze(1))
            ctx_len = K * (self.block_size + 1)

        x = torch.cat(parts, dim=1)
        ctx_mask = self.attn_mask[:ctx_len, :ctx_len]
        rope_cos_ctx = self.rope_cos[:ctx_len]
        rope_sin_ctx = self.rope_sin[:ctx_len]

        for block in self.blocks:
            x, _ = block(x, ctx_mask, rope_cos_ctx, rope_sin_ctx,
                         use_cache=False, cond=cond)
        x = self.ln_f(x)
        return x[:, -1]  # h_t at last context position

    @torch.no_grad()
    def predict_next_frame(self, frame_tokens, actions,
                           temperature=0.0, top_k=0, top_p=0.0,
                           return_hidden=False):
        """Predict next frame via parallel token decoding.

        Phase 1: Prefill context with KV cache.
        Phase 2: Predict all target tokens in parallel (single forward pass).

        Args:
            frame_tokens: (B, K, tokens_per_frame+1) long -- K context frames
                          with status tokens.
            actions: (B, K) long -- actions for context frames.
            temperature: Sampling temperature. 0 = greedy (default).
            top_k: Keep only top-k logits. 0 = disabled.
            top_p: Nucleus sampling threshold. 0 = disabled.
            return_hidden: If True, also return the hidden state h_t (embed_dim)
                          at the last context position (post layer-norm).

        Returns:
            predicted: (B, tokens_per_frame) long -- predicted visual tokens.
            death_prob: (B,) float -- probability of death.
            h_t: (B, embed_dim) float -- only if return_hidden=True.
        """
        B = frame_tokens.size(0)
        TPF = self.tokens_per_frame
        K = self.context_frames
        device = frame_tokens.device
        n_tokens = self.block_size  # 65

        # --- Phase 1: Prefill context ---
        parts = []
        cond = None

        if self.adaln:
            act_onehot = F.one_hot(actions, self.n_actions).float()
            cond = self.action_cond(act_onehot.reshape(B, -1))
            for i in range(K):
                parts.append(self.token_embed(frame_tokens[:, i]))
            ctx_len = K * self.block_size
        else:
            for i in range(K):
                parts.append(self.token_embed(frame_tokens[:, i]))
                act = self.action_embed(actions[:, i])
                parts.append(act.unsqueeze(1))
            ctx_len = K * (self.block_size + 1)

        x = torch.cat(parts, dim=1)  # (B, ctx_len, D)
        ctx_mask = self.attn_mask[:ctx_len, :ctx_len]
        rope_cos_ctx = self.rope_cos[:ctx_len]
        rope_sin_ctx = self.rope_sin[:ctx_len]

        context_kvs = []
        for block in self.blocks:
            x, kv = block(x, ctx_mask, rope_cos_ctx, rope_sin_ctx,
                          use_cache=True, cond=cond)
            context_kvs.append(kv)
        x = self.ln_f(x)
        h_t = x[:, -1] if return_hidden else None

        # --- Phase 2: Parallel decode (all tokens in one forward pass) ---
        target_embed = self.mask_embed.expand(B, n_tokens, -1)
        full_rope_cos = self.rope_cos[:ctx_len + n_tokens]
        full_rope_sin = self.rope_sin[:ctx_len + n_tokens]

        h = target_embed
        for i, block in enumerate(self.blocks):
            h, _ = block(h, None, full_rope_cos, full_rope_sin,
                         past_kv=context_kvs[i], use_cache=False, cond=cond)
        h = self.ln_f(h)
        logits = self.head(h)  # (B, 65, vocab)

        # Visual positions (0-63): mask out status tokens so they can't be sampled
        logits[:, :TPF, self.ALIVE_TOKEN] = -float('inf')
        logits[:, :TPF, self.DEATH_TOKEN] = -float('inf')

        predicted = self._sample_token(
            logits.reshape(-1, logits.size(-1)),
            temperature, top_k, top_p,
        ).reshape(B, n_tokens)

        # Death probability from status token (position 64)
        death_prob = F.softmax(logits[:, -1], dim=-1)[:, self.DEATH_TOKEN]

        if return_hidden:
            return predicted[:, :TPF], death_prob, h_t
        return predicted[:, :TPF], death_prob


class AdaLNTransformerBlock(nn.Module):
    """Transformer block with Adaptive Layer Normalization for action conditioning.

    Instead of interleaving action tokens in the sequence, actions modulate
    each layer via scale/shift on the LayerNorms. Zero-initialized so the
    model starts as vanilla LayerNorm and gradually learns to use actions.

    Reference: Peebles & Xie, 2023 (DiT); LeWorldModel (Galilai group).
    """

    def __init__(self, embed_dim, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        # LayerNorm without learned affine (AdaLN provides scale/shift)
        self.ln1 = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        # QK-norm: RMSNorm per head prevents attention logit explosion (SD3/MMDiT)
        self.ln_q = RMSNorm(self.head_dim)
        self.ln_k = RMSNorm(self.head_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

        # AdaLN-Zero: (scale1, shift1, gate1, scale2, shift2, gate2)
        # Gates multiply sublayer output before residual addition.
        # At zero-init, gate=0 makes each block pure identity.
        # Reference: DiT (Peebles & Xie, 2023), LeWorldModel (Galilai).
        self.adaln_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 6 * embed_dim),
        )

    def forward(self, x, attn_mask, rope_cos, rope_sin,
                past_kv=None, use_cache=False, cond=None):
        B, T, D = x.shape

        # AdaLN-Zero modulation from conditioning vector.
        # Force float32 for scale/shift to prevent FP16 overflow in (1+scale)*x.
        mods = self.adaln_proj(cond.float()).float()  # (B, 6*D)
        scale1, shift1, gate1, scale2, shift2, gate2 = mods.chunk(6, dim=-1)
        scale1 = scale1.unsqueeze(1)  # (B, 1, D)
        shift1 = shift1.unsqueeze(1)
        gate1 = gate1.unsqueeze(1)
        scale2 = scale2.unsqueeze(1)
        shift2 = shift2.unsqueeze(1)
        gate2 = gate2.unsqueeze(1)

        h = (self.ln1(x).float() * (1 + scale1) + shift1).to(x.dtype)

        qkv = self.qkv(h).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        q, k = self.ln_q(q), self.ln_k(k)

        present_kv = (k, v) if use_cache else None

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
            q = apply_rope(q, rope_cos[-T:], rope_sin[-T:])
            k = apply_rope(k, rope_cos, rope_sin)
        else:
            q = apply_rope(q, rope_cos, rope_sin)
            k = apply_rope(k, rope_cos, rope_sin)

        if attn_mask is not None:
            sdpa_mask = ~attn_mask
        else:
            sdpa_mask = None

        drop_p = self.attn_drop.p if self.training else 0.0
        h = F.scaled_dot_product_attention(
            q, k, v, attn_mask=sdpa_mask, dropout_p=drop_p,
        ).transpose(1, 2).reshape(B, T, D)
        h = self.out_proj(h)

        x = x + gate1 * self.resid_drop(h)
        h2 = (self.ln2(x).float() * (1 + scale2) + shift2).to(x.dtype)
        x = x + gate2 * self.mlp(h2)
        return x, present_kv


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
                past_kv=None, use_cache=False, cond=None):
        B, T, D = x.shape
        h = self.ln1(x)

        qkv = self.qkv(h).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        # Cache raw (un-rotated) K/V so RoPE can be recomputed for
        # shifted positions when the context window slides.
        present_kv = (k, v) if use_cache else None

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
            # rope_cos/sin must cover all key positions (past + current).
            # Q uses only the last T positions (current queries).
            q = apply_rope(q, rope_cos[-T:], rope_sin[-T:])
            k = apply_rope(k, rope_cos, rope_sin)
        else:
            q = apply_rope(q, rope_cos, rope_sin)
            k = apply_rope(k, rope_cos, rope_sin)

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
