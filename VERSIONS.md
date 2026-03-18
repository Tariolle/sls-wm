# DeepDash Version History

## V0 -- Ha & Schmidhuber Baseline (2026-03-03)

Starting point: faithful reproduction of the World Models architecture (Ha & Schmidhuber, 2018).

| Component | Design |
|-----------|--------|
| **V** | beta-VAE, continuous Gaussian latents, 64x64 RGB input |
| **M** | MDN-RNN (LSTM + Mixture Density Network), models stochastic dynamics |
| **C** | Linear controller optimized with CMA-ES |

### Limitations discovered
- beta-VAE reconstructions fundamentally blurry (posterior averaging). Spikes indistinguishable from blocks.
- MDN unnecessary: Geometry Dash physics are fully deterministic, visual stochasticity is noise.
- CMA-ES converged to local optimum (~12 steps survival).
- RGB input wastes capacity on particles, backgrounds, decorations.

---

## V1 -- Current (2026-03-03 to 2026-03-17)

Full architecture redesign. Every component replaced.

| Component | V0 | V1 | Why |
|-----------|----|----|-----|
| **V** | beta-VAE (continuous) | FSQ-VAE [8,5,5,5] (discrete) | Sharp reconstructions, no codebook collapse, 100% utilization |
| **M** | MDN-RNN | Transformer 256d/8H/8L | Discrete token classification, 51x compression, spatial attention |
| **C** | Linear + CMA-ES | CNNPolicy + BC + PPO | Spatial reasoning on 8x8 grid, BC pretraining for signal |
| **Input** | 64x64 RGB | 64x64 Sobel grayscale, square crop | Noise filtering, UI removal |
| **Decoding** | N/A (continuous) | Parallel single-step (all 65 tokens) | Real-time compatible |

### V1 novel techniques
- FSQ-structured label smoothing (Gaussian kernel over FSQ distance, sigma=0.9)
- Dual token noise (random 5% + FSQ neighbor 5%)
- Vertical-only shift augmentation (5x data multiplier)
- Death token as 65th position (death prediction = token classification)
- AC-CPC contrastive loss (from TWISTER)
- 3D-RoPE (row, col, frame axes)

### V1 results
- Transformer: 36.06% val acc, death F1 0.72
- Controller: 19.86 -> 23.04 eval survival (9K PPO iters)
- Deployment: 27ms avg, 30 FPS real-time
- Dodges simple obstacles, fails on complex sequences

### V1 known issues
- **Train/inference mismatch**: partial masking during training (cosine schedule), full masking at inference. Model can peek at unmasked true tokens during training but gets none at inference.
- **PPO signal is slow**: first 2K iterations are near-random noise even with BC.
- **PPO cosine LR schedule**: decayed too early, caused plateau before 9K iters.
- **Dream quality**: world model still predicts death too aggressively near obstacles.
- **Controller**: jumps too much/too early on complex obstacles.

See [experiments/v1/](experiments/v1/) for full logs and hyperparameters.

---

## V2 -- Planned

### Changes from V1

#### Transformer (M)
| Change | Motivation |
|--------|-----------|
| **Remove masking entirely** | Fix train/inference mismatch. Delete mask_embed, cosine schedule, partial masking code. Always predict all tokens. |
| **Increase AC-CPC weight** (0.1 -> sweep 0.5, 1.0) | Richer h_t representations. Complementary with masking removal. |
| **torch.compile full model** | Static graph now possible with masking removed. |

#### Controller (C) / PPO
| Change | Motivation |
|--------|-----------|
| **Soft continuation gating** | Use existing death token probability as c_t = 1 - death_prob. Multiply into returns: R = r + gamma * c_t * V. Replaces hard threshold cutoff. |
| **Percentile-based advantage normalization** | EMA of 5th-95th percentile range instead of mean/std. Prevents outlier returns from dominating. |
| **EMA target critic** (decay 0.98) | Stabilizes value learning during imagination. |
| **Symlog discrete value prediction** (255 bins, two-hot) | Scale-robust critic. Used by DreamerV3 and TWISTER. Low impact with +1/step reward but doesn't hurt. |
| **Constant LR for PPO** | V1 cosine schedule decayed too early, caused plateau before 9K iters. Switch to constant LR. |
| **Longer PPO training** | V1 only ran 9K iters. Need 20K+ to evaluate properly. |

### V2 goals
- Consistent train/inference behavior (no masking mismatch)
- Stronger h_t representations via AC-CPC
- Smooth dream termination via soft continuation gating
- More stable PPO via percentile advantages + EMA critic
- Better dream quality, higher survival on complex obstacle sequences
