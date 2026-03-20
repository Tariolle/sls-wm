# DeepDash Version History

## V0 -- Ha & Schmidhuber Baseline (2026-03-03)

Starting point: the World Models architecture (Ha & Schmidhuber, 2018).

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

Full architecture redesign, cherry-picking techniques from recent world model papers.

| Component | V0 | V1 | Why |
|-----------|----|----|-----|
| **V** | beta-VAE (continuous) | FSQ-VAE [8,5,5,5] (discrete) | Sharp reconstructions, no codebook collapse, 100% utilization |
| **M** | MDN-RNN | Transformer 256d/8H/8L | Discrete token classification, 51x compression, spatial attention |
| **C** | Linear + CMA-ES | CNNPolicy + BC + PPO | Spatial reasoning on 8x8 grid, BC pretraining for signal |
| **Input** | 64x64 RGB | 64x64 Sobel grayscale, square crop | Noise filtering, UI removal |
| **Decoding** | N/A (continuous) | Parallel single-step (all 65 tokens) | Real-time compatible |

### V1 sources
- FSQ-VAE tokenizer (from **FSQ**, Mentzer et al. 2023)
- Transformer world model on discrete tokens (from **IRIS**, Micheli et al. 2023)
- Parallel single-step decoding (from **IRIS**, simplified by dropping MaskGIT iterative decoding)
- AC-CPC contrastive loss (from **TWISTER**, Burchert et al. 2025)
- 3D-RoPE (from **V-JEPA 2**, Bardes et al. 2025)
- PPO controller training (from **PPO**, Schulman et al. 2017)

### V1 techniques
- FSQ-structured label smoothing (novel): Gaussian kernel over FSQ distance, sigma=0.9
- Dual token noise (novel): random 5% + FSQ neighbor 5%
- Vertical-only shift augmentation: 5x data multiplier
- Death token as 65th position: death prediction = token classification

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

## V2 -- Complete (2026-03-18 to 2026-03-20)

### Changes applied

#### Vision (V) / FSQ-VAE
- Retrained on global episode-level split + expert episodes. Val recon 2.49 (vs V1 2.57).

#### Transformer (M)
- Removed masking entirely (all tokens predicted, no ground truth leakage)
- AC-CPC weight 0.1 -> 0.5
- torch.compile full model (static graph)
- Val acc 34.2% (vs V1 36.1%), death F1 0.76 (vs V1 0.72), train/val gap halved (1.7% vs 3.7%)

#### Controller (C) / PPO
- Jump penalty 0.2/jump, BC class weight 1.5x (was 2.6x)
- Percentile-based advantage normalization (EMA 5th-95th)
- EMA target critic (decay 0.98)
- Constant LR, 15K iterations
- BC val acc 83.6% (vs V1 78%), PPO best eval 24.68 (vs V1 23.04), jump ratio 0.27 (vs V1 0.43)

### V2 results
- **Level 1 progress**: 12% (vs V1 10%)
- Less random jumping, but timing still off by 1-2 frames
- World model understands jump orbs/pads, controller ignores them (representation exists, PPO signal insufficient)
- GPU Sobel in deployment: 7ms (was 10ms CPU)

### What worked
- Masking removal: halved train/val gap, more honest metrics, no quality loss
- Jump penalty (0.2) + lower BC weight (1.5x): jump ratio 0.43 -> 0.27, less over-jumping
- Percentile advantage normalization + EMA critic: stable training, no NaN
- Strict data split: proper train/val separation across all models
- AC-CPC 0.5: death F1 improved (0.72 -> 0.76), dream quality visibly better

### What didn't work
- **Soft continuation gating**: agent exploited airtime (jump once, idle while airborne for full reward). Reverted to hard death cutoff.
- **Symlog discrete value prediction**: skipped, +1/step reward doesn't need it
- **Resume state loss**: percentile normalizer + EMA critic reset on resume caused artificial eval jump. Fixed by saving/restoring in checkpoint.
- **Eval context RNG**: shared with training RNG, changed on resume. Fixed with dedicated eval RNG.
- **FSQ neighbor substitution ablation**: not run, deprioritized

### V2 known issues
- **Jump timing**: main bottleneck. Off by 1-2 frames at 30 FPS.
- **Dream/reality gap**: policy learned in dream doesn't fully transfer.
- **Controller ignores game mechanics**: h_t encodes jump orbs/pads but PPO doesn't learn to use them.

### V2 next experiments
- **AC-CPC weight 1.0** (in progress)
- **Longer rollouts** (30 -> 45 steps)
- **PMPO** (Dreamer 4): use sign(advantage) instead of magnitude
- **Multi-token prediction** (Dreamer 4): predict 8 actions ahead from h_t for better timing

---

## V3 -- Ideas

Goal: maximize model quality while fitting within the 30 FPS inference window.

#### Vision (V) / FSQ-VAE
| Idea | Description |
|------|-------------|
| **Asymmetric encoder/decoder** | Lighter encoder (1 ResBlock/stage), heavier decoder (3 ResBlocks/stage). Faster deploy since only encoder runs at inference. |
| **FSQ levels sweep** | Try [8,8,5,5]=1600 or [10,5,5,4]=1000 to redistribute capacity across dimensions. |
| **Larger input grid** | 16x16 instead of 8x8 for finer spatial resolution. Better capture of small game mechanics (jump pads, orbs). |

#### Transformer (M)
| Idea | Description |
|------|-------------|
| **Scale up model** | Increase embedding dim (256 -> 384/512) and/or depth (8 -> 12 layers). Push model size to the edge of 30 FPS budget. |
| **Distill to smaller model** | Train large, distill to deployment-sized model. Best of both worlds. |
| **Separate space/time attention** (Dreamer 4) | 3 space-only + 1 temporal layer, repeated. Space layers skip KV cache from prior frames. More relevant at larger grid or longer context. |
