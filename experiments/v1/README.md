# V1 Experiment Archive

All training runs and results from the first 2-week sprint (2026-03-03 to 2026-03-17).

## Timeline

### Phase 1: Vision Model (V)
| Run | Model | Epochs | Key Result | Status |
|-----|-------|--------|------------|--------|
| beta-VAE | beta-VAE | 100 | Blurry reconstructions, no spatial structure | Abandoned |
| VQ-VAE | VQ-VAE | 200 | Sharp but codebook collapse issues | Abandoned |
| FSQ-VAE | FSQ-VAE [8,5,5,5] | 200 | RMSE 0.025, 100% utilization | **Final** |

### Phase 2: World Model (M)
| Run | Config | Epochs | Val Acc | Death Metric | Notes |
|-----|--------|--------|---------|-------------|-------|
| Initial | 256d/8H/8L, LR 1e-3, 15x death, h+v shifts | 200 | 33.66% | acc 97.6% | First good model |
| Overfitted | 256d/8H/8L, LR 2e-3, 15x death, h+v shifts | 400 | 34.69% | P=48% R=92% F1=64% | Death precision gap 44pp |
| **V1 Final** | 256d/8H/8L, LR 2e-3, 5x death, v-only shifts | 200 | 36.06% | P=61% R=87% F1=72% | Best balanced |

### Phase 3: Controller (C)

#### Failed approaches (all ~12 step survival, no trend)
1. CMA-ES (linear 257 params) -- sigma diverged
2. REINFORCE (h_t only, h_t+death_prob, h_t+z_t) -- no signal
3. BC + REINFORCE (trimmed death episodes) -- no signal
4. DART-style ViT actor-critic (670K params) -- no signal
5. PPO + auto-entropy -- NaN crash
6. PPO pure RL (CNNPolicy, 512 episodes, near-obstacle) -- flat at 12 steps, 833 iters

#### Working approach
| Stage | Config | Result |
|-------|--------|--------|
| BC pretrain | Death+expert episodes, class-weighted BCE (2.6x), early stop | 78% val acc |
| PPO finetune | 9K iters, uniform sampling, constant LR 1e-4, 512 eps | 19.86 -> 23.04 eval survival |

### Deployment
- Real-time at 30 FPS (27ms avg inference: 10ms Sobel, 9ms Transformer, 4ms FSQ, 1ms Controller)
- Dodges simple/double spikes inconsistently (~75% success)
- Fails on complex obstacles (jumps too much / too early)

## Key Findings
- Death oversampling 15x caused massive overfitting (44pp precision gap). 5x is better.
- Horizontal shift augmentation was invalid (player X is fixed). Vertical-only.
- MaskGIT iterative decoding degraded dream quality on 64-token grid. Single parallel step best.
- BC alone learns "always idle" without class weighting.
- PPO needs 2000+ iterations minimum to show signal.
- Near-obstacle sampling was bugged (context frames counted in death window).

## Hyperparameters (V1 Final)
```
FSQ-VAE: levels [8,5,5,5], 1000 codes, 8x8 grid, 64 tokens/frame
Transformer: 256d, 8 heads, 8 layers, 6.7M params
  - LR 2e-3 cosine -> 1e-4, batch 512, 200 epochs
  - Death oversample 5x, vertical shifts [-4,-2,0,2,4]
  - Token noise 5%, FSQ noise 5%, label smoothing 0.1 (sigma 0.9)
  - Focal loss gamma 2.0, AC-CPC weight 0.1
Controller: CNNPolicy 40K params
  - Embed(1000,16) -> Conv(32) -> Conv(64) -> 256d + h_t(256) -> actor+critic
  - BC: LR 1e-3, AdamW, class weight 2.6x, patience 10
  - PPO: LR 1e-4 constant, gamma 0.995, lambda 0.95, clip 0.2, 4 epochs
```
