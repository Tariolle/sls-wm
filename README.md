# DeepDash

**World Models for Geometry Dash** -- Train a controller entirely in imagination, deploy at 30 FPS on the real game via screen capture.

## Architecture

![V -> M -> C Pipeline](docs/architecture_pipeline.png)

| Component | Model | Params | Function |
|-----------|-------|--------|----------|
| **V** (Vision) | FSQ-VAE [8,5,5,5] | 1.4M | 64x64 Sobel frame -> 8x8 discrete tokens (1000 codes) |
| **M** (Memory) | Transformer 256d/8H/8L | 6.7M | Predicts next tokens + death, produces h_t |
| **C** (Controller) | CNNPolicy | 40K | Token grid + h_t -> jump/idle |

## Key Results (V1)

| Metric | Value |
|--------|-------|
| Transformer val accuracy | 36.06% |
| Death prediction F1 (val) | 0.72 |
| PPO eval survival | 19.86 -> 23.04 (9K iters) |
| Inference latency | 27ms avg (30 FPS) |
| Deployment | Dodges simple obstacles, struggles with complex sequences |

## Novel Contributions

- **FSQ-structured label smoothing**: Gaussian kernel over FSQ coordinate distance instead of uniform smoothing. +0.81pp val acc, -31% CPC loss.
- **Real-time World Models deployment**: Screen capture agent on a real game at 30 FPS (no game API).

## Pipeline

```
1. Record gameplay    ->  data/death_episodes/, data/expert_episodes/
2. Train FSQ-VAE      ->  checkpoints/fsq_best.pt
3. Tokenize episodes  ->  tokens.npy per episode
4. Train Transformer  ->  checkpoints/transformer_best.pt
5. BC pretrain        ->  checkpoints/controller_bc_best.pt
6. PPO finetune       ->  checkpoints/controller_ppo_best.pt
7. Deploy             ->  python scripts/deploy.py
```

## Data

- **Death episodes**: ~3,600 episodes, ~179K frames (intentional deaths at obstacles)
- **Expert episodes**: ~36 clean runs, ~33K frames (no deaths, for BC + world model rebalancing)
- Global episode-level train/val split shared across all models (`deepdash/data_split.py`)

## Training Details

### FSQ-VAE
- RMSE 0.025/pixel, 100% codebook utilization
- GRWM regularization, shift augmentation, cosine LR, 200 epochs on A100

### Transformer
- Block-causal attention + 3D-RoPE + AC-CPC (TWISTER)
- Focal loss + structured label smoothing (sigma=0.9) + dual token noise
- Vertical-only shift augmentation (5x), death oversample 5x
- 200 epochs, LR 2e-3, batch 512

### Controller
- **BC**: death + expert episodes, class-weighted BCE (2.6x jumps), early stopping. Val acc 78%.
- **PPO**: uniform sampling (excluding last 2K frames), constant LR 1e-4, 512 episodes/iter, GAE (gamma=0.995, lambda=0.95)

### Deployment
- Screen capture (dxcam) -> Sobel (10ms) -> FSQ encode (4ms) -> Transformer h_t (9ms) -> Controller (1ms) -> keyboard input
- 30 FPS with ~6ms headroom

## What Worked / Didn't Work

See [experiments/v1/README.md](experiments/v1/README.md) for full V1 experiment archive.

## References

- **World Models**: Ha & Schmidhuber (2018). [arXiv:1803.10122](https://arxiv.org/abs/1803.10122)
- **IRIS**: Micheli et al. (2023). *Transformers are Sample-Efficient World Models*. [arXiv:2209.00588](https://arxiv.org/abs/2209.00588)
- **FSQ**: Mentzer et al. (2023). *Finite Scalar Quantization*. [arXiv:2309.15505](https://arxiv.org/abs/2309.15505)
- **TWISTER**: Burchert et al. (2025). *AC-CPC for World Models*. [arXiv:2503.04416](https://arxiv.org/abs/2503.04416)
- **DreamerV3**: Hafner et al. (2023). [arXiv:2301.04104](https://arxiv.org/abs/2301.04104)
- **PPO**: Schulman et al. (2017). [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
- **Label Smoothing**: Szegedy et al. (2016). CVPR
- **Focal Loss**: Lin et al. (2017). ICCV
