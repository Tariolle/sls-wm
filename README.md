# SLS-WM: Structured Label Smoothing for Discrete World Models

Official implementation of **SLS-WM**, a world model architecture that introduces *FSQ-Structured Label Smoothing*: a topology-aware training objective for discrete latent predictors. The method replaces uniform label smoothing with a Gaussian kernel defined over the metric structure of the Finite Scalar Quantization (FSQ) codebook, weighted by per-dimension visual sensitivity.

**DeepDash** is the reinforcement learning environment suite developed to evaluate SLS-WM. It targets Geometry Dash, a deterministic platformer with binary actions (jump/idle) and precision timing constraints, where a controller is trained entirely in imagination and deployed at 30 FPS on the real game via screen capture.

## Contributions

### Primary: FSQ-Structured Label Smoothing

Standard label smoothing redistributes probability mass uniformly across the vocabulary, treating all incorrect tokens as equally wrong. In discrete latent spaces produced by FSQ, this assumption is incorrect: a token differing by one quantization level in a single dimension reconstructs to a near-identical image patch, while a distant token produces an unrelated output.

SLS-WM addresses this by defining a structured target distribution:

$$
q(j \mid i) = \begin{cases}
1 - \varepsilon & \text{if } j = i \\
\varepsilon \cdot \dfrac{w(i, j)}{\sum_{k \neq i} w(i, k)} & \text{otherwise}
\end{cases}
$$

where the kernel weight is a Gaussian over weighted squared Euclidean distance in FSQ coordinate space:

$$
w(i, j) = \exp\!\left(-\frac{\sum_d \alpha_d \cdot (\Delta_d)^2}{2\sigma^2}\right)
$$

The per-dimension weights $\alpha_d$ are calibrated from empirical patch-level MSE analysis (800 samples per perturbation type), measuring the actual visual impact of each FSQ dimension. This formulation generalizes to any discrete latent space equipped with a coordinate metric.

### Secondary: Real-Time World Model Deployment

A complete inference pipeline achieving 30 FPS real-time play on a live game, with no game API access. The pipeline uses GPU-resident token ring buffers, CUDA Graphs, and `torch.compile` to maintain sub-33ms latency per frame.

## Architecture

![V -> M -> C Pipeline](docs/architecture_pipeline.png)

| Component | Model | Params | Function |
|-----------|-------|--------|----------|
| **V** (Vision) | FSQ-VAE [8,5,5,5] | 1.9M (0.9M encoder) | 64x64 Sobel frame to 8x8 discrete tokens (1000 codes) |
| **M** (Memory) | Transformer 512d/8H/8L + AdaLN + QK-norm | ~35M | Predicts next tokens + death, produces h_t |
| **C** (Controller) | CNNPolicy + MTP | ~40K | Token grid + h_t to jump/idle (+ 8-step action prediction) |

## Results (V3)

| Metric | V1 | V2 | V3 (current) |
|--------|-----|-----|-------------|
| Transformer params | 6.7M | 6.7M | 14.7M |
| Val accuracy | 36.1% | 34.2% | 35.6% |
| Death F1 (val) | 0.72 | 0.73 | 0.78 |
| BC val acc | 78% | 83.6% | 87.1% |
| Level 1 progress | 10% | 11% | 20% |
| Inference | 27ms | 24ms | ~27ms |

## Pipeline

```
1. Record gameplay       ->  data/death_episodes/, data/expert_episodes/
2. Shift augmentation    ->  5x vertical shifts per episode
3. Train FSQ-VAE         ->  checkpoints/fsq_best.pt
4. Tokenize episodes     ->  tokens.npy per episode
5. Train Transformer     ->  checkpoints/transformer_best.pt
6. BC pretrain           ->  checkpoints/controller_bc_best.pt
7. PPO finetune          ->  checkpoints/controller_ppo_best.pt
8. Deploy                ->  python scripts/deploy.py
```

## Data

- **Death episodes**: 4,229 episodes, ~218K frames across 16 levels (intentional deaths at obstacles)
- **Expert episodes**: 36 clean runs, ~34K frames (no deaths, for BC + world model rebalancing)
- **Total**: 4,265 episodes, ~252K frames
- Global episode-level train/val split (seed 42, stratified) shared across all models (`deepdash/data_split.py`)

## Training Details

### FSQ-VAE
- RMSE 0.025/pixel, 100% codebook utilization
- GRWM regularization, shift augmentation, cosine LR, 200 epochs on A100

### Transformer
- AdaLN-Zero action conditioning (DiT/LeWorldModel) + QK-norm (SD3/MMDiT)
- Block-causal attention + 3D-RoPE + AC-CPC weight 1.0 (TWISTER)
- Focal loss + structured label smoothing (sigma=0.9) + dual token noise
- Vertical-only shift augmentation (5x), death oversample 5x
- No masking (all target tokens predicted, no ground truth leakage)
- 512d embedding, 8 heads, 8 layers, dropout 0.15
- 200 epochs, LR 4e-3, batch 512

### Controller
- **BC**: death + expert episodes, class-weighted BCE (1.5x jumps), early stopping
- **PPO**: clipped surrogate + MTP auxiliary loss (8-step), jump penalty 0.2/jump, percentile-based advantage normalization, EMA target critic (0.98), 45-step dream rollouts, constant LR 1e-4

### Deployment
- Screen capture (dxcam), Sobel (7ms, GPU), FSQ encode (4ms), Transformer h_t (14ms), Controller (1ms), keyboard input
- GPU token ring buffer (no CPU round-trip), CUDA Graph for encode_context, pinned memory transfer
- torch.compile on all platforms (PyTorch 2.11+)
- 30 FPS real-time

## Version History

See [VERSIONS.md](VERSIONS.md) for full V0 through V3 evolution.

## References

- **FSQ**: Mentzer et al. (2024). [*Finite Scalar Quantization: VQ-VAE Made Simple*](https://arxiv.org/abs/2309.15505). ICLR
- **Label Smoothing**: Szegedy et al. (2016). [*Rethinking the Inception Architecture for Computer Vision*](https://arxiv.org/abs/1512.00567). CVPR
- **World Models**: Ha & Schmidhuber (2018). [*World Models*](https://arxiv.org/abs/1803.10122). NeurIPS
- **IRIS**: Micheli et al. (2023). [*Transformers are Sample-Efficient World Models*](https://arxiv.org/abs/2209.00588). ICLR
- **TWISTER**: Burchert et al. (2025). [*Transformer-based World Models with AC-CPC*](https://arxiv.org/abs/2503.04416). ICLR
- **LeWorldModel**: Charraut et al. (2026). [*Stable End-to-End Joint-Embedding Predictive Architecture from Pixels*](https://arxiv.org/abs/2603.19312). Preprint
- **DiT**: Peebles & Xie (2023). [*Scalable Diffusion Models with Transformers*](https://arxiv.org/abs/2212.09748). ICCV
- **SD3/MMDiT**: Esser et al. (2024). [*Scaling Rectified Flow Transformers for High-Resolution Image Synthesis*](https://arxiv.org/abs/2403.03206). ICML
- **DreamerV3**: Hafner et al. (2025). [*Mastering Diverse Domains through World Models*](https://arxiv.org/abs/2301.04104). Nature
- **Dreamer 4**: Hafner et al. (2025). [*Training Agents Inside of Scalable World Models*](https://arxiv.org/abs/2509.24527). Preprint
- **PPO**: Schulman et al. (2017). [*Proximal Policy Optimization Algorithms*](https://arxiv.org/abs/1707.06347). Preprint
- **Focal Loss**: Lin et al. (2017). [*Focal Loss for Dense Object Detection*](https://arxiv.org/abs/1708.02002). ICCV
