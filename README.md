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

The per-dimension weights $\alpha_d$ are calibrated from empirical patch-level MSE analysis over 5,000 validation frames, measuring the actual visual impact of a one-step perturbation in each FSQ dimension. This formulation generalizes to any discrete latent space equipped with a coordinate metric.

### Secondary: Real-Time World Model Deployment

A complete inference pipeline achieving 30 FPS real-time play on a live game, with no game API access. The pipeline uses GPU-resident token ring buffers, CUDA Graphs, and `torch.compile` to maintain sub-33ms latency per frame.

## Architecture

![V -> M -> C Pipeline](docs/architecture_pipeline.png)

| Component | Model | Params | Function |
|-----------|-------|--------|----------|
| **V** (Vision) | FSQ-VAE [5,5,5,5] | 1.9M (0.9M encoder) | 64x64 Sobel frame to 8x8 discrete tokens (625 codes) |
| **M** (Memory) | Transformer 512d/8H/8L + AdaLN + QK-norm | ~35M | Predicts next tokens + death, produces h_t |
| **C** (Controller) | MLPPolicy | ~265K | h_t to jump/idle (1-hidden-layer MLP) |

## Results (V5 headline)

V5 baseline = FSQ [5,5,5,5] + Transformer with calibrated SLS (`sigma=0.9`, `dim_weights=[1.02, 0.94, 0.83, 1.20]`, `epsilon=0.1`).

| Metric | V5 |
|--------|-----|
| Transformer params | ~35M |
| Val CPC | 0.3069 |
| Val accuracy | 0.3215 |
| Val death P/R/F1 | 0.666 / 0.885 / 0.760 |
| Controller params | ~265K |
| Controller / deployment | pending (controller gate run in progress) |

Full V0 -> V5 evolution with per-version decisions and ablation trails is in [VERSIONS.md](VERSIONS.md).

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
- 100% codebook utilization (625/625 active), 76% perplexity (exp(H)/V; uniform = 100%)
- iFSQ bounding (2σ(1.6z)−1 instead of tanh) for near-uniform bin utilization
- GRWM temporal slowness (0.1) + uniformity loss (0.01), shift augmentation, cosine LR, BF16, 200 epochs on A100

### Transformer
- AdaLN-Zero action conditioning (DiT/LeWorldModel) + QK-norm (SD3/MMDiT)
- Block-causal attention + 3D-RoPE + AC-CPC weight 1.0 (TWISTER)
- Focal loss (gamma=2.0) + structured label smoothing (sigma=0.9, eps=0.1, calibrated dim_weights) + dual token noise
- Vertical-only shift augmentation (5x), death oversample 4x
- No masking (all target tokens predicted, no ground truth leakage)
- 512d embedding, 8 heads, 8 layers, dropout 0.15
- 200 epochs, LR 4e-3, batch 512, BF16

### Training Throughput (A100)

Benchmarked 11 configurations (all `torch.compile` modes × precisions) with subprocess isolation to prevent CUDA allocator fragmentation. Winning config applied to all training scripts:

| Config | ms/step | Speedup |
|--------|---------|---------|
| Eager FP32 (baseline) | ~2036ms | 1.00x |
| Compile default + BF16 | ~317ms | 6.42x |
| **Compile reduce-overhead + BF16** | **~314ms** | **6.48x** |
| Compile max-autotune + BF16 | ~311ms | 6.55x |

`reduce-overhead` is our default -- best of both worlds: faster than `default` and without the ~10-minute autotune phase `max-autotune` adds at startup for only ~1% more steady-state throughput.

### Controller
- **BC**: death + expert episodes, class-weighted BCE (1.5x jumps), early stopping
- **PPO**: clipped surrogate, jump penalty 0.2/jump, percentile-based advantage normalization, EMA target critic (0.98), 45-step dream rollouts, constant LR 1e-4
- **BC ablation (V4)**: cold-start PPO converges to the same plateau as BC-initialized PPO but needs ~2,000 extra iterations (~6h on A100). BC itself takes ~5 min. ROI: 5 min saves 6 hours.

### Deployment
- Screen capture (dxcam), Sobel (7ms, GPU), FSQ encode (4ms), Transformer h_t (14ms), Controller (1ms), keyboard input
- GPU token ring buffer (no CPU round-trip), CUDA Graph for encode_context, pinned memory transfer
- torch.compile on all platforms (PyTorch 2.11+)
- 30 FPS real-time

## Version History

See [VERSIONS.md](VERSIONS.md) for full V0 through V5 evolution.

## References

- **FSQ**: Mentzer et al. (2024). [*Finite Scalar Quantization: VQ-VAE Made Simple*](https://arxiv.org/abs/2309.15505). ICLR
- **iFSQ**: Vali et al. (2026). [*iFSQ: Improving FSQ for Image Generation with 1 Line of Code*](https://arxiv.org/abs/2601.17124). ICLR
- **Label Smoothing**: Szegedy et al. (2016). [*Rethinking the Inception Architecture for Computer Vision*](https://arxiv.org/abs/1512.00567). CVPR
- **Striving for Simplicity**: Springenberg et al. (2015). [*Striving for Simplicity: The All Convolutional Net*](https://arxiv.org/abs/1412.6806). ICLR Workshop
- **World Models**: Ha & Schmidhuber (2018). [*World Models*](https://arxiv.org/abs/1803.10122). NeurIPS
- **IRIS**: Micheli et al. (2023). [*Transformers are Sample-Efficient World Models*](https://arxiv.org/abs/2209.00588). ICLR
- **TWISTER**: Burchi & Timofte (2025). [*Transformer-based World Models with AC-CPC*](https://arxiv.org/abs/2503.04416). ICLR
- **LeWorldModel**: Maes et al. (2026). [*Stable End-to-End Joint-Embedding Predictive Architecture from Pixels*](https://arxiv.org/abs/2603.19312). Preprint
- **DiT**: Peebles & Xie (2023). [*Scalable Diffusion Models with Transformers*](https://arxiv.org/abs/2212.09748). ICCV
- **SD3/MMDiT**: Esser et al. (2024). [*Scaling Rectified Flow Transformers for High-Resolution Image Synthesis*](https://arxiv.org/abs/2403.03206). ICML
- **FiLM**: Perez et al. (2018). [*FiLM: Visual Reasoning with a General Conditioning Layer*](https://arxiv.org/abs/1709.07871). AAAI
- **RoPE**: Su et al. (2024). [*RoFormer: Enhanced Transformer with Rotary Position Embedding*](https://arxiv.org/abs/2104.09864). Neurocomputing
- **CPC / InfoNCE**: van den Oord et al. (2018). [*Representation Learning with Contrastive Predictive Coding*](https://arxiv.org/abs/1807.03748). Preprint
- **GRWM (temporal slowness)**: Huh et al. (2023). [*Straightening Out the Frame-by-Frame Video Prediction Problem*](https://arxiv.org/abs/2311.17009). Preprint
- **Uniformity loss**: Wang & Isola (2020). [*Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere*](https://arxiv.org/abs/2005.10242). ICML
- **DreamerV3**: Hafner et al. (2025). [*Mastering Diverse Control Tasks through World Models*](https://arxiv.org/abs/2301.04104). Nature
- **Dreamer 4**: Hafner et al. (2025). [*Training Agents Inside of Scalable World Models*](https://arxiv.org/abs/2509.24527). Preprint
- **PPO**: Schulman et al. (2017). [*Proximal Policy Optimization Algorithms*](https://arxiv.org/abs/1707.06347). Preprint
- **Focal Loss**: Lin et al. (2017). [*Focal Loss for Dense Object Detection*](https://arxiv.org/abs/1708.02002). ICCV
