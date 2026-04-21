# SLS-WM
### Structured Label Smoothing for Joint-Embedding Discrete World Models

**Abstract:** Discrete world models tokenize observations with a learned quantizer and predict next-frame tokens with a transformer. Standard cross-entropy training treats every incorrect prediction as equally wrong, discarding a signal that Finite Scalar Quantization (FSQ) makes available by construction: each code sits at a point on an integer coordinate lattice, so some wrong predictions are near-misses (one quantization step away in one dimension, a near-identical reconstruction) while others are gross errors (opposite corner of the codebook, unrelated content). SLS-WM's contribution is *FSQ-Structured Label Smoothing* (SLS), a training objective that replaces uniform label smoothing with a kernel over the integer-lattice coordinates of the FSQ codebook, so a near-miss prediction is treated as a near-miss rather than a gross error. We train the FSQ encoder jointly with the dynamics transformer (prediction gradients flow back into the encoder through a straight-through estimator, so the codebook is shaped by both reconstruction and prediction) and study SLS under this setup. We evaluate on Geometry Dash, a deterministic platformer with binary actions (jump/idle) and precision timing, where the controller is trained entirely in imagination and deployed at 30 FPS on the real game via screen capture.

> **Status:** Pre-freeze research code. Model freeze target **2026-05-31**, NeurIPS 2026 workshop submission. Numerical results and ablation tables will land with V6 deployment; everything below describes concepts and architecture, not outcomes.

## Key ideas

**1. Joint FSQ + transformer training with a pixel anchor.** FSQ was originally introduced as a tokenizer for downstream tasks and is typically used in subsequent work as a frozen pre-processing stage. Dreamer-family models (DreamerV3, TWISTER) do train their encoders jointly with the dynamics model, but use categorical-VAE latents rather than FSQ. SLS-WM sits at the intersection: we use FSQ for its coordinate-lattice structure (which SLS then exploits) and train it jointly with the transformer. A learnable linear projection `fsq_grad_proj` sits between the encoder's continuous latents and the transformer's input embedding; it contributes zero in the forward path but carries prediction gradients back into the encoder under an STE. We keep a light pixel-reconstruction loss as an anchor: a pure-JEPA variant without pixel anchor collapses in our experiments. Our attempt at a marginal anti-collapse regularizer (a Cramér-von-Mises test against a factorized uniform target, adapted from SIGReg's continuous-Gaussian formulation in LeWorldModel to the discrete factorized FSQ setting) was insufficient in our experiments to prevent joint-distribution degeneracy when the prediction target comes from the online encoder. Finding an anti-codebook-collapse loss that lets us drop the pixel anchor is open work; for now recon stays.

**2. FSQ-Structured Label Smoothing.** Standard label smoothing redistributes mass uniformly across the vocabulary, treating all incorrect tokens as equally wrong. Under FSQ this is wrong: a token differing by one quantization level in a single dimension reconstructs to a near-identical image patch, while a distant token is an unrelated output. SLS replaces the uniform component with a kernel over FSQ coordinate distances:

$$
q(j \mid i) = \begin{cases}
1 - \varepsilon & \text{if } j = i \\
\varepsilon \cdot \dfrac{k(\|\mathbf{c}_i - \mathbf{c}_j\|)}{\sum_{l \neq i} k(\|\mathbf{c}_i - \mathbf{c}_l\|)} & \text{otherwise}
\end{cases}
$$

where $\mathbf{c}_i$ is the FSQ coordinate of code $i$ on the integer lattice. We compare three kernels (Gaussian $e^{-d^2/2\sigma^2}$, Laplace $e^{-d/\sigma}$, Cauchy $1/(1+d^2/\sigma^2)$) and find Laplace preferable: its cusp at $d=0$ concentrates mass on the target, while exponential (not polynomial) tail decay keeps distant neighbours from leaking meaningful probability. Under joint training the codebook drifts during the run, so any per-dimension sensitivity weights calibrated from a snapshot are stale; we use an isotropic kernel with bandwidth set by a zero-calibration first-neighbour rule ($\sigma = 1$ at integer lattice spacing). This formulation generalizes to any discrete latent space with a coordinate metric.

**3. Real-time deployment of a discrete-latent world model.** A complete inference pipeline with GPU-resident token ring buffers, CUDA Graphs, and `torch.compile` (mode `reduce-overhead`, `fullgraph=True`), demonstrating that a joint-embedding discrete world model can drive a real game at 30 FPS on consumer hardware without API access.

## Architecture

Three components, trained in two stages (world model first, then controller):

| Component | Model | Training |
|-----------|-------|----------|
| **V** (Vision) | FSQ-VAE over $64\times64$ Sobel edge maps, $[5,5,5,5]$ levels, $8\times8$ spatial grid (625-code vocabulary) | Joint with M: pixel MSE anchor + prediction gradient via STE |
| **M** (Memory) | Causal transformer with AdaLN-Zero action conditioning, QK-norm, 3D-RoPE (row, col, frame) | Joint with V: token CE + SLS-weighted targets, focal loss |
| **C** (Controller) | MLP policy on transformer hidden state $h_t$ | BC warm-start on expert demos, then RL in imagination |

The transformer predicts next-frame token logits; cross-entropy targets come from the online FSQ encoder. Per-group gradient clipping (encoder and transformer clipped independently at the same norm) prevents the pixel-MSE gradient from saturating the prediction-side budget during warm-up. A separate `scripts/fsq_sensitivity.py` probe measures per-dimension reconstruction sensitivity and cross-dimension coupling of any trained codebook, retained as an analysis tool, not a hyperparameter pipeline.

## Running the code

**Environment.** Conda, PyTorch 2.10, CUDA 12.6. `conda run -n <env> python -m pip install -r requirements.txt`.

**Train the world model (V + M jointly):**
```bash
python scripts/train_world_model.py --config configs/e6.8-recon-laplacesls.yaml
```

**Train the controller (BC warm-start, then PPO in imagination):**
```bash
python scripts/train_controller_bc.py  --config configs/e6.8-recon-laplacesls.yaml
python scripts/train_controller_ppo.py --config configs/e6.8-recon-laplacesls.yaml
```

**Deploy to the live game (screen capture, 30 FPS):**
```bash
python scripts/deploy.py --config configs/e6.8-recon-laplacesls.yaml
```

**Analyse the learned codebook** (kernel fit per family, anisotropy, coupling):
```bash
python scripts/fsq_sensitivity.py --checkpoint checkpoints_e6.8/fsq_best.pt --levels 5 5 5 5
```

Cluster launches (SLURM, A100): `sbatch slurm/train_world_model.sl`, `sbatch slurm/train_controller.sl`.

## Data

- **Death episodes**: 4,229 episodes, ~218K frames across 16 levels (intentional deaths at obstacles).
- **Expert episodes**: 36 clean runs, ~34K frames (no deaths, for BC + world-model rebalancing).
- Global episode-level train/val split (seed 42, stratified) shared across all models (`deepdash/data_split.py`).

## Roadmap

| Tag | Stage | Status |
|-----|-------|--------|
| V5 | Frozen-FSQ + transformer + calibrated-SLS baseline | shipped (tagged on `main`) |
| E6.4 | First joint training iteration (CWU anti-collapse) | done, strong representation signal |
| E6.5 | Pure-JEPA ablation (no pixel anchor) | done, confirmed marginal CWU insufficient for joint collapse |
| E6.7 | Joint training + isotropic Cauchy SLS | done, confirmed Cauchy tail over-smooths dream rollouts |
| E6.8 | Joint training + isotropic Laplace SLS (current) | running |
| V6 | Final architecture post E6.x sweep | pre-freeze, 2026-05-31 target |

## References

### World models
- Ha & Schmidhuber (2018). [World Models](https://arxiv.org/abs/1803.10122). NeurIPS.
- Micheli et al. (2023). [IRIS: Transformers are Sample-Efficient World Models](https://arxiv.org/abs/2209.00588). ICLR.
- Burchi & Timofte (2025). [TWISTER: Transformer World Models with AC-CPC](https://arxiv.org/abs/2503.04416). ICLR.
- Hafner et al. (2025). [DreamerV3](https://arxiv.org/abs/2301.04104). Nature.
- Hafner et al. (2025). [Dreamer 4](https://arxiv.org/abs/2509.24527). Preprint.
- Maes et al. (2026). [LeWorldModel: Stable End-to-End JEPA from Pixels](https://arxiv.org/abs/2603.19312). Preprint.

### Quantization
- Mentzer et al. (2024). [FSQ: Finite Scalar Quantization](https://arxiv.org/abs/2309.15505). ICLR.
- Vali et al. (2026). [iFSQ: Improving FSQ for Image Generation](https://arxiv.org/abs/2601.17124). ICLR.

### Training objectives
- Szegedy et al. (2016). [Rethinking Inception](https://arxiv.org/abs/1512.00567). CVPR.
- Lin et al. (2017). [Focal Loss](https://arxiv.org/abs/1708.02002). ICCV.

### Conditioning and attention
- Peebles & Xie (2023). [DiT](https://arxiv.org/abs/2212.09748). ICCV.
- Esser et al. (2024). [SD3 / MMDiT (QK-norm)](https://arxiv.org/abs/2403.03206). ICML.
- Su et al. (2024). [RoFormer / RoPE](https://arxiv.org/abs/2104.09864). Neurocomputing.

### Representation learning
- van den Oord et al. (2018). [CPC / InfoNCE](https://arxiv.org/abs/1807.03748).
- Wang & Isola (2020). [Alignment and Uniformity on the Hypersphere](https://arxiv.org/abs/2005.10242). ICML.
- Huh et al. (2023). [GRWM temporal slowness](https://arxiv.org/abs/2311.17009).

### RL
- Schulman et al. (2017). [PPO](https://arxiv.org/abs/1707.06347).

## Acknowledgments

A100 compute provided by MesoNet (CRIANN, Rouen) through the Representation Learning course at INSA Rouen Normandy. Gameplay data contributions from Maël Planchot. Geometry Dash is developed by RobTop Games.
