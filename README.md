# DeepDash

**Latent Dynamics & Temporal Sequence Control for Geometry Dash**

**Course:** Representation Learning

**Architecture:** World Models (FSQ-VAE + Transformer + Controller)

## 1. Project Overview

![V → M → C Architecture Pipeline](docs/architecture_pipeline.png)

DeepDash is a Deep Reinforcement Learning agent designed to master *Geometry Dash* levels directly from raw gameplay footage. Unlike standard RL approaches that map pixels directly to actions, DeepDash explicitly separates **visual perception** from **control policy**.

This project implements a modified "World Models" architecture, demonstrating how an agent can learn a compressed **latent representation** of the real game world and "dream" deterministic future states to optimize its trajectory.

## 2. Technical Architecture

The system is composed of three distinct neural networks trained sequentially:

### A. Vision Model (V) - *The Tokenizer*

* **Type:** Finite Scalar Quantization VAE (FSQ-VAE) (Mentzer et al., 2023).
* **Input:** Preprocessed grayscale Sobel edge maps ($64 \times 64 \times 1$) captured from real *Geometry Dash* gameplay.
* **Preprocessing Pipeline:**

| Raw frame (1920x1080) | Square crop (1032x1032) | Sobel edges (1032x1032) | Final input (64x64) |
|:---:|:---:|:---:|:---:|
| ![Original](docs/preprocessing_1_original.png) | ![Cropped](docs/preprocessing_2_cropped.png) | ![Sobel](docs/preprocessing_3_sobel.png) | ![Final](docs/preprocessing_4_final.png) |

  Raw 1080p footage is cropped to a 1032x1032 square at (660, 48) — bottom-aligned to discard the UI progress bar, player flush to the left edge, maximizing forward obstacle visibility. Sobel edge detection is applied at full resolution before downscaling to $64 \times 64$ with `cv2.INTER_AREA`. The UI is deliberately excluded to prevent the model from memorizing level layouts via the progress indicator, forcing it to learn **reactive dynamics** rather than positional lookup.

* **Function:** Tokenizes visual data into an $8 \times 8$ grid of discrete codes (64 tokens per frame, vocabulary of 1000 from FSQ levels $[8,5,5,5]$). Each token is a discrete symbol, not a continuous vector — enabling **51x compression** (638 bits vs 32,768 bits per frame).
* **Noise Filtering:** Real game footage contains high-frequency stochastic noise (particles, weather effects, visual polish). Sobel edge detection extracts only structural boundaries (platforms, spikes, player outline), and FSQ further discards sub-token noise by snapping each latent dimension to a fixed number of levels.
* **Why FSQ over VQ-VAE over beta-VAE:** A standard beta-VAE was extensively evaluated first (beta=0/0.1/1.0, cyclical annealing, MSE/L1/BCE, latent dims 32-64). All configurations produced fundamentally blurry reconstructions due to Gaussian posterior averaging — spikes were indistinguishable from blocks. For a precision rhythm game requiring pixel-accurate obstacle recognition, this is a dealbreaker. VQ-VAE's discrete codebook eliminates this blurriness but introduces codebook collapse risk, commitment cost tuning, and EMA update complexity. FSQ replaces the learned codebook with deterministic scalar quantization — each latent dimension is simply rounded to one of $L$ fixed levels. This guarantees 100% codebook utilization by construction, eliminates all auxiliary losses, and achieved lower reconstruction error.

### B. Memory Model (M) - *The Dynamics Learner*

* **Type:** Transformer with MaskGIT masked token prediction (parallel decoding at inference).
* **Input:** Sequence of 64 FSQ codes per frame + action token + status token, with block-causal attention (bidirectional within frames, causal across) and 3D-RoPE (row, col, frame axes with factored frequency bases, inspired by V-JEPA 2).
* **Function:** Predicts the next frame's 64 tokens given the current tokenized state and action — a classification task over vocabulary 1002 (1000 visual codes + ALIVE + DEATH status tokens), not continuous regression. Death is predicted via a dedicated **death token** appended as the 65th position of each frame block, turning death prediction into the same next-token classification task.
* **Decoding:** MaskGIT (Chang et al., 2022) — during training, a random cosine-scheduled subset of target tokens is masked and predicted. At inference, all tokens start masked and are iteratively unmasked in order of confidence over ~8 steps, enabling parallel decoding instead of 65 sequential autoregressive steps.
* **Training Losses:** Focal cross-entropy on masked next-frame tokens + AC-CPC contrastive loss (TWISTER, ICLR 2025) that predicts future hidden states conditioned on actions.
* **FSQ-structured label smoothing:** Standard label smoothing (Szegedy 2016) spreads probability mass uniformly across all wrong tokens — a ±1 FSQ neighbor (visually identical) receives the same target probability as a completely unrelated token. This is a misaligned objective: the model is equally penalized for predicting a semantically equivalent neighbor as for predicting visual nonsense. FSQ-structured label smoothing replaces the uniform distribution with a Gaussian kernel over squared FSQ coordinate distance: $w(a,b) = \exp(-d^2(a,b) / 2\sigma^2)$, where $d^2$ is the sum of squared per-dimension differences. Empirical validation showed that (1) ±1 in a single dimension produces near-identical reconstructions, (2) concentrated perturbations (+2 in one dim) cause more visual change than distributed ones (+1 in two dims), matching the squared distance's superlinear per-dimension penalty. The smoothing mass is thus concentrated on the ~6 immediate FSQ neighbors, giving the model explicit credit for "near-miss" predictions. Status tokens (ALIVE/DEATH) use hard targets — death prediction must be exact.
* **Regularization:**
  * **Spatial shift augmentation:** Episodes are re-tokenized through the frozen FSQ-VAE at multiple pixel offsets ({-4,-2,0,2,4} × {-3,0,3} with edge padding), creating 15 tokenization variants per episode (~15× data multiplier). Horizontal shifts are physically equivalent to a different camera scroll position.
  * **Dual token noise:** Random token replacement (5%) acts as context-forcing dropout, while **FSQ neighbor substitution** (5%) replaces tokens with ±1 neighbors in the FSQ mixed-radix space. FSQ neighbors decode to visually near-identical patches but are distinct token indices — this forces the Transformer's embedding layer to learn that geometrically adjacent codes are semantically equivalent, injecting the codebook's topological structure as an inductive bias. The two noise types are complementary: random noise forces global robustness, neighbor noise smooths the embedding manifold.
  * Dropout, weight decay, death frame oversampling (15×).
* **AC-CPC Lineage:** AC-CPC extends Contrastive Predictive Coding (CPC, Oord et al., 2018), which pioneered the idea of predicting future representations in latent space rather than reconstructing raw inputs. AC-CPC adds **action conditioning** — predicting future hidden states conditioned on the action sequence taken between timesteps, making it suitable for control settings. This principle of latent-space prediction is the same one that LeCun later formalized as the **JEPA** (Joint-Embedding Predictive Architecture, 2022) framework, which retroactively unifies methods like CPC, BYOL, and VICReg under a single conceptual family. In this sense, AC-CPC can be seen as an action-conditioned instance of the JEPA paradigm.
* **Architecture:** 8 layers, 256d embeddings, 8 heads, C=4 context frames (~6.7M parameters).
* **Why Transformer over RNN:** With an LSTM/GRU, the FSQ's quantized vectors must be flattened into a continuous input (64 x 4d = 256 floats), yielding only 16x compression over the raw frame. A Transformer operates directly on discrete token indices, preserving the full 51x compression — a 3.2x improvement. The Transformer also naturally handles action conditioning via attention and captures long-range spatial dependencies across the token grid. This aligns with modern world model architectures (IRIS, GENIE) that use Transformers on discrete visual tokens.
* **Relevance:** Learns the game's physics and temporal dynamics entirely in discrete latent space, allowing the agent to "hallucinate" precise trajectories as token sequences.

### C. Controller (C) - *The Agent*

* **Type:** Linear Single-Layer Perceptron.
* **Function:** Maps the Transformer's hidden state to an optimal action (Jump / No Jump).
* **Environment:** **Latent Dream.** The agent is trained entirely inside the hallucinated environment generated by the Memory Model.
* **Optimization:** Covariance Matrix Adaptation Evolution Strategy (CMA-ES).

## 3. Design Rationale & Engineering Decisions

This implementation optimizes the original World Models architecture (Ha & Schmidhuber, 2018) to align with the specific constraints of the *Geometry Dash* environment.

### 3.1 Deterministic Dynamics (Removal of MDN)

The original architecture utilized a **Mixture Density Network (MDN)** to model environmental uncertainty (e.g., enemy movement in *Doom*).

* **Observation:** *Geometry Dash* physics are strictly deterministic; a specific input at a specific state always yields the same outcome. Visual stochasticity (particles, effects) is noise, not meaningful state.
* **Decision:** Replaced the probabilistic MDN-RNN with a deterministic sequence model.
* **Benefit:** Eliminates sampling noise and "representation blurring," allowing for high-fidelity latent rollouts with significantly lower computational overhead.

### 3.2 Discrete Tokenization (FSQ over VQ-VAE over beta-VAE)

The original World Models paper uses a beta-VAE with continuous Gaussian latents.

* **Observation (beta-VAE):** Beta-VAE reconstructions are fundamentally blurred by posterior averaging. After exhaustive hyperparameter search (beta values, annealing schedules, loss functions, latent dimensions), reconstructions could not distinguish spikes from blocks — a fatal limitation for a precision game.
* **Observation (VQ-VAE):** VQ-VAE eliminates blurriness but introduces codebook collapse, commitment cost tuning (cc), and EMA update complexity. Achieving stable training required extensive hyperparameter search (embedding dim, codebook size, commitment cost, EMA vs gradient updates, k-means init).
* **Decision:** Replaced the VQ-VAE with an FSQ-VAE (Mentzer et al., 2023) producing 64 discrete tokens per frame from levels $[8,5,5,5]$ (1000 codes).
* **Benefit:** FSQ quantizes each latent dimension to a fixed number of scalar levels via simple rounding — no learned codebook, no commitment loss, no EMA updates, no collapse risk. 100% codebook utilization by construction. Lower reconstruction error than VQ-VAE (val_recon 1.79 vs 2.73) with simpler training.

### 3.3 Transformer World Model (over LSTM/GRU)

The original architecture uses an LSTM (MDN-RNN) operating on flattened continuous latent vectors.

* **Observation:** Feeding an RNN flattened FSQ vectors (64 x 4d = 256 floats) yields only 16x compression over the raw 4,096-pixel input — wasting most of the FSQ's compression potential on continuous vector overhead.
* **Decision:** Replaced the RNN with a Transformer operating on discrete FSQ codes.
* **Benefit:** Preserves the full 51x compression ratio (64 tokens x $\log_2(1000) \approx 10$ bits = 638 bits vs 32,768 bits) — a 3.2x improvement over the RNN approach. The Transformer classifies over a 1000-entry vocabulary rather than regressing continuous vectors, and its own embedding layer decouples working dimensionality from the FSQ latent dimension.

### 3.4 Inference Strategy

* **Primary: Linear Controller (CMA-ES).** Reactive policy ($Action = W \cdot h_t$) trained via evolution in the dream environment. $O(1)$ inference, suitable for high-frequency decisions. Canonical approach from Ha & Schmidhuber (2018).
* **Potential extension: Beam Search Planning.** Geometry Dash is deterministic with a binary action space, which makes tree search attractive in principle. At each frame, branch into jump/no-jump, roll out each branch in the world model for $H \leq C$ steps (where $C$ is the context size), prune branches that predict death, execute the best surviving action. $H$ must stay within the context window so at least one real observation anchors the rollout — beyond that, compounding prediction errors degrade branch quality. This idea may be explored if the linear controller struggles with obstacle patterns requiring multi-step planning, but the compute cost ($2^H$ forward passes per decision) may be prohibitive at 30 FPS.

### 3.5 Input Preprocessing (64x64 Square Crop, Sobel Edges)

The original World Models paper uses $64 \times 64$ RGB inputs.

* **Observation:** *Geometry Dash* renders in 16:9 widescreen with visually noisy backgrounds. Raw RGB wastes encoder capacity on particles, color gradients, and decorations that carry zero gameplay information.
* **Decision:** Crop to a 1032x1032 gameplay square (player left-aligned, forward obstacles visible), apply Sobel edge detection at full resolution, then downscale to $64 \times 64$ grayscale. The encoder uses 3 stride-2 convolutions (64 → 32 → 16 → 8) with residual blocks and SiLU activations, producing the $8 \times 8$ spatial grid for FSQ tokenization.
* **Benefit (Sobel):** Extracts structural boundaries — platforms, spikes, player outline — while discarding most of the visual noise. Binary-like edges are far easier for the FSQ-VAE to reconstruct sharply.
* **Benefit (UI Removal):** The progress bar encodes the player's absolute position within a specific level. Retaining it would allow the model to memorize level layouts ("at 47%, a triple spike appears") rather than learning **reactive obstacle dynamics**. Removing it forces the agent to rely solely on visual obstacle perception, producing a more generalizable policy.

### 3.6 Controller Training: Data-Anchored Dream Rollouts

The controller is trained on short dream segments anchored to real gameplay data, avoiding compounding prediction errors from long autoregressive rollouts:

1. **Sample** a real episode with $\geq 2C$ frames (where $C$ is the Transformer's context size, currently 4).
2. **Warmup:** Feed the first $C$ frames as ground-truth context to the Transformer.
3. **Dream rollout:** The controller plays for $C$–20 frames autoregressively in the Transformer's hallucinated environment, choosing jump/idle at each step.
4. **Fitness:** Number of frames survived before the Transformer predicts death (via the death token).
5. **Optimize:** CMA-ES evolves the controller's weight matrix across many such segments sampled from diverse episodes.

This design keeps rollouts short and grounded — the Transformer never drifts far from real data, and different episodes expose the controller to diverse obstacle patterns. No level conditioning is needed since all relevant obstacles are visible within the context window.

## 4. Data Collection

Training data is recorded directly from live *Geometry Dash* gameplay using `scripts/record_gameplay.py`. The recorder captures screen frames at 30 FPS via hardware-accelerated DXGI capture (`dxcam`), preprocesses them into 64×64 Sobel edge maps, and logs the player's action (jump/idle) each frame. Death detection is handled by reading the game's process memory (`deepdash/gd_mem.py`) — specifically `PlayerObject.m_isDead` — which allows automatic episode splitting on each death and skipping the respawn overlay, requiring zero manual intervention beyond playing the game.

### Frame Rate Choice

The recording frame rate (30 FPS) was chosen based on the tightest timing constraint in the target levels: **triple spikes require a 3-frame window at 60 FPS (~50ms)**. At 30 FPS (33ms/frame), this gives ~1.5 frames of margin — tight but sufficient. Recording at 30 FPS also halves dataset redundancy compared to 60 FPS, since consecutive frames in a scrolling game are highly similar. The same frame rate will be used at inference time (stepping the game every 8 physics ticks at 240 TPS).

### Data Strategy

The initial dataset was recorded on the first official levels of the game (levels 1–7), chosen because their visual style is graphically "vanilla" — clean geometric shapes, minimal particle effects, no custom decorations that would confuse the edge detector. However, *Geometry Dash* only ships ~21 official levels, and the early ones are short. To scale the dataset without waiting for skill improvement on harder official levels, we turned to **community-made custom levels** as surrogates. The *Geometry Dash* level editor has produced millions of user-created levels, many of which use the same vanilla visual style as the early official levels. By selecting graphically compatible custom levels, we can generate practically unlimited training data with diverse obstacle patterns while staying within the visual distribution our models were trained on.

### Current Dataset

* **~3,600 episodes**, **~179K frames** across official levels (1–7) and vanilla-style custom levels, recorded at 30 FPS.
* Episodes range from a few frames (instant deaths) to several hundred (successful runs or clears).
* Multi-user recording supported (`--user` flag) with 10K episode ranges per user, enabling parallel data collection.
* The dataset has been scaled incrementally — the last time we tripled the dataset size, transformer accuracy roughly doubled, suggesting the model is still in the data-hungry regime.

## 5. Project Roadmap

### Phase 1: Vision — FSQ-VAE Tokenizer on Real Game Footage ✓

* **Status:** Complete. Retrained on expanded 30 FPS dataset (~179K frames) on A100 with bf16 AMP and torch.compile (val_recon 2.57). Sharp reconstructions preserving platforms, spikes, and player position while filtering visual noise.
* **Result:** FSQ levels $[8,5,5,5]$ (1000 codes), 4d latents, 8×8 spatial grid (64 tokens/frame).
* **Training:** GRWM regularization (temporal slowness + uniformity), ±4px random shift augmentation, cosine LR schedule. Batch size 2048 on A100.

### Phase 2: Dynamics — Transformer World Model (current)

* **Status:** Training on A100 with expanded 30 FPS dataset (~3,600 episodes, ~179K frames, ~2.2M samples with shift augmentation).
* **Architecture:** MaskGIT + block-causal attention + 3D-RoPE (row, col, frame) + AC-CPC contrastive loss + death token + focal loss + spatial shift augmentation (15×) + dual token noise (random 5% + FSQ neighbor 5%) + death oversampling (15×). 256d/8H/8L (~6.7M params).

### Phase 3: Control — Dream-Trained Agent

* **Goal:** Train an agent that masters *Geometry Dash* levels without ever touching the real game.
* **Method:** CMA-ES linear controller trained on data-anchored dream rollouts (C-frame warmup from real episodes, ~20-frame dream rollout). Beam search planning as complementary approach.
* **Deployment:** Screen capture → FSQ encode → Transformer predicts → policy acts on Transformer hidden state. Game stepped via tick injection (every 8 ticks at 240 TPS = 30 decisions/sec).
* **Success Metric:** Zero-shot deployment — the agent plays the real game using only the learned latent dynamics.

## 6. References

* **Primary Architecture:** Ha, D., & Schmidhuber, J. (2018). *World Models*. [arXiv:1803.10122](https://arxiv.org/abs/1803.10122)
* **FSQ:** Mentzer, F., Minnen, D., Agustsson, E., & Tschannen, M. (2023). *Finite Scalar Quantization: VQ-VAE Made Simple*. [arXiv:2309.15505](https://arxiv.org/abs/2309.15505)
* **VQ-VAE:** van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). *Neural Discrete Representation Learning*. [arXiv:1711.00937](https://arxiv.org/abs/1711.00937)
* **Transformer World Model (IRIS):** Micheli, V., Alonso, E., & Fleuret, F. (2023). *Transformers are Sample-Efficient World Models*. [arXiv:2209.00588](https://arxiv.org/abs/2209.00588)
* **MaskGIT:** Chang, H., Zhang, H., Jiang, L., Liu, C., & Freeman, W. T. (2022). *MaskGIT: Masked Generative Image Transformer*. [arXiv:2202.04200](https://arxiv.org/abs/2202.04200)
* **CPC:** Oord, A. van den, Li, Y., & Vinyals, O. (2018). *Representation Learning with Contrastive Predictive Coding*. [arXiv:1807.03748](https://arxiv.org/abs/1807.03748)
* **AC-CPC (TWISTER):** Burchert, J., et al. (2025). *Learning Transformer-based World Models with Contrastive Predictive Coding*. [arXiv:2503.04416](https://arxiv.org/abs/2503.04416)
* **JEPA:** LeCun, Y. (2022). *A Path Towards Autonomous Machine Intelligence*. [OpenReview](https://openreview.net/pdf?id=BZ5a1r-kVsf)
* **Block-Causal Attention:** Gupta, A., et al. (2025). *Improving Transformer World Models for Data-Efficient RL*. [arXiv:2502.01591](https://arxiv.org/abs/2502.01591)
* **Foundational RL:** Mnih, V., et al. (2013). *Playing Atari with Deep Reinforcement Learning*. [arXiv:1312.5602](https://arxiv.org/abs/1312.5602)
