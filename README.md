# DeepDash
**Latent Dynamics & Temporal Sequence Control for Geometry Dash**

**Course:** Representation Learning

**Architecture:** World Models (VAE + GRU + Controller)

## 1. Project Overview
DeepDash is a Deep Reinforcement Learning agent designed to master procedural *Geometry Dash* levels within a custom-built game engine. Unlike standard RL approaches that map pixels directly to actions, DeepDash explicitly separates **visual perception** from **control policy**.

This project implements a modified "World Models" architecture, demonstrating how an agent can learn a compressed **latent representation** of the game world and "dream" deterministic future states to optimize its trajectory.

## 2. Technical Architecture
The system is composed of three distinct neural networks trained sequentially:

### A. Vision Model (V) - *The Representation Learner*
* **Type:** Variational Autoencoder (VAE).
* **Input:** Semantic mask frames ($64 \times 64 \times 3$) from the custom engine.
* **Function:** Compresses sparse visual data into a low-dimensional latent vector ($z \in \mathbb{R}^{32}$).
* **Relevance:** Demonstrates unsupervised feature extraction of game entities (spikes, blocks, player) from simplified semantic inputs.

### B. Memory Model (M) - *The Dynamics Learner*
* **Type:** Gated Recurrent Unit (GRU).
* **Function:** Predicts the exact next latent state ($z_{t+1}$) given the current state ($z_t$) and action ($a_t$).
* **Relevance:**  Learns the rigid physics engine and temporal dynamics using a computationally efficient RNN, allowing the agent to "hallucinate" precise trajectories without the complexity of LSTM gates.

### C. Controller (C) - *The Agent*
* **Type:** Linear Single-Layer Perceptron.
* **Function:** Maps the concatenated state vector ($z_t, h_t$) to an optimal action (Jump / No Jump).
* **Optimization:** Covariance Matrix Adaptation Evolution Strategy (CMA-ES).

## 3. The Custom Engine
The environment is a custom-written *Geometry Dash* clone designed for accelerated training.
* **Headless Mode:** Decoupled rendering for high-speed simulation.
* **Semantic Rendering:** Engine outputs simplified "semantic masks" (Player=Green, Danger=Red, Environment=Blue/Black).
* **Deterministic Physics:** Fixed framerate and gravity ensure reproducibility, eliminating the need for stochastic modeling.

## 4. References
* **Primary Architecture:** Ha, D., & Schmidhuber, J. (2018). *World Models*. [arXiv:1803.10122](https://arxiv.org/abs/1803.10122)
* **Recurrent Dynamics (GRU):** Cho, K., et al. (2014). *Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation*. [arXiv:1406.1078](https://arxiv.org/abs/1406.1078)
* **Foundational RL:** Mnih, V., et al. (2013). *Playing Atari with Deep Reinforcement Learning*. [arXiv:1312.5602](https://arxiv.org/abs/1312.5602)
