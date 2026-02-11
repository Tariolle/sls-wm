# DeepDash
**Latent Dynamics & Temporal Sequence Control for Geometry Dash**

**Course:** Representation Learning

**Architecture:** World Models (VAE + MDN-RNN + Controller)

## 1. Project Overview
DeepDash is a Deep Reinforcement Learning agent designed to master procedural *Geometry Dash* levels within a custom-built game engine. Unlike standard RL approaches that map pixels directly to actions, DeepDash explicitly separates **visual perception** from **control policy**.

This project implements a "World Models" architecture, demonstrating how an agent can learn a compressed **latent representation** of the game world and "dream" future states to optimize its trajectory.

## 2. Technical Architecture
The system is composed of three distinct neural networks trained sequentially:

### A. Vision Model (V) - *The Representation Learner*
* **Type:** Variational Autoencoder (VAE).
* **Input:** Raw pixel frames from the custom engine.
* **Function:** Compresses high-dimensional visual data into a low-dimensional latent vector ($z$).
* **Relevance:** Demonstrates unsupervised feature extraction of game entities (spikes, blocks, player) without labeled data.

### B. Memory Model (M) - *The Dynamics Learner*
* **Type:** Mixture Density Network - Recurrent Neural Network (MDN-RNN).
* **Function:** Predicts the probability distribution of the *next* latent state given the current state and action.
* **Relevance:** Learns the physics engine and temporal dynamics, allowing the agent to "hallucinate" outcomes before acting.

### C. Controller (C) - *The Agent*
* **Type:** Linear Single-Layer Perceptron.
* **Function:** Maps the distinct representation of the world (Vision + Memory) to an optimal action (Jump / No Jump).
* **Optimization:** Covariance Matrix Adaptation Evolution Strategy (CMA-ES).

## 3. The Custom Engine
The environment is a custom-written *Geometry Dash* clone designed for accelerated training.
* **Headless Mode:** Decoupled rendering for high-speed simulation.
* **Semantic Rendering:** Engine can output simplified "semantic masks" (Player=Green, Danger=Red) to accelerate representation learning.
* **Standard Interface:** Compatible with standard RL environment APIs.

## 4. References
* **Primary Architecture:** Ha, D., & Schmidhuber, J. (2018). *World Models*. [arXiv:1803.10122](https://arxiv.org/abs/1803.10122)
* **Foundational RL:** Mnih, V., et al. (2013). *Playing Atari with Deep Reinforcement Learning*. [arXiv:1312.5602](https://arxiv.org/abs/1312.5602)
