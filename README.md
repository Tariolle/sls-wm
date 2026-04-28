# SLS-WM
### Structured Label Smoothing over Finite Scalar Quantization for Discrete World Models

[Florent Tariolle](mailto:florent.tariolle@insa-rouen.fr)

**Abstract:** Discrete World Models tokenize observations with a learned quantizer and predict next-frame tokens with a transformer, but standard cross-entropy treats every incorrect prediction as equally wrong. Finite Scalar Quantization (FSQ) makes a richer signal available by construction: each code sits on an integer coordinate lattice, so a token one step away in one dimension is a near-miss while a token at the opposite corner is a gross error. We introduce *Structured Label Smoothing* (SLS), which replaces the one-hot training target with a kernel over codebook coordinates, so a near-miss prediction is treated as a near-miss rather than a gross error. An isotropic kernel with bandwidth fixed by a first-neighbour rule gives a zero-calibration hyperparameter that is robust to codebook drift. We integrate SLS into a complete Vision-Model-Controller pipeline for Geometry Dash, where the controller is trained entirely in imagination and deployed at 30 FPS on the real game via screen capture.

<p align="center">
   <b>[ <a href="https://tariolle.github.io/sls-wm/static/pdfs/sls_wm.pdf">Paper Draft</a> | <a href="https://tariolle.github.io/sls-wm/">Website</a> ]</b>
</p>

<p align="center">
  <img src="docs/static/images/pipeline.png" width="80%">
</p>

> **Status:** Pre-freeze research code. Model freeze target **2026-05-31**, NeurIPS 2026 workshop submission. Numerical results and ablation tables will land with the final model; this repository currently describes the method and architecture, not outcomes.

## Using the code

**Environment.** Conda, PyTorch 2.10, CUDA 12.6.
```bash
conda run -n <env> python -m pip install -r requirements.txt
```

**Train the FSQ-VAE (V):**
```bash
python scripts/train_fsq.py
```

**Train the transformer world model (M) on the frozen FSQ tokens:**
```bash
python scripts/train_transformer.py
```

**Train the controller (C): BC warm-start, then PPO in imagination:**
```bash
python scripts/train_controller_bc.py
python scripts/train_controller_ppo.py --pretrained checkpoints/controller_bc_best.pt
```

**Deploy to the live game (screen capture, 30 FPS):**
```bash
python scripts/deploy.py
```

Cluster launches (SLURM, A100): `sbatch slurm/train_fsq.sl`, `sbatch slurm/train_transformer.sl`, `sbatch slurm/train_controller.sl`.

## Contact

Feel free to open [issues](https://github.com/Tariolle/sls-wm/issues). For questions or collaborations, contact `florent.tariolle@insa-rouen.fr`.
