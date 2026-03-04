# VAE Training Log

## v1 — Baseline (beta=1.0, no annealing)

**Config:** 100 epochs, batch_size=32, lr=1e-3, beta=1.0 (standard VAE)

**Result:** FAILURE — Posterior collapse. Reconstructions are blurry color blobs with no structural detail. The model outputs the average background color + smeared vertical player position + pause button artifact.

**Diagnosis:** KL loss (~25% of total) dominated too early, preventing the encoder from learning meaningful features. The model collapses to outputting the dataset mean.

**Evidence:**

- Final val loss: 0.1566 (epoch 76), recon=0.1107, kl=0.0459
- See `docs/vae_v1_fail_sample1.png` and `docs/vae_v1_fail_sample6.png`
- Full training log: `checkpoints/train_log_v1_no_annealing.csv`

## v2 — Beta annealing (beta_max=0.1, warmup=20)

**Config:** 24 epochs (interrupted), batch_size=32, lr=1e-3, beta annealed 0 -> 0.1 over 20 epochs

**Fix:** KL annealing (beta-VAE). Beta starts at 0 (pure autoencoder phase), then linearly increases to 0.1 over 20 epochs.

**Result:** FAILURE — Better than v1 but still blurry. No structural detail (platforms/spikes not visible). Val recon plateaued at ~0.074.

**Issues:** ReduceLROnPlateau watches total val_loss (which increases with beta), causing premature LR drops (1e-3 → 1.3e-4 by epoch 21).

## v3 — Pure autoencoder (beta=0)

**Config:** 30 epochs, batch_size=32, lr=1e-3, beta=0 (no KL penalty)

**Goal:** Diagnose whether the bottleneck is KL pressure or architecture capacity.

**Result:** PARTIAL — Best recon loss so far (val recon 0.062 vs v2's 0.074). Structures slightly more visible but still not sharp enough for gameplay. Confirms the architecture can learn somewhat, but 32 latent dims + 3 color channels on complex multi-level data is too much.

**Diagnosis:** Problem is likely visual complexity across levels. Early levels have clean black/white geometry. Later levels have flashy particles, complex backgrounds, visual noise with zero gameplay value. VAE wastes capacity on noise.

**Next steps:** See Issue #2 comment — try grayscale + early levels only to validate architecture on clean data.
