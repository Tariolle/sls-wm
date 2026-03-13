#!/bin/bash
#SBATCH -J "train_tfm"
#SBATCH -o slurm/logs/train_transformer.out
#SBATCH -e slurm/logs/train_transformer.err
#SBATCH -p ar_a100
#SBATCH --gres=gpu:a100:1
#SBATCH -n 1
#SBATCH --cpus-per-gpu 8
#SBATCH --mem 64G
#SBATCH --time=08:00:00

# Train Transformer world model on A100.
# Step 1: tokenize episodes with frozen FSQ-VAE
# Step 2: train transformer on tokenized data
#
# Prerequisites (run once on login node):
#   module load aidl/pytorch/2.6.0-cuda12.6
#   pip install --user numpy
#
# Submit:  sbatch slurm/train_transformer.sl
# Monitor: tail -f slurm/logs/train_transformer.out

module purge
module load aidl/pytorch/2.6.0-cuda12.6

echo "=== Step 1: Tokenize episodes (with shift augmentation) ==="
python -u scripts/tokenize_episodes.py \
    --model fsq \
    --checkpoint checkpoints/fsq_best.pt \
    --episodes-dir data/episodes \
    --batch-size 512 \
    --levels 8 5 5 5 \
    --shifts -4 -2 0 2 4 \
    --shifts-v -3 0 3

echo "=== Step 2: Train Transformer ==="
python -u scripts/train_transformer.py \
    --episodes-dir data/episodes \
    --epochs 200 \
    --batch-size 512 \
    --lr 1e-4 \
    --context-frames 4 \
    --vocab-size 1000 \
    --tokens-per-frame 64 \
    --embed-dim 256 \
    --n-heads 8 \
    --n-layers 8 \
    --dropout 0.1 \
    --weight-decay 0.01 \
    --cpc-weight 0.1 \
    --token-noise 0.05 \
    --fsq-noise 0.05 \
    --label-smoothing 0.1 \
    --focal-gamma 2.0 \
    --death-oversample 15 \
    --steps-per-epoch 500 \
    --checkpoint-dir checkpoints \
    --patience 30 \
    --seed 42
