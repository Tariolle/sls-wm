#!/bin/bash
#SBATCH -J "train_ctrl"
#SBATCH -o slurm/logs/train_controller.out
#SBATCH -e slurm/logs/train_controller.err
#SBATCH -p ar_a100
#SBATCH --gres=gpu:a100:1
#SBATCH -n 1
#SBATCH --cpus-per-gpu 8
#SBATCH --mem 64G
#SBATCH --time=08:00:00

# Train CMA-ES controller in dream rollouts on A100.
# Step 1: tokenize episodes with frozen FSQ-VAE (if not already done)
# Step 2: run CMA-ES controller training
#
# Prerequisites (run once on login node):
#   module load aidl/pytorch/2.6.0-cuda12.6
#   pip install --user cma
#
# Submit:  sbatch slurm/train_controller.sl
# Monitor: tail -f slurm/logs/train_controller.out

module purge
module load aidl/pytorch/2.6.0-cuda12.6

echo "=== Step 1: Tokenize episodes ==="
python -u scripts/tokenize_episodes.py \
    --model fsq \
    --checkpoint checkpoints/fsq_best.pt \
    --episodes-dir data/death_episodes \
    --batch-size 512 \
    --levels 8 5 5 5

echo "=== Step 2: Train CMA-ES Controller ==="
python -u scripts/train_controller.py \
    --transformer-checkpoint checkpoints/transformer_best.pt \
    --episodes-dir data/death_episodes \
    --max-generations 500 \
    --mlp-hidden 64 \
    --popsize 64 \
    --sigma0 0.5 \
    --n-episodes 64 \
    --max-dream-steps 20 \
    --death-threshold 0.5 \
    --context-frames 4 \
    --vocab-size 1000 \
    --tokens-per-frame 64 \
    --embed-dim 256 \
    --n-heads 8 \
    --n-layers 8 \
    --dropout 0.1 \
    --checkpoint-dir checkpoints \
    --seed 42
