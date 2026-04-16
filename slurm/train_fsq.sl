#!/bin/bash
#SBATCH -J "train_fsq"
#SBATCH -o slurm/logs/train_fsq.out
#SBATCH -e slurm/logs/train_fsq.err
#SBATCH -p ar_a100
#SBATCH --gres=gpu:a100:1
#SBATCH -n 1
#SBATCH --cpus-per-gpu 8
#SBATCH --mem 64G
#SBATCH --time=08:00:00

# Train FSQ-VAE on A100 with bf16 AMP and torch.compile.
#
# Prerequisites (run once on login node):
#   module load aidl/pytorch/2.6.0-cuda12.6
#   pip install --user numpy
#
# Submit:  sbatch slurm/train_fsq.sl
# Monitor: tail -f slurm/logs/train_fsq.out

module purge
module load aidl/pytorch/2.6.0-cuda12.6
export PATH="$HOME/.local/bin:$PATH"
pip install --user wandb 2>/dev/null

echo "=== Step 0: Pre-compute shift augmentation ==="
python -u scripts/shift_episodes.py \
    --episodes-dir data/death_episodes \
    --expert-episodes-dir data/expert_episodes \
    --shifts-v -4 -2 0 2 4

echo "=== Step 1: Train FSQ-VAE ==="
python -u scripts/train_fsq.py \
    --config configs/v5.yaml
