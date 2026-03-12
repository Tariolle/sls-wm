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

python -u scripts/train_fsq.py \
    --episodes-dir data/episodes \
    --epochs 200 \
    --batch-size 2048 \
    --lr 1e-3 \
    --checkpoint-dir checkpoints \
    --levels 8 5 5 5 \
    --alpha-slow 0.1 \
    --alpha-uniform 0.01 \
    --val-ratio 0.1 \
    --seed 42 \
    --amp \
    --compile
