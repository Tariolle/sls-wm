#!/bin/bash
#SBATCH -J "tokenize"
#SBATCH -o slurm/logs/tokenize_episodes.out
#SBATCH -e slurm/logs/tokenize_episodes.err
#SBATCH -p ar_a100
#SBATCH --gres=gpu:a100:1
#SBATCH -n 1
#SBATCH --cpus-per-gpu 4
#SBATCH --mem 32G
#SBATCH --time=00:30:00

# Tokenize all episodes with frozen FSQ-VAE.
#
# Submit:  sbatch slurm/tokenize_episodes.sl
# Monitor: tail -f slurm/logs/tokenize_episodes.out

module purge
module load aidl/pytorch/2.6.0-cuda12.6

python -u scripts/tokenize_episodes.py \
    --model fsq \
    --checkpoint checkpoints/fsq_best.pt \
    --episodes-dir data/death_episodes \
    --batch-size 512 \
    --levels 8 5 5 5 \
    --grid-size 16

python -u scripts/tokenize_episodes.py \
    --model fsq \
    --checkpoint checkpoints/fsq_best.pt \
    --episodes-dir data/expert_episodes \
    --batch-size 512 \
    --levels 8 5 5 5 \
    --grid-size 16
