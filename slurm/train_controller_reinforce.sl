#!/bin/bash
#SBATCH -J "train_ctrl_rf"
#SBATCH -o slurm/logs/train_controller_reinforce.out
#SBATCH -e slurm/logs/train_controller_reinforce.err
#SBATCH -p ar_a100
#SBATCH --gres=gpu:a100:1
#SBATCH -n 1
#SBATCH --cpus-per-gpu 8
#SBATCH --mem 64G
#SBATCH --time=08:00:00

# Train controller via actor-critic Reinforce in dream rollouts.
# DART-style ViT, uniform sampling, percentile normalization.
#
# Submit:  sbatch slurm/train_controller_reinforce.sl
# Monitor: tail -f slurm/logs/train_controller_reinforce.out

module purge
module load aidl/pytorch/2.6.0-cuda12.6

# Tokenization skipped — episodes already tokenized from prior runs.
# To re-tokenize, run: python -u scripts/tokenize_episodes.py --model fsq \
#   --checkpoint checkpoints/fsq_best.pt --episodes-dir data/episodes \
#   --batch-size 512 --levels 8 5 5 5

echo "=== Train Controller (Actor-Critic) ==="
python -u scripts/train_controller_reinforce.py \
    --transformer-checkpoint checkpoints/transformer_best.pt \
    --episodes-dir data/episodes \
    --n-iterations 500 \
    --n-episodes 64 \
    --lr 3e-4 \
    --gamma 0.99 \
    --lam 0.95 \
    --entropy-coeff 0.01 \
    --critic-coeff 0.5 \
    --max-dream-steps 20 \
    --death-threshold 0.5 \
    --policy-embed-dim 128 \
    --policy-n-heads 4 \
    --policy-n-layers 3 \
    --policy-dropout 0.1 \
    --context-frames 4 \
    --vocab-size 1000 \
    --tokens-per-frame 64 \
    --embed-dim 256 \
    --n-heads 8 \
    --n-layers 8 \
    --dropout 0.1 \
    --checkpoint-dir checkpoints \
    --eval-interval 10 \
    --seed 42
