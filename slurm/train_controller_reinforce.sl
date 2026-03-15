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
# CNN policy on 8x8 token grid, survival reward, high exploration.
#
# Submit:  sbatch slurm/train_controller_reinforce.sl
# Monitor: tail -f slurm/logs/train_controller_reinforce.out

module purge
module load aidl/pytorch/2.6.0-cuda12.6

echo "=== Train Controller (CNN Actor-Critic) ==="
python -u scripts/train_controller_reinforce.py \
    --transformer-checkpoint checkpoints/transformer_best.pt \
    --episodes-dir data/episodes \
    --n-iterations 2000 \
    --n-episodes 64 \
    --lr 1e-4 \
    --gamma 0.995 \
    --lam 0.95 \
    --entropy-coeff 0.01 \
    --critic-coeff 0.5 \
    --max-dream-steps 30 \
    --death-threshold 0.5 \
    --token-embed-dim 16 \
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
