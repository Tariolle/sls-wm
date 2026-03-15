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

# Train controller via Reinforce policy gradients in dream rollouts.
#
# Submit:  sbatch slurm/train_controller_reinforce.sl
# Monitor: tail -f slurm/logs/train_controller_reinforce.out

module purge
module load aidl/pytorch/2.6.0-cuda12.6

echo "=== Step 1: Tokenize episodes ==="
python -u scripts/tokenize_episodes.py \
    --model fsq \
    --checkpoint checkpoints/fsq_best.pt \
    --episodes-dir data/episodes \
    --batch-size 512 \
    --levels 8 5 5 5

echo "=== Step 2: Train Controller (Reinforce) ==="
python -u scripts/train_controller_reinforce.py \
    --transformer-checkpoint checkpoints/transformer_best.pt \
    --episodes-dir data/episodes \
    --n-iterations 500 \
    --n-episodes 64 \
    --lr 3e-4 \
    --gamma 0.99 \
    --entropy-coeff 0.01 \
    --max-dream-steps 20 \
    --death-threshold 0.5 \
    --mlp-hidden 64 \
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
