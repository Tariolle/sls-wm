#!/bin/bash
#SBATCH -J "train_ctrl_ppo"
#SBATCH -o slurm/logs/train_controller_reinforce.out
#SBATCH -e slurm/logs/train_controller_reinforce.err
#SBATCH -p ar_a100
#SBATCH --gres=gpu:a100:1
#SBATCH -n 1
#SBATCH --cpus-per-gpu 8
#SBATCH --mem 64G
#SBATCH --time=08:00:00
#SBATCH --exclude=c23hpda2

# Train controller via PPO in dream rollouts.
# CNN policy, survival reward, near-obstacle sampling, 512 episodes.
#
# Submit:  sbatch slurm/train_controller_reinforce.sl
# Monitor: tail -f slurm/logs/train_controller_reinforce.out

module purge
module load aidl/pytorch/2.6.0-cuda12.6

echo "=== Train Controller (PPO) ==="
python -u scripts/train_controller_reinforce.py \
    --transformer-checkpoint checkpoints/transformer_best.pt \
    --episodes-dir data/death_episodes \
    --n-iterations 2000 \
    --n-episodes 512 \
    --lr 1e-4 \
    --gamma 0.995 \
    --lam 0.95 \
    --clip-eps 0.2 \
    --ppo-epochs 4 \
    --minibatch-size 512 \
    --entropy-coeff 0.01 \
    --critic-coeff 0.5 \
    --max-grad-norm 0.5 \
    --max-dream-steps 30 \
    --max-frames-to-death 25 \
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
    --n-eval-episodes 512 \
    --eval-interval 10 \
    --seed 42
