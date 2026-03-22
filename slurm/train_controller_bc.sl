#!/bin/bash
#SBATCH -J "train_ctrl_bc"
#SBATCH -o slurm/logs/train_controller_bc.out
#SBATCH -e slurm/logs/train_controller_bc.err
#SBATCH -p ar_a100
#SBATCH --gres=gpu:a100:1
#SBATCH -n 1
#SBATCH --cpus-per-gpu 8
#SBATCH --mem 64G
#SBATCH --time=01:00:00

# Behavioral cloning: pretrain controller on expert episodes.
# Should be fast (~33K samples, 50 epochs).
#
# Submit:  sbatch slurm/train_controller_bc.sl
# Monitor: tail -f slurm/logs/train_controller_bc.out

module purge
module load aidl/pytorch/2.6.0-cuda12.6

echo "=== Train Controller (BC) ==="
python -u scripts/train_controller_bc.py \
    --expert-episodes-dir data/expert_episodes \
    --transformer-checkpoint checkpoints/transformer_best.pt \
    --epochs 50 \
    --batch-size 512 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --val-ratio 0.1 \
    --vocab-size 1000 \
    --embed-dim 256 \
    --n-heads 8 \
    --n-layers 8 \
    --tokens-per-frame 64 \
    --context-frames 4 \
    --dropout 0.1 \
    --checkpoint-dir checkpoints \
    --seed 42
