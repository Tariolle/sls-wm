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
#SBATCH --signal=B:USR1@300

# Train Transformer world model on A100.
# Auto-resumes: SLURM sends USR1 5 min before time limit,
# the trap saves checkpoint and resubmits the job.
#
# Prerequisites (run once on login node):
#   module load aidl/pytorch/2.6.0-cuda12.6
#   pip install --user numpy
#
# Submit:  sbatch slurm/train_transformer.sl
# Monitor: tail -f slurm/logs/train_transformer.out

# Auto-resume: the trap creates a sentinel file before requeuing.
# On next run, if the sentinel exists, we pass --resume.
RESUME_FLAG=checkpoints/.resume_transformer

handle_timeout() {
    echo "=== USR1 received ($(date)), saving and resubmitting ==="
    kill -TERM "$TRAIN_PID" 2>/dev/null
    wait "$TRAIN_PID"
    touch "$RESUME_FLAG"
    scontrol requeue "$SLURM_JOB_ID" || sbatch "$0"
    exit 0
}
trap handle_timeout USR1

module purge
module load aidl/pytorch/2.6.0-cuda12.6

# Tokenization is idempotent (skips already-tokenized episodes)
echo "=== Step 1a: Tokenize death episodes (with shift augmentation) ==="
python -u scripts/tokenize_episodes.py \
    --model fsq \
    --checkpoint checkpoints/fsq_best.pt \
    --episodes-dir data/death_episodes \
    --batch-size 512 \
    --levels 8 5 5 5 \
    --shifts-v -4 -2 0 2 4

echo "=== Step 1b: Tokenize expert episodes (with shift augmentation) ==="
python -u scripts/tokenize_episodes.py \
    --model fsq \
    --checkpoint checkpoints/fsq_best.pt \
    --episodes-dir data/expert_episodes \
    --batch-size 512 \
    --levels 8 5 5 5 \
    --shifts-v -4 -2 0 2 4

RESUME_ARG=""
if [ -f "$RESUME_FLAG" ]; then
    RESUME_ARG="--resume"
    rm "$RESUME_FLAG"
    echo "=== Resuming from checkpoint ==="
fi

echo "=== Step 2: Train Transformer ($(date)) ==="
python -u scripts/train_transformer.py \
    --episodes-dir data/death_episodes \
    --expert-episodes-dir data/expert_episodes \
    --epochs 200 \
    --batch-size 512 \
    --lr 2e-3 \
    --context-frames 4 \
    --vocab-size 1000 \
    --tokens-per-frame 64 \
    --embed-dim 256 \
    --n-heads 8 \
    --n-layers 8 \
    --dropout 0.1 \
    --weight-decay 0.01 \
    --cpc-weight 1.0 \
    --token-noise 0.05 \
    --fsq-noise 0.05 \
    --label-smoothing 0.1 \
    --fsq-sigma 0.9 \
    --focal-gamma 2.0 \
    --death-oversample 5 \
    --steps-per-epoch 500 \
    --checkpoint-dir checkpoints \
    --patience 30 \
    --seed 42 \
    $RESUME_ARG &

TRAIN_PID=$!
wait "$TRAIN_PID"
