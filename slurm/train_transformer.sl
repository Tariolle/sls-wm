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
RESUME_FLAG=checkpoints_v5/.resume_transformer

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
export PATH="$HOME/.local/bin:$PATH"
pip install --user wandb 2>/dev/null

# Tokenization is idempotent (skips already-tokenized episodes)
echo "=== Step 1a: Tokenize death episodes ==="
python -u scripts/tokenize_episodes.py \
    --model fsq \
    --checkpoint checkpoints_v5/fsq_best.pt \
    --episodes-dir data/death_episodes \
    --batch-size 512 \
    --levels 6 6 6 6

echo "=== Step 1b: Tokenize expert episodes ==="
python -u scripts/tokenize_episodes.py \
    --model fsq \
    --checkpoint checkpoints_v5/fsq_best.pt \
    --episodes-dir data/expert_episodes \
    --batch-size 512 \
    --levels 6 6 6 6

RESUME_ARG=""
if [ -f "$RESUME_FLAG" ]; then
    RESUME_ARG="--resume"
    rm "$RESUME_FLAG"
    echo "=== Resuming from checkpoint ==="
fi

echo "=== Step 2: Train Transformer ($(date)) ==="
python -u scripts/train_transformer.py \
    --config configs/v5.yaml \
    $RESUME_ARG &

TRAIN_PID=$!
wait "$TRAIN_PID"
