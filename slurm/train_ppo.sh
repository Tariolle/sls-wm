#!/bin/bash
#SBATCH -J "train_ppo_20k"
#SBATCH -o slurm/logs/train_ppo_%j.out
#SBATCH -e slurm/logs/train_ppo_%j.err
#SBATCH -p ar_a100
#SBATCH --gres=gpu:a100:1
#SBATCH -n 1
#SBATCH --cpus-per-gpu 8
#SBATCH --mem 64G
#SBATCH --time=08:00:00
#SBATCH --exclude=c23hpda2
#SBATCH --signal=B:USR1@300

# Auto-resubmit on timeout: SIGUSR1 is sent 5 minutes before the time limit.
cleanup_and_resubmit() {
    echo "$(date): Caught signal, saving and resubmitting..."
    sbatch "$0"
    exit 0
}
trap cleanup_and_resubmit USR1

module purge
module load aidl/pytorch/2.6.0-cuda12.6

echo "=== Train Controller (PPO, constant LR, auto-resume) ==="
echo "$(date): Starting on $(hostname), Job ID: $SLURM_JOB_ID"

# First run: fresh from BC. Subsequent runs: resume.
if [ -f checkpoints/controller_ppo_latest.pt ]; then
    echo "Resuming from latest checkpoint"
    RESUME_FLAG="--resume"
else
    echo "Starting fresh from BC checkpoint"
    RESUME_FLAG=""
fi

python -u scripts/train_controller_ppo.py \
    --transformer-checkpoint checkpoints/transformer_best.pt \
    --pretrained checkpoints/controller_bc_best.pt \
    $RESUME_FLAG \
    --n-iterations 15000 \
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
    --max-dream-steps 45 \
    --death-threshold 0.5 \
    --token-embed-dim 16 \
    --context-frames 4 \
    --vocab-size 1000 \
    --tokens-per-frame 64 \
    --embed-dim 384 \
    --n-heads 8 \
    --n-layers 8 \
    --dropout 0.1 \
    --checkpoint-dir checkpoints \
    --n-eval-episodes 512 \
    --eval-interval 10 \
    --seed 42 &

# Wait for training process (needed for signal handling)
wait $!
echo "$(date): Training finished normally"
