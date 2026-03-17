#!/bin/bash
#SBATCH --job-name=deepdash-ppo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/ppo_%j.out
#SBATCH --error=slurm/logs/ppo_%j.err
#SBATCH --signal=B:USR1@300

# Auto-resubmit on timeout: SIGUSR1 is sent 5 minutes before the time limit.
# The trap saves state and resubmits this script.
cleanup_and_resubmit() {
    echo "$(date): Caught signal, saving and resubmitting..."
    # The training script saves checkpoints every eval_interval iterations,
    # so the latest checkpoint is already on disk. Just resubmit.
    sbatch "$0"
    exit 0
}
trap cleanup_and_resubmit USR1

mkdir -p slurm/logs

cd "$SLURM_SUBMIT_DIR" || cd /home/florent/Documents/DeepDash

echo "$(date): Starting PPO training on $(hostname), GPU: $CUDA_VISIBLE_DEVICES"
echo "Job ID: $SLURM_JOB_ID"

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate deepdash

# First run: initialize from BC checkpoint
# Subsequent runs: resume from latest PPO checkpoint
if [ -f checkpoints/controller_reinforce_latest.pt ]; then
    echo "Resuming from latest checkpoint"
    python scripts/train_controller_reinforce.py \
        --resume \
        --n-iterations 20000 \
        --pretrained checkpoints/controller_bc_best.pt &
else
    echo "Starting fresh from BC checkpoint"
    python scripts/train_controller_reinforce.py \
        --pretrained checkpoints/controller_bc_best.pt \
        --n-iterations 20000 &
fi

# Wait for the training process (needed for signal handling)
wait $!
echo "$(date): Training finished normally"
