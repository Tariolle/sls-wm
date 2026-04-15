#!/bin/bash
#SBATCH -J "train_ctrl_nobc"
#SBATCH -o slurm/logs/train_controller_nobc.out
#SBATCH -e slurm/logs/train_controller_nobc.err
#SBATCH -p ar_a100
#SBATCH --gres=gpu:a100:1
#SBATCH -n 1
#SBATCH --cpus-per-gpu 8
#SBATCH --mem 64G
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@300

# PPO cold-start ablation: no BC pretraining, separate checkpoint dir.
# Compares against train_controller.sl (BC-init) to measure BC value.
# Three-outcome test: collapse / slowdown / parity (see project_bc_ablation.md).
#
# Submit:  sbatch slurm/train_controller_nobc.sl
# Monitor: tail -f slurm/logs/train_controller_nobc.out

cleanup_and_resubmit() {
    echo "$(date): Caught signal, saving and resubmitting..."
    sbatch "$0"
    exit 0
}
trap cleanup_and_resubmit USR1

module purge
module load aidl/pytorch/2.6.0-cuda12.6
export PATH="$HOME/.local/bin:$PATH"
pip install --user wandb 2>/dev/null

echo "$(date): Starting on $(hostname), Job ID: $SLURM_JOB_ID"

RESUME_FLAG=""
if [ -f checkpoints_nobc/controller_ppo_latest.pt ]; then
    RESUME_FLAG="--resume"
fi

python -u scripts/train_controller_ppo.py \
    --config configs/v4.yaml \
    --checkpoint-dir checkpoints_nobc \
    --wandb-name ppo-512d-nobc \
    --no-pretrained \
    --seed 42 \
    $RESUME_FLAG &

wait $!
echo "$(date): Training finished normally"
