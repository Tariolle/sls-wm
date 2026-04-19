#!/bin/bash
#SBATCH -J "train_ctrl"
#SBATCH -o slurm/logs/train_controller.out
#SBATCH -e slurm/logs/train_controller.err
#SBATCH -p ar_a100
#SBATCH --gres=gpu:a100:1
#SBATCH -n 1
#SBATCH --cpus-per-gpu 8
#SBATCH --mem 64G
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@300

# Combined BC pretraining + PPO fine-tuning with auto-resubmit.
# On first run: BC trains (~5 min), then PPO starts.
# On resume (job restart): BC skipped, PPO resumes from checkpoint.
# SIGUSR1 is sent 5 minutes before time limit, triggering resubmit.
#
# Submit:  sbatch slurm/train_controller.sl
# Monitor: tail -f slurm/logs/train_controller.out

cleanup_and_resubmit() {
    echo "$(date): Caught signal, saving and resubmitting..."
    sbatch "$0"
    exit 0
}
trap cleanup_and_resubmit USR1

module purge
module load aidl/pytorch/2.10.0-py3.12-cuda12.6
export PATH="$HOME/.local/bin:$PATH"
pip install --user --upgrade wandb "protobuf>=6.32" 2>/dev/null

echo "$(date): Starting on $(hostname), Job ID: $SLURM_JOB_ID"

# --- Phase 1: BC (skipped if PPO checkpoint exists) ---
if [ ! -f checkpoints_v5_dimweights/controller_ppo_latest.pt ]; then
    echo "=== Phase 1: Behavioral Cloning ==="
    python -u scripts/train_controller_bc.py \
        --config configs/v5.yaml \
        --seed 42

    BC_EXIT=$?
    if [ $BC_EXIT -ne 0 ]; then
        echo "BC failed with exit code $BC_EXIT"
        exit $BC_EXIT
    fi
    echo "=== BC complete ==="
    PRETRAINED="--pretrained checkpoints_v5_dimweights/controller_bc_best.pt"
else
    echo "=== Skipping BC (PPO checkpoint found, resuming) ==="
    PRETRAINED=""
fi

# --- Phase 2: PPO ---
echo "=== Phase 2: PPO ==="
RESUME_FLAG=""
if [ -f checkpoints_v5_dimweights/controller_ppo_latest.pt ]; then
    RESUME_FLAG="--resume"
fi

python -u scripts/train_controller_ppo.py \
    --config configs/v5.yaml \
    --wandb-name ppo-512d-v5fsq \
    --seed 42 \
    $PRETRAINED \
    $RESUME_FLAG &

# Wait for training process (needed for signal handling)
wait $!
echo "$(date): Training finished normally"
