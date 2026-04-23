#!/bin/bash
#SBATCH -J "train_ctrl"
#SBATCH -o slurm/logs/train_controller.out
#SBATCH -e slurm/logs/train_controller.err
#SBATCH -p ar_h200
#SBATCH --gres=gpu:h200:1
#SBATCH -n 1
#SBATCH --cpus-per-gpu 8
#SBATCH --mem 64G
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@300

# BC pretraining then PPO fine-tuning, USR1 auto-resume.
# Submit: sbatch slurm/train_controller.sl [config]

CONFIG=${1:-configs/e6.10-gaussian-single-group.yaml}

# Extract checkpoint_dir from the transformer section of the config.
CKPT_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG')).get('transformer',{}).get('checkpoint_dir','checkpoints'))")

echo "=== Config: $CONFIG ==="
echo "=== Checkpoint dir: $CKPT_DIR ==="

cleanup_and_resubmit() {
    echo "$(date): Caught signal, saving and resubmitting..."
    sbatch "$0" "$CONFIG"
    exit 0
}
trap cleanup_and_resubmit USR1

module purge
module load aidl/pytorch/2.10.0-py3.12-cuda12.6
export PATH="$HOME/.local/bin:$PATH"
pip install --user --upgrade wandb "protobuf>=6.32" 2>/dev/null

# Uncomment to force synchronous CUDA launches for debugging (slow, but
# reports the actual faulting kernel instead of CUBLAS_NOT_INITIALIZED).
# export CUDA_LAUNCH_BLOCKING=1

echo "$(date): Starting on $(hostname), Job ID: $SLURM_JOB_ID"

# --- Phase 1: BC (skipped if PPO checkpoint exists) ---
if [ ! -f "$CKPT_DIR/controller_ppo_latest.pt" ]; then
    echo "=== Phase 1: Behavioral Cloning ==="
    python -u scripts/train_controller_bc.py \
        --config "$CONFIG" \
        --fsq-checkpoint "$CKPT_DIR/fsq_best.pt" \
        --seed 42

    BC_EXIT=$?
    if [ $BC_EXIT -ne 0 ]; then
        echo "BC failed with exit code $BC_EXIT"
        exit $BC_EXIT
    fi
    echo "=== BC complete ==="
    PRETRAINED="--pretrained $CKPT_DIR/controller_bc_best.pt"
else
    echo "=== Skipping BC (PPO checkpoint found, resuming) ==="
    PRETRAINED=""
fi

# --- Phase 2: PPO ---
echo "=== Phase 2: PPO ==="
RESUME_FLAG=""
if [ -f "$CKPT_DIR/controller_ppo_latest.pt" ]; then
    RESUME_FLAG="--resume"
fi

python -u scripts/train_controller_ppo.py \
    --config "$CONFIG" \
    --fsq-checkpoint "$CKPT_DIR/fsq_best.pt" \
    --seed 42 \
    $PRETRAINED \
    $RESUME_FLAG &

# Wait for training process (needed for signal handling)
wait $!
echo "$(date): Training finished normally"
