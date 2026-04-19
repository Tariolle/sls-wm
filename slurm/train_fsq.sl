#!/bin/bash
#SBATCH -J "train_fsq"
#SBATCH -o slurm/logs/train_fsq.out
#SBATCH -e slurm/logs/train_fsq.err
#SBATCH -p ar_a100
#SBATCH --gres=gpu:a100:1
#SBATCH -n 1
#SBATCH --cpus-per-gpu 8
#SBATCH --mem 64G
#SBATCH --time=08:00:00

# Train FSQ-VAE on A100 with bf16 AMP and torch.compile.
# Auto-resumes from fsq_state.pt if a previous run was interrupted.
#
# Submit:  sbatch slurm/train_fsq.sl [config]
# Example: sbatch slurm/train_fsq.sl configs/v5.yaml
# Monitor: tail -f slurm/logs/train_fsq.out

CONFIG=${1:-configs/v5.yaml}

module purge
module load aidl/pytorch/2.10.0-py3.12-cuda12.6
export PATH="$HOME/.local/bin:$PATH"
pip install --user --upgrade wandb "protobuf>=6.32" 2>/dev/null

echo "=== Config: $CONFIG ==="

# Shift augmentation moved on-the-fly inside scripts/train_world_model.py
# (joint mode). Pre-shift step removed 2026-04-19.

# Auto-resume: extract checkpoint_dir from config, check for fsq_state.pt
CKPT_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG')).get('fsq',{}).get('checkpoint_dir','checkpoints'))")
RESUME_ARG=""
if [ -f "$CKPT_DIR/fsq_state.pt" ]; then
    RESUME_ARG="--resume $CKPT_DIR/fsq_state.pt"
    echo "=== Resuming from $CKPT_DIR/fsq_state.pt ==="
fi

echo "=== Step 1: Train FSQ-VAE ==="
python -u scripts/train_fsq.py \
    --config "$CONFIG" \
    $RESUME_ARG
