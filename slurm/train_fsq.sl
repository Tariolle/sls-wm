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
# Example: sbatch slurm/train_fsq.sl configs/v5b.yaml
# Monitor: tail -f slurm/logs/train_fsq.out

CONFIG=${1:-configs/v5.yaml}

module purge
module load aidl/pytorch/2.6.0-cuda12.6
export PATH="$HOME/.local/bin:$PATH"
pip install --user wandb 2>/dev/null

echo "=== Config: $CONFIG ==="

echo "=== Step 0: Pre-compute shift augmentation ==="
python -u scripts/shift_episodes.py \
    --episodes-dir data/death_episodes \
    --expert-episodes-dir data/expert_episodes \
    --shifts-v -4 -2 0 2 4

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
