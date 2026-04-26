#!/bin/bash
#SBATCH -J "train_fsq"
#SBATCH -o slurm/logs/train_fsq.out
#SBATCH -e slurm/logs/train_fsq.err
#SBATCH -p ar_h200
#SBATCH --gres=gpu:h200:1
#SBATCH -n 1
#SBATCH --cpus-per-gpu 8
#SBATCH --mem 64G
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@300

# V7 Phase 0: FSQ-VAE training with auto-resume on USR1.
#
# Submit:  sbatch slurm/train_fsq.sl [config]
# Default config: configs/v7-phase0.yaml
# Monitor: tail -f slurm/logs/train_fsq.out

CONFIG=${1:-configs/v7-phase0.yaml}

CKPT_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG')).get('fsq',{}).get('checkpoint_dir','checkpoints'))")

echo "=== Config: $CONFIG ==="
echo "=== Checkpoint dir: $CKPT_DIR ==="

RESUME_FLAG="$CKPT_DIR/.resume_fsq"

handle_timeout() {
    echo "=== USR1 received ($(date)), saving and resubmitting ==="
    mkdir -p "$CKPT_DIR"
    touch "$RESUME_FLAG"
    echo "=== Submitting resume job ==="
    sbatch "$0" "$CONFIG"
    kill -TERM "$TRAIN_PID" 2>/dev/null
    wait "$TRAIN_PID"
    exit 0
}
trap handle_timeout USR1

module purge
module load aidl/pytorch/2.10.0-py3.12-cuda12.6
export PATH="$HOME/.local/bin:$PATH"

RESUME_ARG=""
if [ -f "$RESUME_FLAG" ]; then
    RESUME_ARG="--resume $CKPT_DIR/fsq_final.pt"
    rm "$RESUME_FLAG"
    echo "=== Resuming from $CKPT_DIR/fsq_final.pt ==="
fi

echo "=== Train FSQ ($(date)) ==="
python -u scripts/train_fsq.py \
    --config "$CONFIG" \
    $RESUME_ARG &

TRAIN_PID=$!
wait "$TRAIN_PID"
