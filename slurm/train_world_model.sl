#!/bin/bash
#SBATCH -J "train_wm"
#SBATCH -o slurm/logs/train_world_model.out
#SBATCH -e slurm/logs/train_world_model.err
#SBATCH -p ar_a100
#SBATCH --gres=gpu:a100:1
#SBATCH -n 1
#SBATCH --cpus-per-gpu 8
#SBATCH --mem 64G
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@300

# Train the world model (FSQ + Transformer) on A100 via
# scripts/train_world_model.py. Joint mode is the forward path; FSQ is
# trained jointly with the transformer, no pre-tokenisation required.
# Current E6.x candidate is configs/e6.5-jepa.yaml (pure-JEPA: joint +
# CWU-reg as sole anti-collapse, recon + decoder dropped from training,
# post-hoc decoder trained separately via scripts/train_posthoc_decoder.py).
# Earlier E6.4 kept recon as a pixel anchor; see configs/e6.4-cwureg.yaml.
# V5 tokenised-input mode is legacy.
# Auto-resumes: SLURM sends USR1 5 min before time limit, the trap
# saves checkpoint and resubmits the job.
#
# Submit:  sbatch slurm/train_world_model.sl [config]
# Example: sbatch slurm/train_world_model.sl configs/v5.yaml
# Monitor: tail -f slurm/logs/train_world_model.out

CONFIG=${1:-configs/v5.yaml}

# Extract checkpoint_dir from the transformer section of the config.
CKPT_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG')).get('transformer',{}).get('checkpoint_dir','checkpoints'))")

echo "=== Config: $CONFIG ==="
echo "=== Checkpoint dir: $CKPT_DIR ==="

# Auto-resume: the trap creates a sentinel file before requeuing.
RESUME_FLAG="$CKPT_DIR/.resume_transformer"

handle_timeout() {
    echo "=== USR1 received ($(date)), saving and resubmitting ==="
    # Persist resume intent BEFORE waiting for the child: if Python's save
    # overruns the grace period and we get hard-killed, a fresh submit will
    # still pick up --resume from the flag.
    mkdir -p "$CKPT_DIR"
    touch "$RESUME_FLAG"
    # Queue the resubmit immediately and unconditionally. With the cluster's
    # 1-job-at-a-time policy, the new job sits in the queue until this one
    # finishes saving — no race, no scontrol-requeue silent-failure path.
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
# wandb bundles gencode that expects protobuf >= 6.32; the torch 2.10
# module ships protobuf 6.31, so upgrade it here on the compute node.
pip install --user --upgrade wandb "protobuf>=6.32" 2>/dev/null

RESUME_ARG=""
if [ -f "$RESUME_FLAG" ]; then
    RESUME_ARG="--resume"
    rm "$RESUME_FLAG"
    echo "=== Resuming from checkpoint ==="
fi

echo "=== Train world model ($(date)) ==="
python -u scripts/train_world_model.py \
    --config "$CONFIG" \
    $RESUME_ARG &

TRAIN_PID=$!
wait "$TRAIN_PID"
