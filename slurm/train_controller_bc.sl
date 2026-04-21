#!/bin/bash
#SBATCH -J "train_ctrl_bc"
#SBATCH -o slurm/logs/train_controller_bc.out
#SBATCH -e slurm/logs/train_controller_bc.err
#SBATCH -p ar_a100
#SBATCH --gres=gpu:a100:1
#SBATCH -n 1
#SBATCH --cpus-per-gpu 8
#SBATCH --mem 64G
#SBATCH --time=01:00:00

# Behavioral cloning: pretrain controller on expert episodes.
# Should be fast (~33K samples, 50 epochs).
#
# Submit:  sbatch slurm/train_controller_bc.sl [config]
# Example: sbatch slurm/train_controller_bc.sl configs/e6.7-recon-cauchysls.yaml

CONFIG=${1:-configs/e6.7-recon-cauchysls.yaml}
CKPT_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG')).get('transformer',{}).get('checkpoint_dir','checkpoints'))")

module purge
module load aidl/pytorch/2.10.0-py3.12-cuda12.6
export PATH="$HOME/.local/bin:$PATH"
pip install --user --upgrade wandb "protobuf>=6.32" 2>/dev/null

echo "=== Train Controller (BC) ==="
python -u scripts/train_controller_bc.py \
    --config "$CONFIG" \
    --fsq-checkpoint "$CKPT_DIR/fsq_best.pt" \
    --seed 42
