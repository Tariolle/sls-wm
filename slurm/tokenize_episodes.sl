#!/bin/bash
#SBATCH -J "tokenize"
#SBATCH -o slurm/logs/tokenize.out
#SBATCH -e slurm/logs/tokenize.err
#SBATCH -p ar_h200
#SBATCH --gres=gpu:h200:1
#SBATCH -n 1
#SBATCH --cpus-per-gpu 8
#SBATCH --mem 64G
#SBATCH --time=04:00:00

# V7 Phase 0: tokenize episodes with frozen FSQ + vertical shift augmentation
# for the transformer training stage.
#
# Submit:  sbatch slurm/tokenize_episodes.sl [config]
# Default config: configs/v7-phase0.yaml

CONFIG=${1:-configs/v7-phase0.yaml}

CKPT_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG')).get('fsq',{}).get('checkpoint_dir','checkpoints'))")
LEVELS=$(python -c "import yaml; print(' '.join(str(x) for x in yaml.safe_load(open('$CONFIG')).get('model',{}).get('levels',[8,5,5,5])))")

echo "=== Config: $CONFIG ==="
echo "=== FSQ checkpoint: $CKPT_DIR/fsq_best.pt ==="
echo "=== Levels: $LEVELS ==="

module purge
module load aidl/pytorch/2.10.0-py3.12-cuda12.6
export PATH="$HOME/.local/bin:$PATH"

# V3-deploy default: vertical-only shifts at [-4,-2,0,2,4]
# Tokenize death + expert dirs separately so each gets their aug_dirs.
for EP_DIR in data/death_episodes data/expert_episodes; do
    if [ -d "$EP_DIR" ]; then
        echo "=== Tokenizing $EP_DIR ($(date)) ==="
        python -u scripts/tokenize_episodes.py \
            --episodes-dir "$EP_DIR" \
            --checkpoint "$CKPT_DIR/fsq_best.pt" \
            --levels $LEVELS \
            --shifts-v -4 -2 0 2 4
    fi
done
echo "=== Done ($(date)) ==="
