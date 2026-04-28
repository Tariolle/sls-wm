#!/bin/bash
#SBATCH -J "train_beta_vae"
#SBATCH -o slurm/logs/train_beta_vae.out
#SBATCH -e slurm/logs/train_beta_vae.err
#SBATCH -p ar_h200
#SBATCH --gres=gpu:h200:1
#SBATCH -n 1
#SBATCH --cpus-per-gpu 8
#SBATCH --mem 64G
#SBATCH --time=08:00:00

# Beta-VAE training under V3-deploy's exact FSQ recipe.
# Submit:  sbatch slurm/train_beta_vae.sl
# Monitor: tail -f slurm/logs/train_beta_vae.out

module purge
module load aidl/pytorch/2.10.0-py3.12-cuda12.6
export PATH="$HOME/.local/bin:$PATH"

mkdir -p slurm/logs

echo "=== Train beta-VAE ($(date)) ==="
python -u scripts/train_beta_vae.py \
    --epochs 200 \
    --batch-size 32 \
    --lr 4e-3 \
    --beta 1.0 \
    --amp \
    --compile \
    --compile-mode reduce-overhead
