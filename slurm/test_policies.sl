#!/bin/bash
#SBATCH -J "test_pol"
#SBATCH -o slurm/logs/test_policies.out
#SBATCH -e slurm/logs/test_policies.err
#SBATCH -p ar_a100
#SBATCH --gres=gpu:a100:1
#SBATCH -n 1
#SBATCH --cpus-per-gpu 8
#SBATCH --mem 64G
#SBATCH --time=00:10:00

module purge
module load aidl/pytorch/2.6.0-cuda12.6

python -u scripts/test_hardcoded_policies.py
