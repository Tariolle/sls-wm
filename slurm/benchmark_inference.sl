#!/bin/bash
#SBATCH -J "bench_inf"
#SBATCH -o slurm/logs/benchmark_inference.out
#SBATCH -e slurm/logs/benchmark_inference.err
#SBATCH -p ar_a100
#SBATCH --gres=gpu:a100:1
#SBATCH -n 1
#SBATCH --cpus-per-gpu 4
#SBATCH --mem 16G
#SBATCH --time=00:05:00

# Benchmark inference latency on A100.
#
# Submit:  sbatch slurm/benchmark_inference.sl
# Output:  cat slurm/logs/benchmark_inference.out

module purge
module load aidl/pytorch/2.6.0-cuda12.6

python -u scripts/benchmark_inference.py
