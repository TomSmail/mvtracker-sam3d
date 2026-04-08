#!/bin/bash
#SBATCH --job-name=eval-baseline-human
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:80g
#SBATCH --partition=gpupr.24h
#SBATCH --time=24:00:00
#SBATCH --output=./logs/slurm_logs/%x-%j.out

set -x
cat $0
DIR=$(realpath .)
mkdir -p $DIR/logs/slurm_logs

echo "=== Baseline eval: pure MVTracker (no SAM3D) on Panoptic human datasets ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"

module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6
source $DIR/venv/bin/activate
cd $DIR

nvidia-smi

# Eval-only: pretrained checkpoint, no SAM3D augmentation, no finetuning
PYTHONPATH=$DIR python -m mvtracker.cli.eval \
  +experiment=mvtracker_human_eval_baseline

echo "=== Eval complete ==="
