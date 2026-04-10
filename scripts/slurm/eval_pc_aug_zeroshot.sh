#!/bin/bash
#SBATCH --job-name=eval-pc-aug-zeroshot
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

echo "=== Zero-shot PC-aug eval: pretrained MVTracker + SAM3D vertices, no finetuning ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"

module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6
source $DIR/venv/bin/activate
cd $DIR

nvidia-smi

# Eval-only: pretrained checkpoint + SAM3D pointcloud augmentation, no finetuning
PYTHONPATH=$DIR python -m mvtracker.cli.eval \
  +experiment=mvtracker_human_eval_pc_aug_zeroshot

echo "=== Eval complete ==="
