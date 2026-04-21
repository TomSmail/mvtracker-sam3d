#!/bin/bash
#SBATCH --job-name=ft-dust-16seq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:80g
#SBATCH --time=24:00:00
#SBATCH --output=./logs/slurm_logs/%x-%j.out

# Finetune MVTracker with frozen encoder on 16 Panoptic sequences (DUSt3R views 0,1,2,3)
# Previous 4-sequence attempts showed catastrophic forgetting due to data scarcity.

set -ex
cat $0
DIR=$(realpath .)
mkdir -p $DIR/logs/slurm_logs

echo "=== Finetune: frozen encoder, 16 sequences, DUSt3R depths ==="
echo "Job ID: ${SLURM_JOB_ID} | Node: $(hostname) | GPU: ${CUDA_VISIBLE_DEVICES}"

module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6
source $DIR/venv/bin/activate
cd $DIR

nvidia-smi

PYTHONPATH=$DIR python -m mvtracker.cli.train \
  +experiment=mvtracker_human_finetune_duster_16seq

echo "=== Finetune complete ==="
