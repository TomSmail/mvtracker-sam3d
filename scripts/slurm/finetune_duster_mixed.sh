#!/bin/bash
#SBATCH --job-name=ft-duster-mix
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:80g
#SBATCH --time=24:00:00
#SBATCH --output=./logs/slurm_logs/%x-%j.out

# Finetuning with frozen encoder + mixed Kubric/Panoptic(DUSt3R) data
# Fixes: encoder freezing, honest DUSt3R depths, Kubric replay for anti-forgetting

set -ex
cat $0
DIR=$(realpath .)
mkdir -p $DIR/logs/slurm_logs

echo "=== Finetune: frozen encoder + DUSt3R + mixed data ==="
echo "Job ID: ${SLURM_JOB_ID} | Node: $(hostname) | GPU: ${CUDA_VISIBLE_DEVICES}"

module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6
source $DIR/venv/bin/activate
cd $DIR

nvidia-smi

PYTHONPATH=$DIR python -m mvtracker.cli.train \
  +experiment=mvtracker_human_finetune_duster_mixed

echo "=== Finetune complete ==="
