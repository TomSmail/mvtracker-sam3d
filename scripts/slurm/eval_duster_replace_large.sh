#!/bin/bash
#SBATCH --job-name=eval-dust-rlg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --time=04:00:00
#SBATCH --output=./logs/slurm_logs/%x-%j.out

set -ex
cat $0
DIR=$(realpath .)
mkdir -p $DIR/logs/slurm_logs

echo "=== DUSt3R + SAM3D depth-weighted + large replace (r=0.30) ==="
echo "Job ID: ${SLURM_JOB_ID} | Node: $(hostname) | GPU: ${CUDA_VISIBLE_DEVICES}"

module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6
source $DIR/venv/bin/activate
cd $DIR

nvidia-smi

PYTHONPATH=$DIR python -m mvtracker.cli.eval \
  +experiment=mvtracker_human_eval_duster_replace_large

echo "=== Eval complete ==="
