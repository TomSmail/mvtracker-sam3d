#!/bin/bash
#SBATCH --job-name=eval-dust-dwr
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

echo "=== DUSt3R + SAM3D depth-weighted + replace (r=0.15) ==="
echo "Job ID: ${SLURM_JOB_ID} | Node: $(hostname) | GPU: ${CUDA_VISIBLE_DEVICES}"

module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6
source $DIR/venv/bin/activate
cd $DIR

nvidia-smi

PYTHONPATH=$DIR python -m mvtracker.cli.eval \
  +experiment=mvtracker_human_eval_duster_dw_replace

echo "=== Eval complete ==="
