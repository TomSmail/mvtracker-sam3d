#!/bin/bash
#SBATCH --job-name=eval-tapvid3d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --time=08:00:00
#SBATCH --output=./logs/slurm_logs/%x-%j.out

set -ex
cat $0
DIR=$(realpath .)
mkdir -p $DIR/logs/slurm_logs

echo "=== Official TAPVid-3D benchmark evaluation (all configs) ==="
echo "Job ID: ${SLURM_JOB_ID} | Node: $(hostname) | GPU: ${CUDA_VISIBLE_DEVICES}"

module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6
source $DIR/venv/bin/activate
cd $DIR

nvidia-smi

echo "=== 1/4: D3DGS depth baseline ==="
PYTHONPATH=$DIR python -m mvtracker.cli.eval \
  +experiment=mvtracker_tapvid3d_eval_d3dgs

echo "=== 2/4: DUSt3R depth baseline ==="
PYTHONPATH=$DIR python -m mvtracker.cli.eval \
  +experiment=mvtracker_tapvid3d_eval_duster

echo "=== 3/4: DUSt3R depth + kNN bias ==="
PYTHONPATH=$DIR python -m mvtracker.cli.eval \
  +experiment=mvtracker_tapvid3d_eval_duster_knn

echo "=== 4/4: DUSt3R depth + finetuned model ==="
PYTHONPATH=$DIR python -m mvtracker.cli.eval \
  +experiment=mvtracker_tapvid3d_eval_duster_finetuned

echo "=== All evaluations complete ==="
