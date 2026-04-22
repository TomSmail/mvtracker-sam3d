#!/bin/bash
#SBATCH --job-name=eval-duster-v1471
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --time=04:00:00
#SBATCH --output=./logs/slurm_logs/%x-%j.out

set -x
cat $0
DIR=$(realpath .)
mkdir -p $DIR/logs/slurm_logs

echo "=== DUSt3R views1_4_7_11 evaluations (missing from previous run) ==="
echo "Job ID: ${SLURM_JOB_ID} | Node: $(hostname) | GPU: ${CUDA_VISIBLE_DEVICES}"

module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6
source $DIR/venv/bin/activate
cd $DIR

nvidia-smi

echo "=== 1/3: DUSt3R depth baseline ==="
PYTHONPATH=$DIR python -m mvtracker.cli.eval \
  +experiment=mvtracker_tapvid3d_eval_duster \
  'datasets.eval.names=[panoptic-multiview-duster1_4_7_11-views1_4_7_11-cached]' \
  || echo "FAILED: DUSt3R baseline v1471"

echo "=== 2/3: DUSt3R depth + kNN bias ==="
PYTHONPATH=$DIR python -m mvtracker.cli.eval \
  +experiment=mvtracker_tapvid3d_eval_duster_knn \
  'datasets.eval.names=[panoptic-multiview-duster1_4_7_11-views1_4_7_11-cached]' \
  || echo "FAILED: DUSt3R + kNN v1471"

echo "=== 3/3: DUSt3R depth + finetuned model ==="
PYTHONPATH=$DIR python -m mvtracker.cli.eval \
  +experiment=mvtracker_tapvid3d_eval_duster_finetuned \
  'datasets.eval.names=[panoptic-multiview-duster1_4_7_11-views1_4_7_11-cached]' \
  || echo "FAILED: DUSt3R + finetuned v1471"

echo "=== All evaluations complete ==="
