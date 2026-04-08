#!/bin/bash
#SBATCH --job-name=gen-human-tracks
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=2:00:00
#SBATCH --output=./logs/slurm_logs/%x-%j_%a.out
#SBATCH --array=0-89

# No GPU needed — pure numpy/scipy pipeline
# Array covers all 90 sequence indices:
#   - 50 already have human_tracks.npz → skipped by skip_existing
#   - 22 have sam3d_predictions → will be processed
#   - 18 have no sam3d_predictions → skipped with warning

set -ex
cat $0
DIR=$(realpath .)
mkdir -p $DIR/logs/slurm_logs

echo "=== Generate human tracks: seq index ${SLURM_ARRAY_TASK_ID} ==="
echo "Job ID: ${SLURM_JOB_ID}, Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"

module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6
source $DIR/venv/bin/activate
cd $DIR

PYTHONPATH=$DIR python scripts/data_engine/generate_human_tracks.py \
    --data_root /cluster/scratch/tsmail/datasets/panoptic-multiview \
    --seq_index ${SLURM_ARRAY_TASK_ID} \
    --n_vertex_samples 500 \
    --smooth_kernel 3

echo "=== Done seq index ${SLURM_ARRAY_TASK_ID} ==="
