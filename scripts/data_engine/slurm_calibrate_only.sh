#!/bin/bash
#SBATCH --job-name=calibrate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:30:00
#SBATCH --output=./logs/slurm_logs/%x-%A_%a.out

set -euo pipefail

DIR=$(realpath .)
mkdir -p $DIR/logs/slurm_logs

SCRATCH_DIR="${SCRATCH:-/cluster/scratch/tsmail}"
RAW_ROOT="${SCRATCH_DIR}/datasets/panoptic-raw"
OUTPUT_ROOT="${SCRATCH_DIR}/datasets/panoptic-multiview"

echo "=== Calibrate only: Task ${SLURM_ARRAY_TASK_ID} ==="
echo "Node: $(hostname)"

module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 ffmpeg/6.0
source $DIR/venv/bin/activate

python scripts/data_engine/create_calibration.py \
    --raw_root "${RAW_ROOT}" \
    --output_root "${OUTPUT_ROOT}" \
    --seq_index "${SLURM_ARRAY_TASK_ID}"

echo "=== Calibration complete for task ${SLURM_ARRAY_TASK_ID} ==="
