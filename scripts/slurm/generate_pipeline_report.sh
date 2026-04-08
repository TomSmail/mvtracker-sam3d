#!/bin/bash
#SBATCH --job-name=pipeline-report
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=1:00:00
#SBATCH --output=./logs/slurm_logs/%x-%j.out

# CPU-only — no GPU needed

set -ex
cat $0
DIR=$(realpath .)
mkdir -p $DIR/logs/slurm_logs $DIR/reports/pipeline_report

echo "=== Generating SAM3D data pipeline report ==="
echo "Job ID: ${SLURM_JOB_ID} | Node: $(hostname)"

module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6
source $DIR/venv/bin/activate
cd $DIR

PYTHONPATH=$DIR python scripts/data_engine/generate_pipeline_report.py \
    --data_root /cluster/scratch/tsmail/datasets/panoptic-multiview \
    --output_dir $DIR/reports/pipeline_report \
    --n_samples 6

echo "=== Report complete ==="
echo "Output: $DIR/reports/pipeline_report/report.html"
