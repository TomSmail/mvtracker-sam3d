#!/bin/bash
#SBATCH --job-name=regen-dense
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=01:00:00
#SBATCH --output=./logs/slurm_logs/%x-%j.out

# Usage: sbatch regen_one_sequence_dense.sh basketball

set -ex
DIR=$(realpath .)
mkdir -p $DIR/logs/slurm_logs

SEQ_NAME=$1
echo "=== Regenerate $SEQ_NAME with 5000 vertices ==="
echo "Job ID: ${SLURM_JOB_ID}"

module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6
source $DIR/venv/bin/activate
cd $DIR

# Find sequence index
ALL_SEQS=($(ls -1 /cluster/scratch/tsmail/datasets/panoptic-multiview | grep -v '^\.' | sort))
SEQ_INDEX=-1
for i in "${!ALL_SEQS[@]}"; do
    if [[ "${ALL_SEQS[$i]}" == "$SEQ_NAME" ]]; then
        SEQ_INDEX=$i
        break
    fi
done

if [[ $SEQ_INDEX -eq -1 ]]; then
    echo "ERROR: Sequence $SEQ_NAME not found"
    exit 1
fi

echo "Sequence index: $SEQ_INDEX"

PYTHONPATH=$DIR python scripts/data_engine/generate_human_tracks.py \
    --data_root /cluster/scratch/tsmail/datasets/panoptic-multiview \
    --seq_index $SEQ_INDEX \
    --n_vertex_samples 5000 \
    --smooth_kernel 3 \
    --no_skip_existing

echo "=== Done $SEQ_NAME ==="
