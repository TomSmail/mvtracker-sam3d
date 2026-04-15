#!/bin/bash
#SBATCH --job-name=regen-tracks-dense
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=4:00:00
#SBATCH --output=./logs/slurm_logs/%x-%j.out

# Regenerate human_tracks.npz for the 6 TAPVid-3D sequences with 5000 vertices
# instead of 500, to test if dense mesh sampling helps when depth is unavailable

set -ex
cat $0
DIR=$(realpath .)
mkdir -p $DIR/logs/slurm_logs

echo "=== Regenerate tracks with 5000 vertices for TAPVid-3D sequences ==="
echo "Job ID: ${SLURM_JOB_ID} | Node: $(hostname)"

module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6
source $DIR/venv/bin/activate
cd $DIR

# Get indices for the 6 TAPVid-3D sequences by finding them in sorted list
TAPVID_SEQS=("basketball" "boxes" "football" "juggle" "softball" "tennis")
DATA_ROOT="/cluster/scratch/tsmail/datasets/panoptic-multiview"

# Get all sequence names in sorted order
ALL_SEQS=($(ls -1 $DATA_ROOT | grep -v '^\.' | sort))

for target_seq in "${TAPVID_SEQS[@]}"; do
    # Find index in sorted array
    for i in "${!ALL_SEQS[@]}"; do
        if [[ "${ALL_SEQS[$i]}" == "$target_seq" ]]; then
            echo "=== Processing $target_seq (index $i) with 5000 vertices ==="
            PYTHONPATH=$DIR python scripts/data_engine/generate_human_tracks.py \
                --data_root $DATA_ROOT \
                --seq_index $i \
                --n_vertex_samples 5000 \
                --smooth_kernel 3 \
                --no_skip_existing
            break
        fi
    done
done

echo "=== All 6 sequences regenerated ==="
