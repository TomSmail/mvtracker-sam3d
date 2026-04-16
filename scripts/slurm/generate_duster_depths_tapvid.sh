#!/bin/bash
#SBATCH --job-name=gen-duster-tapvid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:80g
#SBATCH --time=24:00:00
#SBATCH --output=./logs/slurm_logs/%x-%j.out

# Generate DUSt3R depth maps for the 6 TAPVid-3D sequences
# Prerequisite: Run scripts/setup_duster.sh first (one-time setup)

set -ex
cat $0
DIR=$(realpath .)
mkdir -p $DIR/logs/slurm_logs

echo "=== Generate DUSt3R depths for TAPVid-3D sequences ==="
echo "Job ID: ${SLURM_JOB_ID} | Node: $(hostname) | GPU: ${CUDA_VISIBLE_DEVICES}"

module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6
source $DIR/venv/bin/activate
cd $DIR

# Verify DUSt3R is set up
if [ ! -f "$DIR/external/duster/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth" ]; then
    echo "ERROR: DUSt3R not set up. Run: bash scripts/setup_duster.sh"
    exit 1
fi

nvidia-smi

# datasets/panoptic_d3dgs should already contain symlinks to the 6 TAPVid-3D sequences
# (set up by: mkdir panoptic_d3dgs && ln -sf panoptic-multiview/{seq} panoptic_d3dgs/{seq})
if [ ! -d "$DIR/datasets/panoptic_d3dgs" ]; then
    echo "ERROR: datasets/panoptic_d3dgs not set up. Create symlinks to the 6 TAPVid-3D sequences."
    exit 1
fi

# Set PYTHONPATH to include DUSt3R
export PYTHONPATH=$DIR/external/duster:$PYTHONPATH

# Run DUSt3R depth estimation
# This will generate duster-views-{viewstring}/ folders for all 3 view configs we use
python scripts/estimate_depth_with_duster.py --dataset panoptic_d3dgs

echo "=== DUSt3R depth generation complete ==="
echo "Output: /cluster/scratch/tsmail/datasets/panoptic-multiview/{sequence}/duster-views-*/"
