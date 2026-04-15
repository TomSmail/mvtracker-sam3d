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
# Prerequisite: DUSt3R must be cloned to external/duster with checkpoint downloaded

set -ex
cat $0
DIR=$(realpath .)
mkdir -p $DIR/logs/slurm_logs

echo "=== Generate DUSt3R depths for TAPVid-3D sequences ==="
echo "Job ID: ${SLURM_JOB_ID} | Node: $(hostname) | GPU: ${CUDA_VISIBLE_DEVICES}"

module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6
source $DIR/venv/bin/activate
cd $DIR

# Install roma if not already present
pip install roma==1.5.1 --quiet

# Check that DUSt3R is set up
if [ ! -f "$DIR/external/duster/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth" ]; then
    echo "ERROR: DUSt3R not set up. Run the following locally or in an interactive job:"
    echo ""
    echo "  mkdir -p external && cd external"
    echo "  git clone --recursive https://github.com/naver/dust3r.git duster"
    echo "  cd duster"
    echo "  find croco dust3r -name '*.py' -exec sed -i 's/from models/from croco.models/g' {} +"
    echo "  mkdir -p checkpoints && cd checkpoints"
    echo "  wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    echo ""
    exit 1
fi

nvidia-smi

# Set PYTHONPATH to include DUSt3R
export PYTHONPATH=$DIR/external/duster:$PYTHONPATH

# NOTE: The script expects data in datasets/panoptic_d3dgs/ but ours is in
# /cluster/scratch/tsmail/datasets/panoptic-multiview. We need to either:
# 1. Create a symlink
# 2. Modify the script to use panoptic-multiview
# For now, create a symlink:
mkdir -p $DIR/datasets
ln -sf /cluster/scratch/tsmail/datasets/panoptic-multiview $DIR/datasets/panoptic_d3dgs

# Run DUSt3R depth estimation
cd $DIR
python scripts/estimate_depth_with_duster.py --dataset panoptic_d3dgs

echo "=== DUSt3R depth generation complete ==="
echo "Output directories: panoptic-multiview/{sequence}/duster-views-{viewstring}/"
