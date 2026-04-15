#!/bin/bash
# One-time setup for DUSt3R (run interactively or in an interactive SLURM job)
# Usage: bash scripts/setup_duster.sh

set -ex

DIR=$(pwd)
echo "Working directory: $DIR"

# Create external directory
mkdir -p $DIR/external
cd $DIR/external

# Clone DUSt3R
if [ -d "duster" ]; then
    echo "DUSt3R already cloned, pulling latest..."
    cd duster && git pull && cd ..
else
    echo "Cloning DUSt3R..."
    git clone --recursive https://github.com/naver/dust3r.git duster
fi

cd duster

# Fix imports (models -> croco.models)
echo "Fixing imports..."
find croco dust3r -name "*.py" -exec sed -i 's/from models/from croco.models/g' {} \;

# Download checkpoint
mkdir -p checkpoints
cd checkpoints

if [ -f "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth" ]; then
    echo "Checkpoint already exists, verifying..."
    CHECKSUM=$(md5sum DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth | awk '{print $1}')
    if [ "$CHECKSUM" = "c3fab9b455b03f23d20e6bf77f2607bb" ]; then
        echo "Checkpoint verified"
    else
        echo "WARNING: Checkpoint checksum mismatch, re-downloading..."
        rm DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
        wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
    fi
else
    echo "Downloading DUSt3R checkpoint..."
    wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
    md5sum DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
fi

cd $DIR

# Install roma
source venv/bin/activate
pip install roma==1.5.1

echo ""
echo "=== DUSt3R setup complete ==="
echo "DUSt3R location: $DIR/external/duster"
echo "Checkpoint: $DIR/external/duster/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
echo ""
echo "Now you can run: sbatch scripts/slurm/generate_duster_depths_tapvid.sh"
