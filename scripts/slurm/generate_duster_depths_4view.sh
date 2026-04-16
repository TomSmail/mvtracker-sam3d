#!/bin/bash
#SBATCH --job-name=gen-duster-4v
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --time=24:00:00
#SBATCH --output=./logs/slurm_logs/%x-%j.out

# Generate DUSt3R depths for 4-view configs only (matching our eval configs)

set -ex
cat $0
DIR=$(realpath .)
mkdir -p $DIR/logs/slurm_logs

echo "=== Generate DUSt3R depths (4-view configs) ==="
echo "Job ID: ${SLURM_JOB_ID} | Node: $(hostname) | GPU: ${CUDA_VISIBLE_DEVICES}"

module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6
source $DIR/venv/bin/activate
cd $DIR

if [ ! -f "$DIR/external/duster/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth" ]; then
    echo "ERROR: DUSt3R not set up."
    exit 1
fi

nvidia-smi

export PYTHONPATH=$DIR/external/duster:$PYTHONPATH

# Run with only the 4-view selections we need
python -c "
import sys
sys.path.insert(0, '$DIR/external/duster')
sys.path.insert(0, '$DIR')

from pathlib import Path
from scripts.estimate_depth_with_duster import main_on_d3dgs_panoptic_scene

duster_kwargs = {
    'model_name_or_path': '$DIR/external/duster/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth',
    'silent': False,
    'output_2d_matches': False,
    'dump_exhaustive_data': False,
    'save_ply': False,
    'save_png_viz': False,
    'show_debug_plots': False,
    'skip_if_output_already_exists': True,
    'save_rerun_viz': False,
    'frame_selection': None,
}

data_root = Path('$DIR/datasets/panoptic_d3dgs/')
views_selections = [
    [1, 7, 14, 20],   # matches panoptic-human-views1_7_14_20
    [27, 16, 14, 8],   # matches panoptic-human-views27_16_14_8
    [1, 4, 7, 11],     # matches panoptic-human-views1_4_7_11
]

for scene_root in sorted(data_root.glob('[!.]*')):
    for views_selection in views_selections:
        print(f'Processing {scene_root.name} with views {views_selection}')
        main_on_d3dgs_panoptic_scene(scene_root, views_selection, **duster_kwargs)

print('Done.')
"

echo "=== DUSt3R depth generation complete ==="
