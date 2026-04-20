#!/bin/bash
#SBATCH --job-name=gen-dust-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --time=24:00:00
#SBATCH --output=./logs/slurm_logs/%x-%j.out

# Generate DUSt3R depths for training sequences (views 1,7,14,20 only)
# These sequences already have SAM3D human_tracks.npz but no depth maps.

set -ex
cat $0
DIR=$(realpath .)
mkdir -p $DIR/logs/slurm_logs

echo "=== Generate DUSt3R depths for training sequences ==="
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

python -c "
import sys, os, numpy as np
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

data_root = Path('$DIR/datasets/panoptic-multiview/')
views_selection = [1, 7, 14, 20]  # only the training view config

# Find sequences that need DUSt3R depths
sequences = []
for scene_root in sorted(data_root.glob('[!.]*')):
    tracks_path = scene_root / 'human_tracks.npz'
    if not tracks_path.exists():
        continue
    d = np.load(str(tracks_path), allow_pickle=True)
    if 'trajectories' not in list(d.keys()):
        continue
    duster_dir = scene_root / f'duster-views-{chr(45).join(map(str, views_selection))}'
    if duster_dir.exists():
        print(f'Skipping {scene_root.name}: DUSt3R already exists')
        continue
    ims_dir = scene_root / 'ims'
    if not all((ims_dir / str(v)).is_dir() for v in views_selection):
        print(f'Skipping {scene_root.name}: missing view directories')
        continue
    n_frames = len(list((ims_dir / '1').iterdir()))
    if n_frames < 24:
        print(f'Skipping {scene_root.name}: only {n_frames} frames')
        continue
    sequences.append(scene_root)

print(f'\nWill process {len(sequences)} sequences')
for i, scene_root in enumerate(sequences):
    print(f'\n[{i+1}/{len(sequences)}] Processing {scene_root.name} with views {views_selection}')
    main_on_d3dgs_panoptic_scene(scene_root, views_selection, **duster_kwargs)

print('Done.')
"

echo "=== DUSt3R depth generation complete ==="
