#!/bin/bash
#SBATCH --job-name=eval-058
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --dependency=singleton
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=tsmail@ethz.ch
#SBATCH --output=./logs/slurm_logs/%x-%j.out
#SBATCH --array=0-74

set -x
cat $0
DIR=$(realpath .)
mkdir -p $DIR/runs

CKPTS=(
# "experiment_path=logs/eval/copycat     model=copycat"
# "experiment_path=logs/dynamic_3dgs     model=locotrack"
# "experiment_path=logs/shape_of_motion  model=locotrack"
#
# "experiment_path=logs/eval/tapip3d            model=tapip3d"
# "experiment_path=logs/eval/scenetracker       model=scenetracker"
# "experiment_path=logs/eval/locotrack          model=locotrack"
# "experiment_path=logs/eval/delta              model=delta"
# "experiment_path=logs/eval/cotracker2_online  model=cotracker2_online"
# "experiment_path=logs/eval/cotracker3_online  model=cotracker3_online"
#
# "experiment_path=logs/eval/spatracker_monocular_pretrained       model=spatracker_monocular_pretrained restore_ckpt_path=checkpoints/spatracker_monocular_original-authors-ckpt.pth"
# "experiment_path=logs/eval/spatracker_monocular_kubric-training  model=spatracker_monocular            restore_ckpt_path=checkpoints/spatracker_monocular_trained-on-kubric-depth_069800.pth"
# "experiment_path=logs/eval/spatracker_monocular_duster-training  model=spatracker_monocular            restore_ckpt_path=checkpoints/spatracker_monocular_trained-on-duster-depth_090800.pth"
# "experiment_path=logs/eval/spatracker_multiview_kubric-training  model=spatracker_multiview            restore_ckpt_path=checkpoints/spatracker_multiview_trained-on-kubric-depth_100000.pth model.triplane_xres=128 model.triplane_yres=128 model.triplane_zres=128"
# "experiment_path=logs/eval/spatracker_multiview_duster-training  model=spatracker_multiview            restore_ckpt_path=checkpoints/spatracker_multiview_trained-on-duster-depth_100000.pth model.triplane_xres=256 model.triplane_yres=256 model.triplane_zres=128"
#
# "experiment_path=logs/eval/mvtracker-v0_kubric-training  model=mvtracker  restore_ckpt_path=checkpoints/mvtracker_v0_trained-on-kubric-depth_091600.pth model.updatetransformer_type=spatracker model.apply_sigmoid_to_vis=true trainer.precision=16-mixed model.fmaps_dim=384 model.hidden_size=384 model.num_heads=8"
# "experiment_path=logs/eval/mvtracker-v0_duster-training  model=mvtracker  restore_ckpt_path=checkpoints/mvtracker_v0_trained-on-duster-depth_100000.pth model.updatetransformer_type=spatracker model.apply_sigmoid_to_vis=true trainer.precision=16-mixed model.fmaps_dim=384"
# "experiment_path=logs/eval/mvtracker-iccv-march2025      model=mvtracker  restore_ckpt_path=checkpoints/mvtracker_160000_march2025.pth model.updatetransformer_type=spatracker model.apply_sigmoid_to_vis=true trainer.precision=16-mixed "
 "experiment_path=logs/mvtracker                          model=mvtracker  restore_ckpt_path=checkpoints/mvtracker_200000_june2025.pth"
)
DATASETS=(
############################
### ~~~ Main results ~~~ ###
############################
  dex-ycb-multiview
  dex-ycb-multiview-duster0123
  dex-ycb-multiview-duster0123cleaned
  panoptic-multiview-views1_7_14_20
  panoptic-multiview-views27_16_14_8
  panoptic-multiview-views1_4_7_11
  kubric-multiview-v3-views0123
  kubric-multiview-v3-duster0123
  kubric-multiview-v3-duster0123cleaned
  tapvid2d-davis-mogewithextrinsics-256x256

#############################
### ~~~ 2DPT Ablation ~~~ ###
#############################
  dex-ycb-multiview-2dpt
  dex-ycb-multiview-duster0123-2dpt
  panoptic-multiview-views1_7_14_20-2dpt
  panoptic-multiview-views27_16_14_8-2dpt
  panoptic-multiview-views1_4_7_11-2dpt
  kubric-multiview-v3-views0123-2dpt
  kubric-multiview-v3-duster0123-2dpt

####################################
### ~~~ Single-point results ~~~ ###
####################################
#  dex-ycb-multiview-single
#  dex-ycb-multiview-duster0123-single
#  dex-ycb-multiview-duster0123cleaned-single
#  panoptic-multiview-views1_7_14_20-single
#  panoptic-multiview-views27_16_14_8-single
#  panoptic-multiview-views1_4_7_11-single
#  kubric-multiview-v3-views0123-single
#  kubric-multiview-v3-duster0123-single
#  kubric-multiview-v3-duster0123cleaned-single
#  tapvid2d-davis-mogewithextrinsics-256x256-single

#####################################
### ~~~ Camera-setup Ablation ~~~ ###
#####################################
  panoptic-multiview-views1_7_14_20
  panoptic-multiview-views27_16_14_8
  panoptic-multiview-views1_4_7_11
  dex-ycb-multiview-duster0123
  dex-ycb-multiview-duster2345
  dex-ycb-multiview-duster4567

########################################
### ~~~ Number-of-views Ablation ~~~ ###
########################################
  kubric-multiview-v3-views0
  kubric-multiview-v3-views01
  kubric-multiview-v3-views012
  kubric-multiview-v3-views0123
  kubric-multiview-v3-views01234
  kubric-multiview-v3-views012345
  kubric-multiview-v3-views0123456
  kubric-multiview-v3-views01234567
  kubric-multiview-v3-duster0123-views0
  kubric-multiview-v3-duster0123-views01
  kubric-multiview-v3-duster0123-views012
  kubric-multiview-v3-duster0123-views0123
  kubric-multiview-v3-duster01234567-views01234
  kubric-multiview-v3-duster01234567-views012345
  kubric-multiview-v3-duster01234567-views0123456
  kubric-multiview-v3-duster01234567-views01234567
  panoptic-multiview-views1
  panoptic-multiview-views1_14
  panoptic-multiview-views1_7_14
  panoptic-multiview-views1_7_14_20
  panoptic-multiview-views1_4_7_14_20
  panoptic-multiview-views1_4_7_14_17_20
  panoptic-multiview-views1_4_7_11_14_17_20
  panoptic-multiview-views1_4_7_11_14_17_20_23
  dex-ycb-multiview-duster0123-views0
  dex-ycb-multiview-duster0123-views01
  dex-ycb-multiview-duster0123-views012
  dex-ycb-multiview-duster0123-views0123
  dex-ycb-multiview-duster01234567-views01234
  dex-ycb-multiview-duster01234567-views012345
  dex-ycb-multiview-duster01234567-views0123456
  dex-ycb-multiview-duster01234567-views01234567


#####################################
### ~~~ For video comparisons ~~~ ###
#####################################
  kubric-multiview-v3-views0123-novelviews4
  panoptic-multiview-views1_7_14_20-novelviews24
  panoptic-multiview-views1_7_14_20-novelviews27
  dex-ycb-multiview-duster0123-novelviews4
  dex-ycb-multiview-duster0123-novelviews5
  dex-ycb-multiview-duster0123-novelviews6
  dex-ycb-multiview-duster0123-novelviews7
  dex-ycb-multiview-duster2345-novelviews7
  dex-ycb-multiview-duster4567-novelviews7
  dex-ycb-multiview-duster4567-novelviews0


####################################
### ~~~ For noise experiment ~~~ ###
####################################
  kubric-multiview-v3-noise0cm
  kubric-multiview-v3-noise1cm
  kubric-multiview-v3-noise2cm
  kubric-multiview-v3-noise5cm
  kubric-multiview-v3-noise10cm
  kubric-multiview-v3-noise20cm
  kubric-multiview-v3-noise50cm
  kubric-multiview-v3-noise100cm
  kubric-multiview-v3-noise200cm
  kubric-multiview-v3-noise1000cm
)

# Compute number of jobs needed
NUM_CKPTS=${#CKPTS[@]}
NUM_DATASETS=${#DATASETS[@]}
TOTAL_JOBS=$((NUM_CKPTS * NUM_DATASETS))

# Check if SLURM_ARRAY_TASK_ID is valid
if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL_JOBS" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID exceeds the max index $((TOTAL_JOBS-1))"
    exit 1
fi

# Map SLURM_ARRAY_TASK_ID to checkpoint and dataset
CKPT_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_CKPTS))
DATASET_INDEX=$((SLURM_ARRAY_TASK_ID / NUM_CKPTS))

SELECTED_CKPT=${CKPTS[$CKPT_INDEX]}
SELECTED_DATASET=${DATASETS[$DATASET_INDEX]}

echo "Selected Checkpoint: $SELECTED_CKPT"
echo "Selected Dataset: $SELECTED_DATASET"

# Load modules and activate venv
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6
source $DIR/venv/bin/activate
cd $DIR

# Run the job with the extracted checkpoint & dataset
PYTHONPATH=$DIR TORCH_HOME=./checkpoints/.cache python mvtracker/cli/eval.py $SELECTED_CKPT datasets.eval.names=[$SELECTED_DATASET]