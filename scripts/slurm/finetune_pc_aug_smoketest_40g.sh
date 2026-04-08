#!/bin/bash
#SBATCH --job-name=pc-aug-smoketest-40g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:40g
#SBATCH --partition=gpupr.4h
#SBATCH --time=4:00:00
#SBATCH --output=./logs/slurm_logs/%x-%j.out

set -x
cat $0
DIR=$(realpath .)
mkdir -p $DIR/logs/slurm_logs

echo "=== Smoke test (40GB): PC-Aug finetune, 50 steps, reduced batch ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"

module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6
source $DIR/venv/bin/activate
cd $DIR

nvidia-smi

# Reduced memory footprint to fit in 40GB A100:
#   - sequence_len 24→12 (~2x reduction in video/pointcloud tensors)
#   - traj_per_sample 384→128 (~3x reduction in track correlation tensors)
#   - fmaps_dim stays at 128 (must match pretrained checkpoint)
# 50 steps only — validates full pipeline end-to-end
PYTHONPATH=$DIR python -m mvtracker.cli.train \
  +experiment=mvtracker_human_finetune_pc_aug \
  datasets.train.sequence_len=12 \
  datasets.train.traj_per_sample=128 \
  trainer.num_steps=50 \
  trainer.eval_freq=50 \
  trainer.save_ckpt_freq=50 \
  trainer.viz_freq=50

echo "=== Smoke test complete ==="
