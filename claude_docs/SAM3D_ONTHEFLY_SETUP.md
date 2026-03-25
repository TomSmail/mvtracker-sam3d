# On-the-Fly SAM3D Inference Setup Guide

This guide explains how to enable real-time SAM3D inference during MVTracker evaluation.

## Overview

Instead of pre-computing SAM3D fits offline, the system now runs SAM3D inference **on-the-fly** during evaluation:
- SAM3D processes each frame as it's loaded
- Outputs are transformed to world coordinates
- MVTracker uses them for semantic-guided kNN search

---

## Prerequisites

### 1. SAM3D Model Checkpoints

Download SAM3D checkpoints:

```bash
# Option A: From Hugging Face (recommended)
mkdir -p ~/checkpoints/sam-3d-body-dinov3
cd ~/checkpoints/sam-3d-body-dinov3

# Download using git-lfs
git lfs install
git clone https://huggingface.co/facebook/sam-3d-body

# Or manually download:
# - model.ckpt (~600MB)
# - assets/mhr_model.pt (~50MB)
# - model_config.yaml
```

Default paths expected:
- Checkpoint: `~/checkpoints/sam-3d-body-dinov3/model.ckpt`
- MHR model: `~/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt`

### 2. SAM3D Dependencies

SAM3D dependencies should already be installed if you set up the environment:

```bash
# Verify SAM3D imports work
python -c "from sam_3d_body import load_sam_3d_body; print('SAM3D OK')"
```

If imports fail:
```bash
cd sam-3d-body
pip install -e .
```

---

## Configuration

### Method 1: Update Experiment Config (Recommended)

Edit your experiment config to enable SAM3D:

**File:** `configs/experiment/mvtracker_dexycb_eval_sam3d.yaml`

```yaml
# @package _global_
defaults:
  - override /model: mvtracker

experiment_path: ./logs/mvtracker_dexycb_eval_sam3d
restore_ckpt_path: ./checkpoints/mvtracker_200000_june2025.pth

model:
  use_sam3d_knn_bias: true   # Enable SAM3D guidance
  sam3d_knn_overfetch: 4     # Over-fetch 4x candidates

datasets:
  root: /cluster/scratch/tsmail/datasets
  eval:
    names:
      - dex-ycb-multiview-views0123-cached
      - dex-ycb-multiview-views0123-removehand-cached
    num_workers: 0
    max_seq_len: 300
    # SAM3D on-the-fly inference settings
    use_sam3d: true
    sam3d_checkpoint_path: ~/checkpoints/sam-3d-body-dinov3/model.ckpt
    sam3d_mhr_path: ~/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt

modes:
  eval_only: true

trainer:
  precision: bf16-mixed

logging:
  log_wandb: false
```

### Method 2: Command-Line Override

Override at runtime:

```bash
python -m mvtracker.cli.eval \
  +experiment=mvtracker_dexycb_eval_sam3d \
  datasets.eval.use_sam3d=true \
  datasets.eval.sam3d_checkpoint_path=~/checkpoints/sam-3d-body-dinov3/model.ckpt \
  datasets.eval.sam3d_mhr_path=~/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt
```

---

## Running Evaluation

### Option 1: Using SLURM Scripts

Update the SLURM script to pass SAM3D params:

**File:** `scripts/slurm/eval_dexycb_sam3d.sh`

```bash
#!/bin/bash
#SBATCH --job-name=eval-dexycb-sam3d
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --gres=gpumem:80g
#SBATCH --mem-per-cpu=32G
#SBATCH --time=08:00:00
#SBATCH --output=./logs/slurm_logs/%x-%j.out

set -x

DIR=$(realpath .)
cd $DIR

# Load environment
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6
source $DIR/venv/bin/activate

# Check GPU
nvidia-smi

echo "=== Running DexYCB Evaluation with On-the-Fly SAM3D ==="

PYTHONPATH=$DIR TORCH_HOME=./checkpoints/.cache python -m mvtracker.cli.eval \
  +experiment=mvtracker_dexycb_eval_sam3d \
  datasets.eval.use_sam3d=true \
  datasets.eval.sam3d_checkpoint_path=~/checkpoints/sam-3d-body-dinov3/model.ckpt \
  datasets.eval.sam3d_mhr_path=~/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt

echo "=== Evaluation complete ==="
```

Submit:
```bash
sbatch scripts/slurm/eval_dexycb_sam3d.sh
```

### Option 2: Direct Python Execution

```bash
# Activate environment
source venv/bin/activate

# Run evaluation
PYTHONPATH=. python -m mvtracker.cli.eval \
  +experiment=mvtracker_dexycb_eval_sam3d \
  datasets.eval.use_sam3d=true \
  datasets.eval.sam3d_checkpoint_path=~/checkpoints/sam-3d-body-dinov3/model.ckpt \
  datasets.eval.sam3d_mhr_path=~/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt
```

---

## Expected Behavior

### Timeline

1. **Dataset initialization** (~10s)
   - Loads dataset metadata
   - Initializes SAM3D model (lazy, on first batch)

2. **First batch** (~60s)
   - SAM3D model loads (one-time cost)
   - Detector and segmentor initialize
   - Inference runs on first sequence

3. **Subsequent batches** (~10-20s each)
   - SAM3D inference only
   - MVTracker inference
   - Metric computation

### Log Output

You should see:
```
Loading 10 videos from /cluster/scratch/tsmail/datasets/dex-ycb-multiview
Initializing SAM3D model from ~/checkpoints/sam-3d-body-dinov3/model.ckpt
Loading SAM 3D Body model...
SAM3D model initialized successfully
Running SAM3D inference on 20200709-subject-01...
SAM3D inference succeeded: 1 persons detected
[Datapoint 0] FPS: 8.3
```

### Performance Impact

- **Without SAM3D:** ~50-60 FPS
- **With SAM3D:** ~5-10 FPS (SAM3D adds ~100-200ms per frame)

Total evaluation time: **~30-60 minutes** for 10 DexYCB sequences

---

## Troubleshooting

### SAM3D Import Error

```
ImportError: No module named 'sam_3d_body'
```

**Fix:**
```bash
cd sam-3d-body
pip install -e .
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Checkpoint Not Found

```
FileNotFoundError: ~/checkpoints/sam-3d-body-dinov3/model.ckpt not found
```

**Fix:**
```bash
# Expand tilde manually
ls ~/checkpoints/sam-3d-body-dinov3/
# Or use absolute path
datasets.eval.sam3d_checkpoint_path=/cluster/home/tsmail/checkpoints/sam-3d-body-dinov3/model.ckpt
```

### CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Fix:** SAM3D + MVTracker need ~40GB combined. Solutions:
1. Use A100 80GB GPU
2. Reduce batch size (already 1 for DexYCB)
3. Use CPU for SAM3D (very slow):
   ```python
   sam3d_wrapper = get_sam3d_wrapper(..., device="cpu")
   ```

### No Detections

```
SAM3D inference returned no detections for 20200709-subject-01
```

**Expected:** Some frames have no visible people. Model will fall back to baseline kNN.

**If ALL frames have no detections:**
- Lower `bbox_thr`: `bbox_thr=0.3` (default: 0.5)
- Check images are loaded correctly

### Results Still Identical

After enabling SAM3D, if baseline and SAM3D results are still identical:

**Check logs for:**
```
Running SAM3D inference on ...
SAM3D inference succeeded: X persons detected
```

**If missing:** SAM3D is not running. Verify:
1. `datasets.eval.use_sam3d=true` in config
2. Checkpoint paths are correct
3. No errors during dataset initialization

---

## Comparison with Baseline

After running both configs:

```bash
# Run baseline (no SAM3D)
sbatch scripts/slurm/eval_dexycb_baseline.sh

# Run SAM3D (on-the-fly)
sbatch scripts/slurm/eval_dexycb_sam3d.sh

# Compare results
python scripts/compare_eval_results.py \
  --baseline logs/mvtracker_dexycb_eval_baseline \
  --sam3d logs/mvtracker_dexycb_eval_sam3d \
  --output dexycb_onthefly_comparison.csv
```

**Expected differences:**
- Jaccard scores should improve (especially on hand regions)
- ATE/FDE should decrease (lower error)
- Occlusion accuracy may improve

---

## Advanced: Caching SAM3D Results

To avoid re-running SAM3D on repeated evaluations:

### 1. Save SAM3D Outputs

Add caching to `mvtracker/datasets/sam3d_inference.py`:

```python
# After SAM3D inference succeeds:
cache_path = f"/cluster/scratch/tsmail/sam3d_cache/{seq_name}.pt"
torch.save(joints_3d_world, cache_path)
```

### 2. Load from Cache

```python
# Before running inference:
if os.path.exists(cache_path):
    joints_3d_world = torch.load(cache_path)
else:
    joints_3d_world = self.sam3d_wrapper.process_multiview_sequence(...)
```

---

## Files Modified

1. **`mvtracker/datasets/sam3d_inference.py`** (NEW)
   - SAM3D inference wrapper
   - Multi-view processing
   - Camera-to-world transformation

2. **`mvtracker/datasets/dexycb_multiview_dataset.py`**
   - Added `use_sam3d` flag
   - Added checkpoint path parameters
   - Integrated SAM3D inference in `__getitem__`

3. **`mvtracker/models/evaluation_predictor_3dpt.py`**
   - Pass through `sam3d_joints_world`

4. **`configs/experiment/mvtracker_dexycb_eval_sam3d.yaml`** (UPDATE NEEDED)
   - Add SAM3D parameters

---

## Next Steps

1. ✅ Code integrated
2. ⏳ Download SAM3D checkpoints
3. ⏳ Update experiment configs
4. ⏳ Run evaluation
5. ⏳ Compare results

Good luck! The system should now run SAM3D automatically during evaluation.

---

## Quick Start Commands

```bash
# 1. Setup checkpoints (if needed)
mkdir -p ~/checkpoints/sam-3d-body-dinov3
# [Download checkpoints to this directory]

# 2. Run evaluation with SAM3D
source venv/bin/activate
python -m mvtracker.cli.eval \
  +experiment=mvtracker_dexycb_eval_sam3d \
  datasets.eval.use_sam3d=true \
  datasets.eval.sam3d_checkpoint_path=~/checkpoints/sam-3d-body-dinov3/model.ckpt \
  datasets.eval.sam3d_mhr_path=~/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt

# 3. Check results
ls logs/mvtracker_dexycb_eval_sam3d/eval_*/step--1_metrics_avg.csv
```
