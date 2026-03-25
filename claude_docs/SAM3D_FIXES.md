# SAM3D Evaluation Fixes

## Summary

Fixed two critical issues preventing SAM3D from working properly:

1. **Missing SAM3D bias parameters** - checkpoint didn't contain trained weights
2. **Excessive GPU memory requests** - causing long queue times

## Changes Made

### 1. GPU Memory Optimization

**Before:** All eval scripts requested 80GB GPU memory
**After:** Reduced to 16GB (actual usage: ~1GB)

**Files modified:**
- `scripts/slurm/eval_baseline.sh`
- `scripts/slurm/eval_sam3d.sh`
- `scripts/slurm/eval_dexycb_baseline.sh`
- `scripts/slurm/eval_dexycb_sam3d.sh`
- `scripts/slurm/eval_dexycb_both.sh`

**Impact:**
- Jobs should now queue ~5x faster (more GPUs available with 16GB)
- Still plenty of headroom (16x over actual usage)

### 2. SAM3D Bias Parameter Initialization

**Problem:** The checkpoint `mvtracker_200000_june2025.pth` was trained without SAM3D, so it lacks the learnable parameters:
- `sam3d_bias_lambda` (controls semantic re-ranking strength)
- `sam3d_bias_tau` (controls joint-proximity soft-assignment temperature)

Without these, SAM3D features were being computed but **not used** in the kNN search.

**Solution:**

1. Added `ensure_sam3d_parameters_initialized()` method to `MVTracker` class in `mvtracker/models/core/mvtracker/mvtracker.py`:
   - Encapsulates initialization logic within the model itself
   - Returns `True` if parameters were initialized, `False` if already present

2. Updated `mvtracker/cli/train.py` to call this method after checkpoint loading:
   - Works for both training and evaluation (eval.py is just a wrapper)
   - Handles Fabric-wrapped models correctly

**Default values chosen:**
- `sam3d_bias_lambda = 0.1` — modest semantic re-ranking weight (geometric distance still dominant)
- `sam3d_bias_tau = 0.15` — temperature for soft joint assignments (reasonable spread)

These match the original initialization in `mvtracker.py:134-135`.

## Next Steps

### Re-run SAM3D Evaluation

Now that the bias parameters will be initialized, re-run the evaluation:

```bash
# Single SAM3D run
sbatch scripts/slurm/eval_dexycb_sam3d.sh

# Or run both baseline + SAM3D + comparison
sbatch scripts/slurm/eval_dexycb_both.sh
```

**Expected changes:**
- You should now see: `"Initializing SAM3D bias parameters (missing from checkpoint)..."`
- SAM3D should actually affect the results (previously identical to baseline)
- Jobs should queue much faster with 16GB requirement

### For Better Results: Train with SAM3D

The initialized bias parameters use hand-picked defaults. For optimal performance, train a checkpoint with SAM3D enabled:

```bash
# Edit config to enable SAM3D during training
# In configs/experiment/mvtracker_overfit.yaml:
model:
  use_sam3d_knn_bias: true
  sam3d_knn_overfetch: 4

datasets:
  train:
    use_sam3d: true
    sam3d_checkpoint_path: /path/to/sam3d/model.ckpt
    sam3d_mhr_path: /path/to/mhr_model.pt

# Then train
python -m mvtracker.cli.train +experiment=mvtracker_overfit
```

This will learn the optimal `sam3d_bias_lambda` and `sam3d_bias_tau` through backpropagation.

## Verification

Check the new evaluation log for:
```
Initializing SAM3D bias parameters (missing from checkpoint)...
  sam3d_bias_lambda = 0.100
  sam3d_bias_tau = 0.150
```

If you see this, SAM3D is now properly enabled!

---

**Date:** 2026-03-16
**Issue:** Job 60327393 showed identical results between baseline and SAM3D
**Root cause:** Missing trainable parameters in checkpoint
