# Code Changes Summary - MVTracker SAM3D Evaluation

**Date**: 2026-03-11

## Files Modified

### 1. Dataset Configuration
**Files**: 
- `configs/experiment/mvtracker_human_eval_baseline.yaml`
- `configs/experiment/mvtracker_human_eval_sam3d.yaml`

**Changes**:
```yaml
# Before
datasets:
  root: ./datasets
  eval:
    num_workers: 4
    max_seq_len: 1000

# After
datasets:
  root: /cluster/scratch/tsmail/datasets
  eval:
    num_workers: 0
    max_seq_len: 300
```

### 2. SLURM Scripts
**Files**:
- `scripts/slurm/eval_baseline.sh`
- `scripts/slurm/eval_sam3d.sh`

**Changes**:
```bash
# Added memory allocation
#SBATCH --mem-per-cpu=32G
```

### 3. Dataset Loader - Fix Einsum Bug
**File**: `mvtracker/datasets/panoptic_human_trajectory_dataset.py`

**Line 377**:
```python
# Before (WRONG)
joints_3d_world_trans = torch.einsum('ij,TPKJ->TPKI', rot.float(), joints_3d_world_trans)

# After (CORRECT)
joints_3d_world_trans = torch.einsum('ij,TPKj->TPKi', rot.float(), joints_3d_world_trans)
```

### 4. Datapoint Class - Add SAM3D Field
**File**: `mvtracker/datasets/utils.py`

**Added to Datapoint class**:
```python
sam3d_joints_world: Optional[torch.Tensor] = None  # B, S, n_persons, 70, 3
```

**Added to collate_fn**:
```python
sam3d_joints_world = (
    torch.stack([b.sam3d_joints_world for b, _ in batch], dim=0)
    if batch[0][0].sam3d_joints_world is not None
    else None
)

# Added to Datapoint constructor in collate_fn
Datapoint(
    ...
    sam3d_joints_world=sam3d_joints_world
)
```

### 5. Evaluator - Fix Dataset Name Recognition
**File**: `mvtracker/evaluation/evaluator_3dpt.py`

**Line 534**:
```python
# Before
elif "panoptic-multiview" in dataset_name:
    evaluation_setting = "panoptic-multiview"

# After
elif "panoptic-multiview" in dataset_name or "panoptic-human" in dataset_name:
    evaluation_setting = "panoptic-multiview"
```

### 6. Evaluator - Pass SAM3D Joints to Model (CRITICAL FIX)
**File**: `mvtracker/evaluation/evaluator_3dpt.py`

**Line 273 - Extract from datapoint**:
```python
sam3d_joints_world = (datapoint.sam3d_joints_world.clone().float().to(device)
                      if datapoint.sam3d_joints_world is not None else None)
```

**Line 479 - Pass to model**:
```python
fwd_kwargs = {
    "rgbs": rgbs,
    "depths": depths,
    "image_features": image_features,
    "query_points_3d": query_points_3d,
    "intrs": intrs,
    "extrs": extrs,
    "sam3d_joints_world": sam3d_joints_world,  # ADDED THIS LINE
    ...
}
```

### 7. Training Script - Fix Checkpoint Loading
**File**: `mvtracker/cli/train.py`

**Line 660**:
```python
# Before
else:
    fabric.load_raw(restore_ckpt_path, model)

# After
else:
    # Load pre-trained weights (not a training checkpoint)
    try:
        fabric.load_raw(restore_ckpt_path, model, strict=True)
    except RuntimeError as e:
        logging.warning(f"Failed to load weights from {restore_ckpt_path} with strict=True: {e}. "
                        f"Trying again with strict=False.")
        fabric.load_raw(restore_ckpt_path, model, strict=False)
```

### 8. Comparison Script - Fix CSV Pattern
**File**: `scripts/compare_eval_results.py`

**Line 20**:
```python
# Before
csv_files = glob.glob(f"{log_dir}/eval_*/step-0_metrics.csv")

# After
csv_files = glob.glob(f"{log_dir}/eval_*/step-*_metrics_avg.csv")
```

---

## Git Status

Modified files:
```
M configs/experiment/mvtracker_human_eval_baseline.yaml
M configs/experiment/mvtracker_human_eval_sam3d.yaml
M scripts/slurm/eval_baseline.sh
M scripts/slurm/eval_sam3d.sh
M mvtracker/datasets/panoptic_human_trajectory_dataset.py
M mvtracker/datasets/utils.py
M mvtracker/evaluation/evaluator_3dpt.py
M mvtracker/cli/train.py
M scripts/compare_eval_results.py
```

---

## Summary

The most critical fix was **#6** - without passing `sam3d_joints_world` to the model, the SAM3D code path was never executed, resulting in identical baseline and SAM3D results.

All other fixes were necessary to get the evaluation pipeline working at all:
- Fixes #1-2: Made jobs run without crashing
- Fixes #3-5: Fixed bugs preventing evaluation from starting
- Fix #7: Allowed loading pretrained checkpoint without SAM3D parameters

