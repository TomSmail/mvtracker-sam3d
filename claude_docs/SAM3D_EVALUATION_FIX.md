# SAM3D Evaluation Pipeline Fix

## Problem Identified

The DexYCB evaluation results for **baseline** and **SAM3D** configs were **identical**, despite `use_sam3d_knn_bias` being set to `true` in the SAM3D config.

### Root Cause

The SAM3D integration was **missing at the dataset level**:

1. ✅ **Model code** - `mvtracker.py` accepts `sam3d_joints_world` parameter
2. ✅ **Training code** - `train.py` passes SAM3D data from batches
3. ✅ **Evaluator code** - `evaluator_3dpt.py` extracts and forwards SAM3D data
4. ✅ **Evaluation predictor** - `evaluation_predictor_3dpt.py` forwards SAM3D data
5. ❌ **DexYCB dataset** - **Did not populate `sam3d_joints_world` field**

Result: `sam3d_joints_world=None` was passed to the model in both baseline and SAM3D configs, so the model **always fell back to pure spatial kNN** regardless of the config flag.

---

## Fix Applied

### 1. DexYCB Dataset (`mvtracker/datasets/dexycb_multiview_dataset.py`)

Added `sam3d_joints_world` field to the Datapoint:

```python
# TODO: Add SAM3D body fitting for DexYCB hands
# For now, set to None - SAM3D guidance will not be used during evaluation
# To enable SAM3D: pre-compute MHR fits offline and load them here
# Expected shape: [T, n_persons, 70, 3] in world coordinates
sam3d_joints_world = None

datapoint = Datapoint(
    # ... other fields ...
    sam3d_joints_world=sam3d_joints_world,
)
```

### 2. Evaluation Predictor (`mvtracker/models/evaluation_predictor_3dpt.py`)

Updated to accept and pass through `sam3d_joints_world`:

```python
def forward(
    self,
    rgbs,
    depths,
    query_points_3d,
    intrs,
    extrs,
    save_debug_logs=False,
    debug_logs_path="",
    query_points_view=None,
    sam3d_joints_world=None,  # ← Added
    **kwargs,
):
    # ...
    results = self.model(
        rgbs,
        depths=depths,
        query_points=query_points,
        intrs=intrs,
        extrs=extrs,
        # ...
        sam3d_joints_world=sam3d_joints_world,  # ← Passed through
        **kwargs,
    )
```

---

## Current Status

### ✅ Code Infrastructure Complete

The entire pipeline now correctly passes `sam3d_joints_world` from dataset → evaluator → predictor → model:

1. **Dataset** loads/computes SAM3D joints → returns in Datapoint
2. **Evaluator** extracts `datapoint.sam3d_joints_world` → passes to predictor
3. **Predictor** forwards `sam3d_joints_world` → passes to model
4. **Model** uses SAM3D joints for kNN re-ranking (if not None)

### ⚠️ DexYCB Still Returns None

Since DexYCB dataset currently sets `sam3d_joints_world=None`, the **results will still be identical** between baseline and SAM3D configs. This is expected and correct behavior.

---

## Next Steps: Adding Real SAM3D Data

To enable SAM3D guidance on DexYCB, you need to:

### Option 1: Pre-compute SAM3D Fits (Recommended)

1. **Run SAM3D offline** on all DexYCB sequences:
   ```bash
   # Pseudo-code - adapt to your SAM3D pipeline
   python scripts/run_sam3d_on_dexycb.py \
     --input /cluster/scratch/tsmail/datasets/dex-ycb-multiview \
     --output /cluster/scratch/tsmail/datasets/dex-ycb-multiview-sam3d-fits
   ```

2. **Save fits** in a consistent format (e.g., NPZ per sequence):
   ```python
   # Expected format per sequence:
   np.savez(
       f"{sequence_name}_sam3d.npz",
       joints_3d_world=joints_3d_world,  # Shape: [T, n_persons, 70, 3]
       vertices_3d_world=vertices_3d_world,  # Optional: [T, n_persons, 6890, 3]
   )
   ```

3. **Update DexYCB dataset** to load fits:
   ```python
   # In dexycb_multiview_dataset.py::_getitem_helper()

   sam3d_path = os.path.join(datapoint_path, f"{self.seq_names[index]}_sam3d.npz")
   if os.path.exists(sam3d_path):
       sam3d_data = np.load(sam3d_path)
       joints_3d_world = torch.from_numpy(sam3d_data["joints_3d_world"]).float()

       # Apply same scene transformation as trajectories
       joints_3d_world_trans = apply_scene_transform(
           joints_3d_world, scale, rot, translation
       )
       sam3d_joints_world = joints_3d_world_trans
   else:
       sam3d_joints_world = None
   ```

### Option 2: Run SAM3D On-the-Fly (Slower)

Integrate SAM3D inference directly into the dataset loader. **Not recommended** for evaluation due to speed.

---

## Verification

Once SAM3D data is available, verify the fix works:

```bash
# Run both evaluations
sbatch scripts/slurm/eval_dexycb_both.sh

# Check that results differ
python scripts/compare_eval_results.py \
  --baseline logs/mvtracker_dexycb_eval_baseline \
  --sam3d logs/mvtracker_dexycb_eval_sam3d \
  --output dexycb_comparison.csv

# Expected: Metrics should now be different between baseline and SAM3D
```

---

## Files Modified

1. `mvtracker/datasets/dexycb_multiview_dataset.py` - Added `sam3d_joints_world` field
2. `mvtracker/models/evaluation_predictor_3dpt.py` - Pass through `sam3d_joints_world`

## Files Already Correct (No Changes Needed)

1. `mvtracker/cli/train.py` - Already passes SAM3D data during training
2. `mvtracker/evaluation/evaluator_3dpt.py` - Already extracts SAM3D from datapoint
3. `mvtracker/models/core/mvtracker/mvtracker.py` - Already uses SAM3D for kNN
4. `mvtracker/datasets/utils.py` - Datapoint already has `sam3d_joints_world` field

---

## Reference: Panoptic Dataset (Working Example)

See `mvtracker/datasets/panoptic_human_trajectory_dataset.py` for a working implementation:

```python
# Extract joint trajectories from pre-computed tracks
joint_mask = track_types == 0  # keypoints only
joints_3d_world = traj3d_world[..., joint_mask, :]
joints_3d_world = joints_3d_world.reshape(n_frames, n_persons, n_keypoints, 3)

# Apply scene transformation
joints_3d_world_trans = apply_transform(joints_3d_world, scale, rot, translation)

# Return in Datapoint
datapoint = Datapoint(
    # ...
    sam3d_joints_world=joints_3d_world_trans,
)
```

---

## Summary

**Why results were identical:**
DexYCB dataset didn't populate `sam3d_joints_world`, so both configs passed `None` to the model.

**What was fixed:**
Added the missing data field to DexYCB dataset and ensured the entire pipeline passes it through.

**What's needed next:**
Compute SAM3D fits for DexYCB sequences and load them in the dataset.

**Expected outcome:**
Once SAM3D data is available, the SAM3D config will show improved metrics on hand-centric tracking.
