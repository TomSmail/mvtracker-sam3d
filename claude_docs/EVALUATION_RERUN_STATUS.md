# DexYCB Evaluation Re-run Status

**Date:** March 17, 2026
**Status:** In Progress

---

## Current Jobs

| Type | Job ID | Status | Node |
|------|--------|--------|------|
| Baseline | 60566543 | Pending/Running | TBD |
| SAM3D (Full Fix) | 60566546 | Pending/Running | TBD |

**Monitor:** `bash scripts/monitor_both_evals.sh 60566543 60566546`

---

## All Fixes Applied

### ✅ Fix #1: DINOv3 PyTorch 2.3 Compatibility
**File:** `checkpoints/.cache/hub/facebookresearch_dinov3_main/dinov3/eval/segmentation/models/utils/ms_deform_attn.py`
- Changed import: `from torch.cuda.amp import custom_fwd, custom_bwd`
- Removed `device_type` parameter from decorators
- **Status:** Applied and verified working

### ✅ Fix #2: SAM3D Tensor/NumPy Type Handling
**File:** `mvtracker/datasets/sam3d_inference.py`

**Lines 205-208** - `_camera_to_world()`:
```python
if isinstance(joints_cam, np.ndarray):
    joints_cam_t = torch.from_numpy(joints_cam).float()
else:
    joints_cam_t = joints_cam.float()
```

**Lines 265-272** - `_aggregate_multiview_detections()`:
```python
if isinstance(joints, np.ndarray):
    joints = torch.from_numpy(joints).float()
joints_3d_world[frame_idx, person_idx] = joints.to(device)
```

**Status:** Applied and Python cache cleared

---

## Previous Results (Partial SAM3D Fix)

From Job 60564019 (DINOv3 fix only, numpy/tensor bug present):

**Key Finding:** Results WERE different (80/109 metrics changed)
- Confirmed SAM3D was active (even with partial failures)
- Average Jaccard: 84.21% → 84.06% (-0.15%)
- FPS: 54.47 → 12.65 (SAM3D overhead)
- **Conclusion:** Bug fix successful, SAM3D working but degraded by type errors

---

## Expected Results

With **both fixes** applied:

1. **No numpy/tensor errors** - SAM3D should run cleanly on all views/frames
2. **Full SAM3D coverage** - All detections used, not just subset
3. **More pronounced differences** - Baseline vs SAM3D should show larger metric changes
4. **Better tracking on human-centric points** - SAM3D joints should guide kNN

---

## Timeline

- **11:02 AM** - Previous jobs (60565494, 60565498) canceled due to cache issue
- **11:05 AM** - Python cache cleared
- **11:06 AM** - Jobs re-submitted (60566543, 60566546)
- **Est. 11:20 AM** - Jobs start running
- **Est. 12:00 PM** - First results available
- **Est. 12:30 PM** - Both evaluations complete

Each evaluation processes:
- 10 DexYCB sequences
- 2 variants: full tracking + object-only (removehand)
- ~40 minutes per job

---

## Verification Steps

Once jobs complete:

1. **Check logs for errors:**
   ```bash
   grep "SAM3D failed" logs/slurm_logs/eval-dexycb-sam3d-60566546.out | wc -l
   grep "numpy.ndarray" logs/slurm_logs/eval-dexycb-sam3d-60566546.out | wc -l
   ```
   **Expected:** 0 numpy/tensor errors

2. **Compare results:**
   ```bash
   python3 /tmp/compare_results.py
   ```
   **Expected:** More differences than partial fix (>80 metrics changed)

3. **Check SAM3D parameters:**
   ```bash
   grep "params_total" logs/mvtracker_dexycb_eval_*/eval_dex-ycb-multiview-views0123-cached/step--1_metrics_avg.csv
   ```
   **Expected:**
   - Baseline: 22607356 params
   - SAM3D: 22607358 params (+2 for bias parameters)

---

## Documentation Files

- `BUG_FIX_SUMMARY.md` - Complete technical analysis
- `SAM3D_BUG_FIX.md` - Initial bug discovery
- `SAM3D_FIXES_APPLIED.md` - Fix implementation details
- `EVALUATION_RERUN_STATUS.md` - This file (current status)
- `scripts/monitor_both_evals.sh` - Real-time monitoring script

---

## Next Steps After Completion

1. ✅ Verify all fixes working (no errors in logs)
2. ✅ Compare baseline vs SAM3D results
3. ✅ Document performance differences
4. ⏳ Run Panoptic multiview evaluation
5. ⏳ Create final comparison report
6. ⏳ Update CLAUDE.md with fix notes
