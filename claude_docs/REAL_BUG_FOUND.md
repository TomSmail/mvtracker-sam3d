# THE REAL BUG - cam_int Type Mismatch

**Date:** March 17, 2026, 11:35 AM
**Status:** Fix applied, evaluation running

---

## Summary

After multiple attempts, found the **actual root cause**: SAM3D expects `cam_int` parameter as a **torch.Tensor**, but we were passing a **numpy.ndarray**.

---

## The Discovery

### Previous Results (All numpy/tensor fixes)
- Job 60567063-60567064 completed
- Still had **1960 numpy/tensor errors**
- Results almost IDENTICAL to baseline (only 3/109 metrics changed)
- Conclusion: SAM3D was failing on EVERY frame

### The Smoking Gun

**File:** `sam-3d-body/sam_3d_body/data/utils/prepare_batch.py`, **Line 64**:

```python
if cam_int is not None:
    batch["cam_int"] = cam_int.to(batch["img"])  # ← Calls .to() on cam_int!
```

**Our code** was doing:
```python
intr_np = intr_frame.cpu().numpy()  # Convert to numpy ❌
outputs = self.estimator.process_one_image(
    rgb_np,
    cam_int=intr_np,  # Pass numpy array ❌
    ...
)
```

**The issue:** `.to()` is a torch.Tensor method, doesn't exist on numpy.ndarray!

---

## Why This Was Confusing

**SAM3D's function signature** says:
```python
def process_one_image(
    self,
    img: Union[str, np.ndarray],
    ...
    cam_int: Optional[np.ndarray] = None,  # ← Says np.ndarray!
    ...
)
```

**But internally** it calls `.to()` on `cam_int`, which only works on tensors. This is a **bug in SAM3D's type hints** - the actual implementation expects a tensor.

---

## The Fix

**File:** `mvtracker/datasets/sam3d_inference.py`, **Lines 149-152**:

```python
# OLD (broken):
intr_np = intr_frame.cpu().numpy()
outputs = self.estimator.process_one_image(rgb_np, cam_int=intr_np, ...)

# NEW (fixed):
intr_tensor = intr_frame  # Keep as tensor
outputs = self.estimator.process_one_image(rgb_np, cam_int=intr_tensor, ...)
```

**Key change:** Don't convert intrinsics to numpy - pass the tensor directly.

---

## All Fixes Applied

### ✅ Fix #1: DINOv3 PyTorch 2.3 Compatibility
**Status:** Applied and working (SAM3D loads successfully)

### ✅ Fix #2: SAM3D Tensor/NumPy Handling
**Status:** Applied but not the root cause (still useful for robustness)

### ✅ Fix #3: cam_int Must Be Tensor (THE REAL FIX)
**Status:** Just applied, evaluation running

---

## Current Jobs

| Type | Job ID | Status | Expected Completion |
|------|--------|--------|---------------------|
| Baseline | 60567547 | Running | ~12:05 PM |
| SAM3D | 60567550 | Pending/Running | ~12:05 PM |

**Monitor:** `bash scripts/monitor_both_evals.sh 60567547 60567550`

---

## Expected Results

With the REAL fix:
1. ✅ **Zero** `'numpy.ndarray' object has no attribute 'to'` errors
2. ✅ SAM3D runs successfully on ALL frames
3. ✅ Results significantly DIFFERENT from baseline
4. ✅ Slower FPS due to SAM3D overhead
5. ✅ Parameter count +2 (SAM3D bias parameters)

---

## Timeline of Bug Discovery

| Time | Event | Status |
|------|-------|--------|
| 09:00 AM | Initial bug found (SAM3D = baseline) | Found |
| 10:30 AM | DINOv3 compatibility fix | ✅ Fixed |
| 10:54 AM | Tensor/numpy handling fix | Partial |
| 11:00-11:24 AM | Multiple resubmissions | Bytecode cache issues |
| 11:30 AM | All fixes applied, still 1960 errors | 🤔 |
| **11:35 AM** | **REAL bug found: cam_int type mismatch** | **✅ Fixed** |
| 11:36 AM | Final resubmission | ⏳ Running |

---

## Why This Was So Hard to Find

1. **Misleading type hints:** SAM3D signature says `np.ndarray` but implementation needs tensor
2. **Silent exception handling:** Error caught and logged as WARNING, no stack trace
3. **Bytecode caching:** Even with source fixes, Python kept using old compiled code
4. **Multiple issues:** DINOv3 + tensor handling + cam_int type - had to fix all three

---

## Verification

Once jobs complete (ETA 12:05 PM):

```bash
# Should be 0 (or very few) errors
grep -c "numpy.ndarray.*has no attribute.*to" logs/slurm_logs/eval-dexycb-sam3d-60567550.out

# Should show successful SAM3D inference
grep "SAM3D inference succeeded" logs/slurm_logs/eval-dexycb-sam3d-60567550.out | wc -l

# Should show significant differences
python3 /tmp/compare_results.py
```

---

## Files Modified

1. `checkpoints/.cache/hub/.../ms_deform_attn.py` (DINOv3 fix)
2. `mvtracker/datasets/sam3d_inference.py` (Tensor handling + cam_int fix)

**Key line:** Line 151: `intr_tensor = intr_frame  # Keep as tensor, don't convert to numpy`

---

## Lessons Learned

1. **Don't trust type hints blindly** - Check actual implementation
2. **Look for .to() calls** - They only work on tensors
3. **Check third-party code** - The bug was in SAM3D's expectations, not our code
4. **Clear ALL bytecode** - Python caching can hide source changes
5. **Test incrementally** - Should have tested SAM3D inference in isolation first

---

**Status:** Waiting for evaluation to complete (~30 minutes)

**Next:** Compare results to verify SAM3D is actually working with real impact on metrics.
