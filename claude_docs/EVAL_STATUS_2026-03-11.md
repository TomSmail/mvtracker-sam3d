# MVTracker Baseline vs SAM3D Evaluation - Session Status
**Date**: 2026-03-11
**Status**: SAM3D evaluation running (Job 60004952)

---

## ✅ COMPLETED

### 1. Baseline Evaluation (SUCCESSFUL)
- **Job ID**: 60001367
- **Status**: COMPLETED (Exit code 0)
- **Runtime**: 3:09
- **Output**: All 3 datasets evaluated successfully
- **Results Location**: `logs/mvtracker_human_eval_baseline/`

**Baseline Metrics Summary:**
| Dataset | Avg Jaccard | Occlusion Acc | Survival | Avg Pts Within Thresh |
|---------|-------------|---------------|----------|-----------------------|
| views1_7_14_20 | 56.48% | 91.50% | 97.66% | 66.82% |
| views27_16_14_8 | 56.66% | 91.18% | 97.42% | 67.07% |
| views1_4_7_11 | 54.57% | 93.86% | 97.79% | 64.21% |

---

## 🔄 IN PROGRESS

### 2. SAM3D Evaluation (RESUBMITTED WITH FIX)
- **Job ID**: 60004952 (current)
- **Status**: PENDING in queue
- **Config**: `configs/experiment/mvtracker_human_eval_sam3d.yaml`
- **Expected Runtime**: ~3 minutes
- **Output Location**: `logs/mvtracker_human_eval_sam3d/`

---

## 🐛 ISSUES FIXED (Chronological)

### Issue 1: Wrong Dataset Path
- **Error**: `FileNotFoundError: './datasets/panoptic-multiview'`
- **Fix**: Changed `datasets.root` from `./datasets` to `/cluster/scratch/tsmail/datasets`
- **Files**: Both eval config files

### Issue 2: Out of Memory
- **Error**: Jobs killed with `OUT_OF_ME+` status
- **Root Cause**: SLURM scripts only allocated 1GB RAM (default)
- **Fix**:
  - Added `#SBATCH --mem-per-cpu=32G` to SLURM scripts
  - Reduced `num_workers: 4 → 0` (disable multiprocessing)
  - Reduced `max_seq_len: 1000 → 300`
- **Files**:
  - `scripts/slurm/eval_baseline.sh`
  - `scripts/slurm/eval_sam3d.sh`
  - Both eval config files

### Issue 3: Einsum Bug
- **Error**: `RuntimeError: einsum(): output subscript I does not appear in the equation`
- **Root Cause**: Incorrect uppercase/lowercase indices in einsum equation
- **Fix**: Changed `'ij,TPKJ->TPKI'` to `'ij,TPKj->TPKi'`
- **File**: `mvtracker/datasets/panoptic_human_trajectory_dataset.py:377`

### Issue 4: Missing Datapoint Field
- **Error**: `TypeError: Datapoint.__init__() got an unexpected keyword argument 'sam3d_joints_world'`
- **Fix**:
  - Added `sam3d_joints_world: Optional[torch.Tensor] = None` to Datapoint dataclass
  - Updated `collate_fn` to handle the new field
- **File**: `mvtracker/datasets/utils.py`

### Issue 5: Evaluator Dataset Name Not Recognized
- **Error**: `NotImplementedError` at evaluator line 543
- **Root Cause**: Evaluator only checked for `"panoptic-multiview"`, not `"panoptic-human"`
- **Fix**: Added `or "panoptic-human" in dataset_name` to the check
- **File**: `mvtracker/evaluation/evaluator_3dpt.py:534`

### Issue 6: Checkpoint Missing SAM3D Parameters
- **Error**: `RuntimeError: Missing key(s) in state_dict: "sam3d_bias_lambda", "sam3d_bias_tau"`
- **Root Cause**: `fabric.load_raw()` doesn't support `strict=False` parameter
- **Fix**: Wrapped `fabric.load_raw()` with try/except to fall back to `strict=False`
- **File**: `mvtracker/cli/train.py:660`

### Issue 7: SAM3D Joints Not Passed to Model (CRITICAL)
- **Error**: Baseline and SAM3D had **identical results** (SAM3D code path never executed)
- **Root Cause**: Evaluator never extracted `sam3d_joints_world` from datapoint or passed it to model
- **Fix**:
  1. Added extraction: `sam3d_joints_world = datapoint.sam3d_joints_world...`
  2. Added to `fwd_kwargs`: `"sam3d_joints_world": sam3d_joints_world`
- **File**: `mvtracker/evaluation/evaluator_3dpt.py:273,479`

---

## 📝 NEXT STEPS (For Tomorrow)

1. **Check SAM3D job status** (Job 60004952):
   ```bash
   myjobs
   # or
   sacct -j 60004952 --format=JobID,State,ExitCode,Elapsed
   ```

2. **If successful, check SAM3D outputs**:
   ```bash
   ls logs/mvtracker_human_eval_sam3d/eval_*/step--1_metrics_avg.csv
   ```

3. **Extract SAM3D metrics**:
   ```bash
   grep -E "(average_jaccard|occlusion_accuracy|survival|average_pts_within_thresh)" \
     logs/mvtracker_human_eval_sam3d/eval_panoptic-human-views1_7_14_20/step--1_metrics_avg.csv
   grep -E "(average_jaccard|occlusion_accuracy|survival|average_pts_within_thresh)" \
     logs/mvtracker_human_eval_sam3d/eval_panoptic-human-views27_16_14_8/step--1_metrics_avg.csv
   grep -E "(average_jaccard|occlusion_accuracy|survival|average_pts_within_thresh)" \
     logs/mvtracker_human_eval_sam3d/eval_panoptic-human-views1_4_7_11/step--1_metrics_avg.csv
   ```

4. **Compare results manually** (until comparison script is fully fixed)

5. **If SAM3D results differ from baseline**:
   - Document the improvements/changes
   - Analyze which metrics benefited most
   - Consider hyperparameter tuning of `sam3d_bias_lambda` and `sam3d_bias_tau`

6. **If SAM3D results are still identical**:
   - Add debug logging to verify SAM3D joints are non-None
   - Check if `use_sam3d_bias` flag is properly propagated
   - Verify SAM3D joints have reasonable values

---

## 🗂️ KEY FILES

### Config Files
- `configs/experiment/mvtracker_human_eval_baseline.yaml` — Baseline config (use_sam3d_knn_bias: false)
- `configs/experiment/mvtracker_human_eval_sam3d.yaml` — SAM3D config (use_sam3d_knn_bias: true)

### SLURM Scripts
- `scripts/slurm/eval_baseline.sh` — Baseline evaluation script
- `scripts/slurm/eval_sam3d.sh` — SAM3D evaluation script

### Results
- `logs/mvtracker_human_eval_baseline/` — Baseline metrics (COMPLETE)
- `logs/mvtracker_human_eval_sam3d/` — SAM3D metrics (IN PROGRESS)

### Modified Source Files
1. `mvtracker/datasets/panoptic_human_trajectory_dataset.py` — Fixed einsum bug
2. `mvtracker/datasets/utils.py` — Added sam3d_joints_world to Datapoint
3. `mvtracker/evaluation/evaluator_3dpt.py` — Fixed dataset name check + **added SAM3D joints passing**
4. `mvtracker/cli/train.py` — Added strict=False fallback for checkpoint loading
5. `scripts/compare_eval_results.py` — Fixed CSV filename pattern (not fully working yet)

---

## 🔬 TECHNICAL DETAILS

### SAM3D Integration
- **Model Flag**: `use_sam3d_knn_bias: true` enables SAM3D-guided kNN
- **Parameters**:
  - `sam3d_bias_lambda` (initialized to 0.1) — blending weight for semantic bias
  - `sam3d_bias_tau` (initialized to 0.15) — temperature for softmax
- **Data**: SAM3D joints are in `datapoint.sam3d_joints_world` (B, T, n_persons, 70, 3)
- **Algorithm**: Over-fetch 4× kNN candidates, then re-rank by semantic proximity to 70 MHR body joints

### Previous Runs (All Failed/Incomplete)
- Jobs 59906535, 59906567 — Wrong dataset path
- Jobs 59994590, 59994591 — OOM (no RAM allocation)
- Jobs 59997801, 59997803 — OOM (still insufficient RAM)
- Jobs 59999747, 59999748 — Einsum bug
- Jobs 59999974, 59999975 — Missing Datapoint field
- Jobs 60001367 (✅ baseline success), 60001371 — SAM3D checkpoint loading error
- Job 60004188 — SAM3D completed but joints not passed (identical results)
- **Job 60004952 — CURRENT (SAM3D with joints properly passed)**

---

## 📊 EXPECTED OUTCOME

If SAM3D is working correctly, you should see:
- **Occlusion accuracy** may improve (SAM3D helps disambiguate under occlusion)
- **Average Jaccard** may improve 3-8% (combined position + occlusion metric)
- **Survival** may stay similar or improve slightly
- Results should **NOT** be identical to baseline

---

## 🚀 RESUMING WORK

To resume tomorrow:
```bash
cd /cluster/home/tsmail/mvtracker-sam3d
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6
source venv/bin/activate

# Check job status
myjobs

# If complete, check results
tail -100 logs/slurm_logs/eval-sam3d-60004952.out
ls logs/mvtracker_human_eval_sam3d/eval_*/

# Extract and compare metrics (see "NEXT STEPS" section above)
```

---

**Last Updated**: 2026-03-11 23:30 UTC
**Current Job**: 60004952 (SAM3D evaluation with proper joint passing)
