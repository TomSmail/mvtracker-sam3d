# MVTracker Baseline vs SAM3D Evaluation Status

## Overview

Evaluating MVTracker with and without SAM3D-guided kNN on human-centric benchmarks using **pretrained checkpoint** (zero-shot transfer, no retraining required).

**Time saved**: ~40-52 hours compared to full retraining approach

---

## Jobs Submitted

| Job ID | Name | Status | Config | Runtime |
|--------|------|--------|--------|---------|
| 59906535 | eval-baseline | PENDING | `mvtracker_human_eval_baseline.yaml` | ~1-2 hours |
| 59906567 | eval-sam3d | PENDING | `mvtracker_human_eval_sam3d.yaml` | ~1-2 hours |

**Check status**: `squeue -u tsmail`

---

## Key Configuration Differences

### Baseline (`use_sam3d_knn_bias: false`)
- Pure spatial 3D kNN (k=16 neighbors)
- No semantic body topology information
- Standard MVTracker correlation block

### SAM3D (`use_sam3d_knn_bias: true`)
- Over-fetch 64 candidates (4x overfetch)
- Semantic re-ranking based on MHR joint assignments
- Soft-weighted distance to 70 body joints
- Learnable parameters: `sam3d_bias_lambda=0.1`, `sam3d_bias_tau=0.15` (initialized to defaults)

---

## Datasets Being Evaluated

All evaluations run on **Panoptic Studio human trajectory benchmarks**:

1. `panoptic-human-views1_7_14_20` — Cameras 1, 7, 14, 20 (wide baseline)
2. `panoptic-human-views27_16_14_8` — Cameras 27, 16, 14, 8 (alternate config)
3. `panoptic-human-views1_4_7_11` — Cameras 1, 4, 7, 11 (dense config)

**Sequences**: softball, tennis (dynamic human motion with occlusions)

**Ground truth**: 384 tracked points per sequence with SAM3D-annotated visibility

---

## Expected Results Location

### Baseline
- **Logs**: `logs/mvtracker_human_eval_baseline/`
- **Metrics**: `logs/mvtracker_human_eval_baseline/eval_*/step-0_metrics.csv`
- **SLURM output**: `logs/slurm_logs/eval-baseline-59906535.out`

### SAM3D
- **Logs**: `logs/mvtracker_human_eval_sam3d/`
- **Metrics**: `logs/mvtracker_human_eval_sam3d/eval_*/step-0_metrics.csv`
- **SLURM output**: `logs/slurm_logs/eval-sam3d-59906567.out`

---

## Monitoring Jobs

```bash
# Check job status
squeue -u tsmail

# View live logs
tail -f logs/slurm_logs/eval-baseline-59906535.out
tail -f logs/slurm_logs/eval-sam3d-59906567.out

# Check if jobs completed
ls logs/mvtracker_human_eval_baseline/eval_*/step-0_metrics.csv
ls logs/mvtracker_human_eval_sam3d/eval_*/step-0_metrics.csv
```

---

## Comparing Results (After Completion)

Once both jobs finish, run the comparison script:

```bash
python scripts/compare_eval_results.py \
  --baseline logs/mvtracker_human_eval_baseline \
  --sam3d logs/mvtracker_human_eval_sam3d \
  --output comparison_results.csv
```

**Key metrics to compare**:
- `average_jaccard` — Combined position + occlusion accuracy (target: +3-8%)
- `occlusion_accuracy` — Visibility prediction correctness (most sensitive to SAM3D)
- `survival` — % tracks within 1m throughout sequence
- `average_pts_within_thresh` — Average across 5/10/20/40cm thresholds

---

## Success Criteria

### ✅ Strong Validation (3-8% improvement)
- SAM3D provides clear benefit for human tracking
- Body topology guidance helps under occlusion
- **Next step**: Consider full end-to-end training with SAM3D enabled

### ⚠️ Marginal Benefit (0-2% improvement)
- Hyperparameters may need tuning (lambda, tau)
- **Next step**: Run hyperparameter sweep (Option D) or fine-tune SAM3D parameters (Option B)

### 🐛 No improvement or regression
- Debug coordinate frame alignment
- Visualize SAM3D joint quality with Rerun
- Check dataset loading and transformation pipeline

---

## Files Created

### Config Files
- `configs/experiment/mvtracker_human_eval_baseline.yaml`
- `configs/experiment/mvtracker_human_eval_sam3d.yaml`

### SLURM Scripts
- `scripts/slurm/eval_baseline.sh`
- `scripts/slurm/eval_sam3d.sh`

### Analysis Tools
- `scripts/compare_eval_results.py`

### Checkpoints
- `checkpoints/mvtracker_200000_june2025.pth` (87 MB, pretrained on Kubric)

---

## Timeline

| Step | Task | Duration | Status |
|------|------|----------|--------|
| 1 | Cancel old training job | 5 min | ✅ Complete (59708917 canceled) |
| 2 | Download checkpoint | 5 min | ✅ Complete |
| 3-5 | Create configs | 15 min | ✅ Complete |
| 6 | Submit jobs | 5 min | ✅ Complete |
| 7 | Wait for jobs | 1-2 hrs | 🔄 In progress |
| 8 | Compare results | 30 min | ⏳ Pending |
| **Total** | **2-4 hours** | **vs 42-56 hours** | **40-52 hours saved** |

---

## Next Steps After Results

### If SAM3D shows improvement:
1. Report findings in thesis
2. Consider full training with SAM3D enabled (optional)
3. Test on additional datasets (DexYCB, EgoExo4D)

### If results are marginal:
1. Hyperparameter sweep for optimal lambda/tau
2. Fine-tune 2 SAM3D scalars (Option B, ~2 hours)
3. Ablation: test with different overfetch factors

### If results are negative:
1. Debug SAM3D joint coordinate frames
2. Visualize with Rerun: `python demo.py --rerun save`
3. Check dataset loading pipeline

---

## Important Notes

- **Zero-shot evaluation**: Pretrained checkpoint never saw human data during training
- **Missing parameters**: SAM3D scalars initialized to defaults (lambda=0.1, tau=0.15)
- **Checkpoint loading**: Uses `strict=False` to handle missing parameters gracefully
- **Scene normalization**: Panoptic datasets use manual similarity transforms (already handled in dataloader)
- **SAM3D joints**: 70 MHR joints per frame, transformed to world space via `extrs`

---

## Troubleshooting

### Jobs don't start
- Check queue: `squeue -u tsmail`
- Check partition limits: `sinfo -p gpupr.4h`
- Adjust time limit if needed

### Out of memory
- Reduce `sam3d_knn_overfetch` from 4 to 2
- Reduce `datasets.eval.max_seq_len` from 1000 to 500

### Coordinate frame issues
- Verify SAM3D joints: `ls /cluster/scratch/tsmail/datasets/panoptic-multiview/*/human_tracks.npz`
- Check transformation in `panoptic_human_trajectory_dataset.py:399`

---

**Last updated**: 2026-03-11 11:30 UTC
**Estimated completion**: 2026-03-11 13:30 UTC (2 hours from submission)
