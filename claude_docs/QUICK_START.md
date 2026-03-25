# Quick Start: Monitoring Evaluation Jobs

## Current Status

✅ **Jobs submitted successfully** (2026-03-11 11:30 UTC)

| Job | ID | Status |
|-----|-----|--------|
| Baseline | 59906535 | PENDING (Priority) |
| SAM3D | 59906567 | PENDING (Priority) |

**Estimated start time**: Within 30 minutes
**Estimated completion**: 2-4 hours from now

---

## Monitor Progress

### Quick Check
```bash
./scripts/monitor_eval.sh
```

### Detailed Monitoring
```bash
# Check job queue
squeue -u tsmail

# Watch baseline log (Ctrl+C to exit)
tail -f logs/slurm_logs/eval-baseline-59906535.out

# Watch SAM3D log (Ctrl+C to exit)
tail -f logs/slurm_logs/eval-sam3d-59906567.out
```

---

## When Jobs Complete

Both jobs should produce **3 CSV files each** (one per camera configuration).

### 1. Compare Results
```bash
python scripts/compare_eval_results.py \
  --baseline logs/mvtracker_human_eval_baseline \
  --sam3d logs/mvtracker_human_eval_sam3d \
  --output comparison_results.csv
```

### 2. View Results
```bash
# Quick preview
cat comparison_results.csv | column -t -s,

# Or open in Python
python
>>> import pandas as pd
>>> df = pd.read_csv('comparison_results.csv')
>>> print(df[['dataset', 'average_jaccard_improvement_pct', 'occlusion_accuracy_improvement_pct']])
```

---

## Key Files

| Path | Description |
|------|-------------|
| `EVALUATION_STATUS.md` | Full implementation details |
| `QUICK_START.md` | This file |
| `scripts/monitor_eval.sh` | Monitoring script |
| `scripts/compare_eval_results.py` | Results comparison tool |
| `comparison_results.csv` | Final results (generated after jobs finish) |

---

## Success Metrics

**Target improvement with SAM3D**: +3-8%

### Key Metrics
- `average_jaccard` — Overall tracking accuracy
- `occlusion_accuracy` — Visibility prediction (most sensitive to SAM3D)
- `survival` — Long-term tracking robustness

### Interpretation
- **+5% or more**: Strong validation → Thesis objective achieved ✅
- **+2-5%**: Promising → May benefit from hyperparameter tuning
- **0-2%**: Marginal → Try Option B (fine-tune) or Option D (sweep)
- **Negative**: Debug needed → Check coordinate frames & visualization

---

## Troubleshooting

### Jobs not starting?
```bash
# Check queue position
squeue -u tsmail -o "%.10i %.12j %.8T %.10M %.9P %Q"

# Check partition status
sinfo -p gpupr.4h

# Jobs will start automatically when GPUs become available
```

### Jobs failed?
```bash
# Check error messages
tail -50 logs/slurm_logs/eval-baseline-59906535.out
tail -50 logs/slurm_logs/eval-sam3d-59906567.out

# Common issues:
# - OOM → Reduce sam3d_knn_overfetch to 2
# - Module load failed → Check venv at ./venv/bin/activate
# - Dataset not found → Verify human_tracks.npz exists
```

---

## What's Different from Original Plan?

**Original**: Full retraining (50k steps, 42-56 hours, 25M parameters)
**Current**: Zero-shot evaluation (pretrained checkpoint, 2-4 hours, 2 new parameters)

**Why this works**:
- SAM3D only adds 2 learnable scalars (`lambda=0.1`, `tau=0.15`)
- Pretrained checkpoint contains all core tracking weights
- Default parameter values are reasonable for initial validation
- Checkpoint loader handles missing parameters with `strict=False`

**Time saved**: ~40-52 hours ⚡

---

## Next Steps

After results are ready:
1. Review `comparison_results.csv`
2. Report findings in thesis
3. If promising → Consider full training or hyperparameter optimization
4. If not → Debug & visualize with Rerun

---

**Questions?** Check `EVALUATION_STATUS.md` for full details.
