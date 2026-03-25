# Quick Start: SAM3D On-the-Fly Evaluation

## TL;DR - Run This

```bash
# 1. Make sure you have SAM3D checkpoints
ls ~/checkpoints/sam-3d-body-dinov3/model.ckpt  # Should exist
ls ~/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt  # Should exist

# 2. Activate environment
source venv/bin/activate

# 3. Run evaluation with SAM3D
python -m mvtracker.cli.eval \
  +experiment=mvtracker_dexycb_eval_sam3d \
  datasets.eval.use_sam3d=true \
  datasets.eval.sam3d_checkpoint_path=~/checkpoints/sam-3d-body-dinov3/model.ckpt \
  datasets.eval.sam3d_mhr_path=~/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt

# 4. Results will be in:
#    logs/mvtracker_dexycb_eval_sam3d/eval_*/step--1_metrics_avg.csv
```

## Status Check

What's been done:
- ✅ SAM3D inference wrapper created
- ✅ DexYCB dataset updated to run SAM3D
- ✅ Evaluation pipeline passes SAM3D data through
- ✅ Model uses SAM3D for kNN guidance

What you need to do:
- ⏳ Get SAM3D checkpoints (see below)
- ⏳ Run evaluation
- ⏳ Compare with baseline

## Get SAM3D Checkpoints

If you don't have them:

```bash
# Create checkpoint directory
mkdir -p ~/checkpoints/sam-3d-body-dinov3
cd ~/checkpoints/sam-3d-body-dinov3

# Option 1: Download from Hugging Face
huggingface-cli download facebook/sam-3d-body --local-dir .

# Option 2: Manual download
# Go to: https://huggingface.co/facebook/sam-3d-body/tree/main
# Download: model.ckpt, assets/mhr_model.pt, model_config.yaml
```

## Verify It Works

```bash
# Test SAM3D import
python -c "from sam_3d_body import load_sam_3d_body; print('✓ SAM3D OK')"

# Test dataset with SAM3D
python -c "
from mvtracker.datasets.dexycb_multiview_dataset import DexYCBMultiViewDataset
ds = DexYCBMultiViewDataset.from_name(
    'dex-ycb-multiview-views0123-cached',
    '/cluster/scratch/tsmail/datasets',
    use_sam3d=True,
    sam3d_checkpoint_path='~/checkpoints/sam-3d-body-dinov3/model.ckpt',
    sam3d_mhr_path='~/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt',
)
print('✓ Dataset with SAM3D OK')
"
```

## Run Comparison

```bash
# Baseline (no SAM3D)
python -m mvtracker.cli.eval +experiment=mvtracker_dexycb_eval_baseline

# SAM3D (on-the-fly)
python -m mvtracker.cli.eval \
  +experiment=mvtracker_dexycb_eval_sam3d \
  datasets.eval.use_sam3d=true \
  datasets.eval.sam3d_checkpoint_path=~/checkpoints/sam-3d-body-dinov3/model.ckpt \
  datasets.eval.sam3d_mhr_path=~/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt

# Compare
cat logs/mvtracker_dexycb_eval_baseline/eval_dex-ycb-multiview-views0123-cached/step--1_metrics_avg.csv
cat logs/mvtracker_dexycb_eval_sam3d/eval_dex-ycb-multiview-views0123-cached/step--1_metrics_avg.csv
```

## Expected Timeline

- SAM3D model loading: ~10 seconds (one-time)
- Per sequence: ~2-5 minutes
- Total (10 sequences): ~30-60 minutes

## Troubleshooting

**Problem:** `ImportError: No module named 'sam_3d_body'`

```bash
cd sam-3d-body
pip install -e .
```

**Problem:** Results still identical

Check logs for:
```
Running SAM3D inference on 20200709-subject-01...
SAM3D inference succeeded: 1 persons detected
```

If missing, SAM3D is not running. Double-check `use_sam3d=true` is set.

**Problem:** CUDA OOM

SAM3D + MVTracker need ~40GB. Use A100 80GB:
```bash
#SBATCH --gres=gpumem:80g
```

## Key Files

- **Implementation:** `mvtracker/datasets/sam3d_inference.py`
- **Dataset:** `mvtracker/datasets/dexycb_multiview_dataset.py`
- **Docs:** `SAM3D_ONTHEFLY_SETUP.md` (detailed guide)

## What Changed from Before

**Before:** Results identical because `sam3d_joints_world=None`

**Now:** SAM3D runs during evaluation:
1. Dataset loads RGB frames
2. SAM3D infers body joints (per view, per frame)
3. Joints transformed to world coords
4. MVTracker uses them for semantic kNN
5. Different predictions → different metrics

## Success Criteria

After running both configs, you should see:
- **Logs show SAM3D inference running**
- **Metrics differ between baseline and SAM3D**
- **SAM3D has better Jaccard/lower ATE on hand regions**

Good luck! 🚀
