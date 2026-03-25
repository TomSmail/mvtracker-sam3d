# SAM3D Evaluation Bug Fix

## Problem
SAM3D and baseline evaluations were producing **identical results** on DexYCB dataset.

## Root Cause
SAM3D inference was **failing silently** due to DINOv3 PyTorch compatibility issue:

```
ERROR - SAM3D inference failed: cannot import name 'custom_fwd' from 'torch.amp'
```

When SAM3D failed, it returned `None`, causing MVTracker to fall back to baseline behavior.

## Solution
Patched DINOv3 cached code at:
```
checkpoints/.cache/hub/facebookresearch_dinov3_main/dinov3/eval/segmentation/models/utils/ms_deform_attn.py
```

### Changes:
1. Updated import to use `torch.cuda.amp` (PyTorch 2.3+):
   ```python
   from torch.cuda.amp import custom_fwd, custom_bwd
   ```

2. Removed deprecated `device_type` parameter from decorators:
   ```python
   @custom_fwd(cast_inputs=torch.float32)  # removed device_type="cuda"
   @custom_bwd  # removed device_type="cuda"
   ```

## Verification
```bash
source venv/bin/activate
TORCH_HOME=./checkpoints/.cache python -c "
import torch
model = torch.hub.load('facebookresearch/dinov3', 'dinov3_vith16plus', 
                       source='github', pretrained=False, drop_path=0.1)
print(f'✓ Loaded {type(model).__name__}, embed_dim={model.embed_dim}')
"
```

## Next Steps
1. Re-run DexYCB SAM3D evaluation: `sbatch scripts/slurm/eval_dexycb_sam3d.sh`
2. Run Panoptic multiview evaluation to check compatibility
3. Compare results to verify SAM3D is now active
