# MVTracker + SAM3D Data Pipeline Status

**Last Updated**: 2026-03-09 15:24 UTC

## Overview

Running AnthroTAP-inspired data generation pipeline on 6 Panoptic Studio sequences:
- basketball, boxes, football, juggle, softball, tennis

**Timeline**:
1. ✅ Merged `data-pipeline` branch into `sam3d-insert` (complete)
2. 🔄 SAM3D inference on all sequences (in progress)
3. ⏳ Dense track generation from mesh vertices (pending)
4. ⏳ Training MVTracker + SAM3D features (pending)

---

## Current Status: SAM3D Inference

**SLURM Job ID**: 59597794 (array job, 6 tasks)

### Inference Progress

| Sequence   | Views | Frames | Status      |
|------------|-------|--------|-------------|
| basketball | 1/31  | 17     | 🔄 Running  |
| boxes      | 1/31  | 21     | 🔄 Running  |
| football   | 0/31  | 0      | ⏳ Queued   |
| juggle     | 0/31  | 0      | ⏳ Queued   |
| softball   | 0/31  | 0      | ⏳ Queued   |
| tennis     | 0/31  | 0      | ⏳ Queued   |

**Note**: GPU quota limits to 2 concurrent jobs. Remaining sequences will start automatically when current jobs finish.

**Expected completion**: ~2-4 hours per sequence (depending on people density)

---

## Monitoring Commands

```bash
# Quick status check
bash scripts/data_engine/monitor_pipeline.sh

# Check SLURM queue
squeue -u tsmail

# Follow live progress (basketball)
tail -f logs/slurm_logs/sam3d-inference-59597794_0.out

# Check inference outputs
ls /cluster/scratch/tsmail/datasets/panoptic-multiview/basketball/sam3d_predictions/
```

---

## Next Steps

1. **Wait for SAM3D inference to complete** (~12-24 hours for all 6 sequences)
2. **Run track generation**:
   ```bash
   bash scripts/data_engine/submit_slurm_generate_all.sh
   ```
3. **Visualize results**:
   ```bash
   python scripts/data_engine/visualize_tracks.py \
     --data_root /cluster/scratch/tsmail/datasets/panoptic-multiview \
     --seq basketball --view 1 --output tracks_basketball_v1.mp4
   ```
4. **Train mixed model**:
   ```bash
   python -m mvtracker.cli.train +experiment=mvtracker_human
   ```

---

## Data Specifications

### Per-Sequence Stats
- **Views**: ~31 cameras per sequence
- **Frames**: ~150 frames per sequence
- **Duration**: ~5 seconds @ 30 fps
- **Resolution**: 640×360 per view
- **Total frames to process**: 6 × 150 × 31 = 27,900 frames

### SAM3D Outputs (per frame, per view)
```
sam3d_predictions/view_XX/frame_XXXXX.npz:
  - n_persons: int
  - person_i_pred_vertices: (6890, 3) MHR mesh in camera frame
  - person_i_pred_keypoints_3d: (70, 3) MHR joints
  - person_i_pred_cam_t: (3,) camera translation
  - person_i_bbox: (4,) detection bbox
```

### Generated Tracks
```
human_tracks.npz:
  - vertices_3d: (N_people, T, V, 3) sampled mesh vertices in world frame
  - vertices_valid: (N_people, T, V) visibility mask
  - joints_3d: (N_people, T, 70, 3) MHR joints in world frame
  - joints_valid: (N_people, T, 70) joint visibility
```

---

## Thesis Validation Plan

**Objective 1**: SAM3D-guided tracking ✅ (implemented on `sam3d-insert`)
**Objective 2**: Human-centric data engine 🔄 (running now)
**Objective 3**: Unified model training ⏳ (next)

### Training Strategy
- **Mixed training**: 80% Kubric + 20% Panoptic human tracks
- **Sequences**: 4 train (basketball, boxes, football, juggle) + 2 test (softball, tennis)
- **Augmentation**: temporal crops, view dropout, photometric
- **Evaluation metrics**: AJ on human joints, occlusion survival, per-body-part breakdown

### Expected Timeline
- SAM3D inference: 12-24 hours (automated)
- Track generation: 1-2 hours (automated)
- Training: 2-3 days on 80GB GPU
- Evaluation & visualization: 1 day
