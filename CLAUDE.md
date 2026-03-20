# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Thesis Context

This project extends **MVTracker** (Multi-View 3D Point Tracking, ICCV 2025) with **SAM-3D-Body** human mesh priors for human-centric scenes. The three objectives are:

1. **SAM3D-guided tracking** — inject MHR mesh vertices, joints, and hand poses as semantic/kinematic anchors into MVTracker's 3D feature point cloud and transformer
2. **Human-centric data engine** — AnthroTAP-inspired pipeline using SAM3D fits to generate dense, occlusion-aware 3D trajectories on multi-view datasets (Panoptic Studio, DexYCB, EgoExo4D)
3. **Unified model** — train and evaluate a model robust to occlusions and articulated human motion

---

## Key Files for Modifications

| Concern | File |
|---|---|
| Core tracking model | `mvtracker/models/core/mvtracker/mvtracker.py` |
| Point cloud construction | `mvtracker/models/core/mvtracker/model_utils.py` |
| Transformer (EfficientUpdateFormer) | `mvtracker/models/core/cotracker2/blocks.py` |
| Training loop & losses | `mvtracker/cli/train.py` |
| Inference wrapper (sliding window) | `mvtracker/models/evaluation_predictor_3dpt.py` |
| Panoptic Studio dataset | `mvtracker/datasets/panoptic_multiview_dataset.py` |
| DexYCB dataset | `mvtracker/datasets/dexycb_multiview_dataset.py` |
| Generic scene loader | `mvtracker/datasets/generic_scene_dataset.py` |
| SAM3D model | `sam-3d-body/sam_3d_body/models/meta_arch/sam3d_body.py` |
| MHR mesh definition | `sam-3d-body/sam_3d_body/metadata/mhr70.py` |

---

## MVTracker Architecture

### Forward Pass (`mvtracker.py`)

**Inputs / Outputs:**
```
rgbs:             (B, V, T, 3, H, W)
depths:           (B, V, T, 1, H, W)
intrs:            (B, V, T, 3, 3)
extrs:            (B, V, T, 3, 4)   # [R|t], camera-to-world
query_points_3d:  (B, N, 4)         # [frame_idx, x, y, z] world space
→ traj_e:         (B, T, N, 3)      # predicted world-space trajectories
→ vis_e:          (B, T, N)         # visibility logits
```

**Per sliding-window iteration (`forward_iteration()`):**
1. `BasicEncoder` → per-frame features `(B*V*S, 128, H//4, W//4)`
2. `init_pointcloud_from_rgbd()` → merged multi-view point cloud `(B*S, V·H·W, 3)` xyz + `(B*S, V·H·W, 128)` features, repeated for 4 pyramid levels
3. `PointcloudCorrBlock.corr_sample()` → kNN (k=16) in 3D, 3-plane correlation → per-track features `(B*S, N, 16, 4)`
4. `EfficientUpdateFormer` (6 temporal + 6 spatial attention layers, 64 virtual tracks) → `delta (B*N, S, 130)` split into `d_coord (3)` + `d_feats (128)`
5. Accumulate `coords += d_coord`, `ffeats += d_feats` over `iters=4` refinement steps

### Point Cloud Construction (`model_utils.py: init_pointcloud_from_rgbd`)

Back-projects all view depth maps to world space and concatenates:
```
fmaps:  (B, V, S, 128, H, W)
depths: (B, V, S, 1, H, W)
→ pointcloud_xyz:  (B*S, V*H*W, 3)
→ pointcloud_fvec: (B*S, V*H*W, 128)
```
**Integration point:** concatenate SAM3D mesh vertices and their projected image features here before passing to `PointcloudCorrBlock`.

### Transformer (`blocks.py: EfficientUpdateFormer`)

Input per track: concatenation of 3D flow embedding (195-d), multi-level correlation features, track features (128-d), visibility mask — total ~320-d. Virtual tracks (64 learnable tokens) attend over all real tracks via cross-attention every few layers. **Integration point:** add mesh-topology-biased attention or extra semantic token embeddings (joint type, body part) here.

### Training (`train.py`)

```python
# Loss
xyz_loss  = sequence_loss_3d(pred_coords, gt_coords, vis_gt, valid, gamma=0.8)
vis_loss  = balanced_ce_loss(vis_preds, vis_gt, valid)
total     = xyz_loss + vis_loss * visibility_loss_weight
```
Losses computed per sliding window, then summed. **Integration point:** add mesh regularization / FK consistency / feature alignment losses here.

---

## SAM-3D-Body Architecture

SAM3D (`sam-3d-body/`) runs per-frame, per-view.

**Outputs from `forward_decoder()`:**
```
pred_keypoints_3d:  (B, 70, 3)   # MHR joints in camera frame
pred_keypoints_2d:  (B, 70, 2)   # 2D joint projections
pred_vertices:      (B, V_mesh, 3) # Full MHR mesh vertices
pred_pose:          (B, 144)      # Pose parameters
pred_shape:         (B, 10)       # Body shape parameters
pred_cam:           (B, 3)        # Camera translation
```
MHR has 70 joints (body + hands + feet). SAM3D outputs are in camera/crop frame — must transform to world frame via `extrs` before injecting into MVTracker.

---

## Dataset Batch Structure

All datasets return a `Datapoint` with:
```
video:          (B, V, T, 3, H, W)
videodepth:     (B, V, T, 1, H, W)
intrs:          (B, V, T, 3, 3)
extrs:          (B, V, T, 3, 4)
trajectory_3d:  (B, T, N, 3)     # GT world-space tracks
trajectory:     (B, V, T, N, 3)  # GT 2D tracks + depth per view
visibility:     (B, V, T, N)
query_points_3d:(B, N, 4)        # [frame_idx, x, y, z]
valid:          (B, T, N)
```
Typical: N=384 tracks, V=2–4 views, T=50–150 frames. Panoptic Studio uses manual scene normalization (similarity transform encoded in the dataloader). DexYCB has hand-object contact — relevant for hand trajectory supervision.

---

## Common Commands

```bash
# Overfitting test (fits on 24GB GPU)
python -m mvtracker.cli.train +experiment=mvtracker_overfit_mini

# Full training (80GB GPU)
python -m mvtracker.cli.train +experiment=mvtracker_overfit

# Evaluate MVTracker on MV-Kubric
python -m mvtracker.cli.eval \
  experiment_path=logs/mvtracker model=mvtracker \
  datasets.eval.names=[kubric-multiview-v3-views0123] \
  restore_ckpt_path=checkpoints/mvtracker_200000_june2025.pth

# Evaluate on Panoptic / DexYCB (see scripts/slurm/eval.sh for all variants)
python -m mvtracker.cli.eval \
  experiment_path=logs/mvtracker model=mvtracker \
  datasets.eval.names=[panoptic-multiview-views1_7_14_20-cached]

# Demo (saves mvtracker_demo.rrd, open at app.rerun.io)
python demo.py --rerun save --lightweight
```

Dataset name modifiers: `-views`, `-duster`, `-cached`, `-novelviews`, `-removehand`, `-2dpt`. Use `-cached` to reproduce paper numbers exactly.

---

## Environment

Python 3.10.12 · PyTorch 2.3.0 · CUDA 12.1 · gcc 11.3.0

```bash
conda create -n 3dpt python=3.10.12 -y && conda activate 3dpt
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install -r requirements.full.txt

# Recommended GPU speedups
pip install --upgrade --no-build-isolation flash-attn==2.5.8
pip install "git+https://github.com/ethz-vlg/pointcept.git@2082918#subdirectory=libs/pointops"
# pointops requires gcc 11.3.0: conda install -c conda-forge gcc_linux-64=11.3.0 gxx_linux-64=11.3.0

# SAM-3D-Body: see sam-3d-body/INSTALL.md
```

Config system: **Hydra** (`configs/`), training runner: **Lightning Fabric**. kNN search has a pure-PyTorch fallback (`torch.cdist + topk`) when `pointops` is unavailable.

### Running Python on the Euler Cluster

The project uses a **venv** at `venv/` (not conda). To run Python scripts:

```bash
# Interactive shell — activate the venv first
source venv/bin/activate
python -m mvtracker.cli.train +experiment=mvtracker_overfit_mini

# Or invoke directly without activating
./venv/bin/python -m mvtracker.cli.train +experiment=mvtracker_overfit_mini
```

**SLURM jobs** (see `scripts/slurm/` for examples) must load modules before activating:

```bash
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6
source $DIR/venv/bin/activate
```

**Key details:**
- System python is `/usr/bin/python3` — this does NOT have project dependencies
- The venv python is `venv/bin/python` (Python 3.11.6, PyTorch 2.3.1+cu121)
- `conda` is **not installed** on this cluster; all deps are in the venv
- GPU jobs require SLURM (`sbatch scripts/slurm/<script>.sh`); login nodes have no GPUs
- Use `--gres=gpumem:80g` for full training, `--gres=gpumem:24g` for overfit tests

---

## Important Notes

- **Scene normalization is critical.** Panoptic and DexYCB use manual similarity transforms baked into their dataloaders. The generic loader auto-normalizes assuming XY-plane ground. SAM3D world-frame outputs must respect the same normalization convention.
- **Coordinate frames:** SAM3D outputs joints/vertices in camera (or crop) frame. Convert to world frame via `extrs_inv` before injecting into MVTracker's point cloud or as query points.
- **MVTracker is point-count-agnostic** — augmenting `query_points_3d` with SAM3D joints (70 MHR joints) or mesh vertices requires no architectural changes to downstream correlation or transformer code; just increase N.
- **Point cloud size scales with views:** at default resolution, `V·H·W ≈ 57k` points per frame. Adding 6890 SMPL vertices is ~12% overhead; adding sampled mesh surface points (e.g., 512–2048) is negligible.
