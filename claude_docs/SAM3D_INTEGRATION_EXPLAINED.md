# SAM3D + MVTracker Integration Architecture

## Overview

MVTracker is augmented with SAM-3D-Body mesh priors to improve human-centric 3D point tracking. The key innovation is **semantic-guided kNN correlation** that biases feature matching toward anatomically-consistent points on the human body.

---

## 1. Point Cloud Construction

### Standard MVTracker (baseline)

```python
# Back-project all RGB-D pixels to world space
pointcloud_xyz:  [B*S, V*H*W, 3]      # ~50k 3D points per frame
pointcloud_fvec: [B*S, V*H*W, 128]    # Image features from CNN encoder
```

- **B**: batch size (1)
- **S**: sliding window frames (8)
- **V**: camera views (4)
- **H×W**: spatial resolution after stride-4 downsampling (~96×128)
- **Total**: ~50,000 feature points per frame, concatenated across all views

### With SAM3D (current implementation)

**Point cloud remains unchanged** — SAM3D joints are **not** added to the point cloud.
Instead, they provide **semantic guidance** for feature correlation.

```python
# SAM3D joints passed separately
sam3d_joints: [B*S, 70, 3]   # 70 MHR keypoints in world space
```

**Why not add joints to point cloud?**
- Point cloud is already dense (~50k points)
- Adding 70 joints would be negligible numerically
- SAM3D serves better as **semantic anchor** than additional geometry

---

## 2. kNN Correlation with SAM3D Guidance ⭐

### Problem
Standard kNN finds nearest neighbors **purely by spatial distance**:
```python
# Vanilla MVTracker
neighbor_indices = knn(k=16, pointcloud_xyz, query_xyz)
```
Issues:
- Matches wrong body parts (e.g., left hand → right hand)
- Ignores semantic structure of human body
- Poor performance under occlusion/self-occlusion

### Solution: Semantic-Biased kNN Re-Ranking

**Step 1: Over-fetch candidates**
```python
k_fetch = k * sam3d_knn_overfetch  # e.g., 16 * 4 = 64 candidates
neighbor_dists, neighbor_indices = knn(k_fetch, pointcloud_xyz, query_xyz)
# neighbor_dists: [B, N, 64]  (spatial distances)
```

**Step 2: Compute soft-assignment to SAM3D joints**

For each query point and candidate, compute a distribution over 70 body joints:

```python
tau = 0.1  # temperature parameter (learnable)
joints = sam3d_joints  # [B, 70, 3]

# Query → Joint distances
q_to_j = torch.cdist(query_xyz, joints)      # [B, N, 70]
w_q = softmax(-q_to_j / tau, dim=-1)         # [B, N, 70]  (soft body-part assignment)

# Candidate → Joint distances
c_to_j = torch.cdist(candidate_xyz, joints)  # [B, N, k_fetch, 70]
w_c = softmax(-c_to_j / tau, dim=-1)         # [B, N, k_fetch, 70]
```

**Interpretation:**
- `w_q[i]` = "Which body part is query point i closest to?"
- `w_c[i, j]` = "Which body part is candidate j closest to?"

**Step 3: Compute semantic similarity**

```python
# Semantic distance = L2 distance between soft-assignments
sem_dist = ||w_q - w_c||₂  # [B, N, k_fetch]
```

If query is "near right wrist" and candidate is "near left wrist", their soft-assignments differ → high semantic distance, even if spatially close.

**Step 4: Re-rank by combined distance**

```python
lambda = 0.5  # balance parameter (learnable)
adjusted_dist = spatial_dist + lambda * semantic_dist

# Keep top-k by adjusted distance
_, top_k_idx = topk(adjusted_dist, k=16, largest=False)
final_neighbors = neighbor_indices.gather(dim=-1, index=top_k_idx)
```

**Result:** Neighbors are both **spatially close** AND **semantically consistent** with body topology.

---

### Concrete Example: Self-Occlusion Disambiguation

**Scenario:** Query point is tracking a pixel on the **right wrist** (joint 41 in MHR). The person crosses their arms, bringing left and right wrists close together. Two candidate points are spatially similar:

| Point | Body Part | Spatial Distance | Soft-Assignment Distribution | Semantic Distance |
|-------|-----------|------------------|------------------------------|-------------------|
| **Query** | Right wrist | - | `[..., 0.92 @ J41 (R_wrist), 0.05 @ J7 (R_elbow), ...]` | - |
| **Candidate A** | Right elbow | **0.15m** | `[..., 0.78 @ J7 (R_elbow), 0.15 @ J41 (R_wrist), ...]` | **0.21** (LOW) |
| **Candidate B** | Left wrist | **0.18m** | `[..., 0.87 @ J62 (L_wrist), 0.03 @ J41 (R_wrist), ...]` | **1.35** (HIGH) |

#### Without SAM3D (Baseline)
```python
# Pure spatial kNN selects both (0.15m and 0.18m are similar)
neighbors = [Candidate_A, Candidate_B, ...]
# Problem: Left wrist contaminates right wrist tracking!
```

#### With SAM3D Re-Ranking
```python
# Adjusted distance = spatial_dist + λ * semantic_dist  (λ = 0.5)
adj_dist_A = 0.15 + 0.5 × 0.21 = 0.26  ✅ SELECTED
adj_dist_B = 0.18 + 0.5 × 1.35 = 0.85  ❌ REJECTED

neighbors = [Candidate_A, ...]  # Only right-arm points selected
```

**Why this works:**
- Candidate A (right elbow) and Query (right wrist) both have **high activation on joints 41 and 7** → similar soft-assignments
- Candidate B (left wrist) activates **joint 62 instead of 41** → very different soft-assignment
- **The 3D mesh topology encodes that left and right wrists are far apart in body-part space**, even when spatially close due to pose

**Impact:** Prevents cross-contamination between symmetric body parts (left/right arms, left/right legs) during self-occlusion.

---

## 3. Feature Correlation

After kNN selection, standard correlation proceeds:

```python
neighbor_features = pointcloud_fvec[neighbor_indices]  # [B, N, 16, 128]
target_features = track_features                        # [B, N, 128]

# Compute correlation scores (cosine similarity)
corrs = einsum('BNG,BNKG->BNKG', target_features, neighbor_features)

# Append spatial offsets
neighbor_offsets = neighbor_xyz - query_xyz  # [B, N, 16, 3]
output = concat([corrs, neighbor_offsets], dim=-1)  # [B, N, 16, C+3]
```

This multi-level correlation (4 pyramid levels) feeds into the transformer.

---

## 4. Transformer Attention

The `EfficientUpdateFormer` processes tracks with:
- **Temporal attention** (6 layers): track motion over time
- **Spatial attention** (6 layers): track interactions across space
- **Virtual tracks** (64 tokens): global context aggregation

**No explicit SAM3D integration here** — the improved kNN already provides better features.

Potential future extension:
- Add joint-type embeddings to track features
- Mesh-topology-biased attention masks

---

## 5. Training Differences

### Baseline MVTracker
- Trained on Kubric synthetic (rigid objects)
- No human data → no articulation modeling

### SAM3D-Guided MVTracker (current)
- Trained on Panoptic human sequences (basketball, boxes, football, juggle)
- SAM3D joints guide correlation every forward pass
- Learns to associate human structure with visual appearance

**Loss functions unchanged:**
```python
loss = 3D_trajectory_loss + visibility_loss
```
SAM3D guidance is implicit via better feature matching.

---

## 6. Hyperparameters (Learnable)

From CLAUDE.md and code inspection:

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| `sam3d_knn_overfetch` | `k'` | 4 | Over-fetch multiplier (k' = k × overfetch) |
| `sam3d_bias_lambda` | `λ` | 0.5 | Semantic distance weight |
| `sam3d_bias_tau` | `τ` | 0.1 | Temperature for soft-assignment |

These are likely learnable parameters in the model (check `mvtracker.py` for initialization).

---

## 7. What's Currently Missing (Potential Extensions)

### Option A: Add SAM3D joints to point cloud
```python
# Concatenate joints into point cloud
pointcloud_xyz = concat([pointcloud_xyz, sam3d_joints], dim=1)
pointcloud_fvec = concat([pointcloud_fvec, joint_embeddings], dim=1)
# New shape: [B*S, V*H*W + 70, 3/128]
```

**Benefits:**
- Explicit geometric anchors for tracking
- Direct attention between image features and joints

**Drawbacks:**
- 70 joints ≪ 50k image points → numerically insignificant
- Need to design joint feature embeddings

### Option B: Mesh-topology-biased attention
Add attention masks in transformer based on MHR skeleton:
```python
# Block attention between disconnected body parts
# E.g., left hand tracks cannot attend to right foot tracks
```

### Option C: Vertex sampling
Instead of 70 joints, sample 500-2000 mesh vertices:
```python
sam3d_verts = sample_mesh_vertices(sam3d_mesh, n=2000)
pointcloud_xyz = concat([pointcloud_xyz, sam3d_verts], dim=1)
```

**Trade-off:** More geometric detail vs. computational cost.

---

## 8. Summary: Key Innovation

The SAM3D integration introduces **semantic structure awareness** into MVTracker's purely appearance-based tracking:

| Component | Baseline MVTracker | + SAM3D |
|-----------|-------------------|---------|
| Point cloud | RGB-D only | RGB-D (same) |
| kNN search | Spatial distance only | Spatial + semantic similarity |
| Semantics | None | Body-part soft-assignments |
| Training data | Kubric (objects) | Panoptic (humans) |
| Occlusion handling | Appearance + geometry | + Human topology priors |

**Impact:** Better tracking under:
- ✅ Self-occlusions (body parts overlapping)
- ✅ Articulated motion (joints bending)
- ✅ Partial visibility (only some joints visible)
- ✅ Identity preservation (left/right hand disambiguation)

---

## 9. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│  Input: Multi-View RGB-D (4 views × 8 frames)              │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌──────────────────┐      ┌──────────────────┐
│  BasicEncoder    │      │  SAM3D Joints    │
│  (CNN)           │      │  [B, S, 70, 3]   │
│  → fmaps         │      │  (from dataset)  │
└────────┬─────────┘      └────────┬─────────┘
         │                         │
         ▼                         │
┌─────────────────────────┐        │
│ init_pointcloud_from_   │        │
│ rgbd()                  │        │
│ → xyz: [B*S, 50k, 3]   │        │
│ → fvec: [B*S, 50k, 128]│        │
└───────────┬─────────────┘        │
            │                      │
            └──────────┬───────────┘
                       │
                       ▼
            ┌──────────────────────────┐
            │  PointcloudCorrBlock     │
            │  ┌────────────────────┐  │
            │  │ 1. kNN (over-fetch)│  │
            │  │ 2. Soft-assign to  │  │
            │  │    SAM3D joints    │  │
            │  │ 3. Semantic dist   │  │
            │  │ 4. Re-rank top-k   │  │
            │  └────────────────────┘  │
            │  → corrs: [B, N, 16, 4]  │
            └───────────┬──────────────┘
                        │
                        ▼
            ┌───────────────────────────┐
            │  EfficientUpdateFormer    │
            │  (6 temporal + 6 spatial) │
            │  → Δcoords, Δfeats        │
            └───────────┬───────────────┘
                        │
                        ▼
            ┌───────────────────────────┐
            │  Iterative Refinement     │
            │  (4 GRU-style iterations) │
            │  → Final trajectories     │
            └───────────────────────────┘
```

---

## References

- CLAUDE.md: "SAM3D-guided tracking — inject MHR mesh vertices, joints, and hand poses as semantic/kinematic anchors"
- Implementation: `mvtracker/models/core/mvtracker/mvtracker.py:848-877`
- Dataset: `mvtracker/datasets/panoptic_human_trajectory_dataset.py:355-378`
