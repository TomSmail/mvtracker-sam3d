"""
Generate dense human 3D trajectories from SAM3D predictions on Panoptic Studio.

Pipeline:
  1. Load per-view, per-frame SAM3D predictions (keypoints, vertices, camera params)
  2. Lift per-view detections to world space using known extrinsics
  3. Associate persons across views via pelvis proximity
  4. Average mesh vertices across views for each person
  5. Apply temporal smoothing
  6. Sample track points (keypoints + vertex subset)
  7. Compute per-view visibility (projection + bounds + depth occlusion)
  8. Save as human_tracks.npz in MVTracker-compatible format

Usage:
    python scripts/data_engine/generate_human_tracks.py \
        --data_root /cluster/scratch/tsmail/datasets/panoptic-multiview

    # Single sequence:
    python scripts/data_engine/generate_human_tracks.py \
        --data_root ... --seq_index 0
"""

import argparse
import os
import sys
import warnings

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import uniform_filter1d


def get_sequences(data_root):
    """Return sorted list of valid sequence directories."""
    seqs = [
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
        and not d.startswith(".")
        and not d.startswith("_")
    ]
    return sorted(seqs)


def load_annotations(seq_path):
    """Load camera calibration from tapvid3d_annotations.npz."""
    ann_file = os.path.join(seq_path, "tapvid3d_annotations.npz")
    ann = np.load(ann_file)
    return {
        "extrinsics": ann["extrinsics"],      # [n_views, 4, 4] or [n_views, 3, 4]
        "intrinsics": ann["intrinsics"],      # [n_views, 3, 3]
    }


def load_sam3d_predictions(pred_dir, n_views, n_frames):
    """
    Load all SAM3D predictions for a sequence.

    Returns:
        predictions: dict mapping (view_idx, frame_idx) -> list of person dicts
    """
    predictions = {}
    for view_idx in range(n_views):
        view_dir = os.path.join(pred_dir, f"view_{view_idx:02d}")
        if not os.path.isdir(view_dir):
            continue
        for frame_idx in range(n_frames):
            pred_file = os.path.join(view_dir, f"frame_{frame_idx:05d}.npz")
            if not os.path.exists(pred_file):
                predictions[(view_idx, frame_idx)] = []
                continue
            data = np.load(pred_file)
            n_persons = int(data["n_persons"])
            persons = []
            for p in range(n_persons):
                prefix = f"person_{p}"
                person = {
                    "pred_keypoints_3d": data[f"{prefix}_pred_keypoints_3d"],   # [70, 3] camera-space
                    "pred_vertices": data[f"{prefix}_pred_vertices"],           # [V, 3] camera-space
                    "pred_cam_t": data[f"{prefix}_pred_cam_t"],                 # [3]
                    "bbox": data[f"{prefix}_bbox"],                             # [4]
                }
                if f"{prefix}_pred_keypoints_2d" in data:
                    person["pred_keypoints_2d"] = data[f"{prefix}_pred_keypoints_2d"]
                persons.append(person)
            predictions[(view_idx, frame_idx)] = persons
    return predictions


def cam_to_world(points_cam, extrinsic):
    """
    Transform points from camera space to world space.

    Args:
        points_cam: [N, 3] points in camera coordinates
        extrinsic: [3, 4] or [4, 4] world-to-camera transform (R|t)

    Returns:
        points_world: [N, 3] points in world coordinates
    """
    if extrinsic.shape[0] == 3:
        R = extrinsic[:3, :3]
        t = extrinsic[:3, 3]
    else:
        R = extrinsic[:3, :3]
        t = extrinsic[:3, 3]

    # extrinsic maps world->camera: p_cam = R @ p_world + t
    # inverse: p_world = R^T @ (p_cam - t)
    R_inv = R.T
    t_inv = -R_inv @ t
    points_world = (R_inv @ points_cam.T).T + t_inv
    return points_world


def get_pelvis_world(person, extrinsic):
    """Get world-space pelvis position (mean of left_hip=9, right_hip=10)."""
    kp3d = person["pred_keypoints_3d"]  # body-centered
    cam_t = person["pred_cam_t"]        # translation to camera space
    pelvis_body = kp3d[[9, 10], :].mean(axis=0)
    pelvis_cam = pelvis_body + cam_t
    pelvis_world = cam_to_world(pelvis_cam[None], extrinsic)[0]
    return pelvis_world


def associate_persons_across_views(predictions, extrinsics, n_views, frame_idx,
                                   max_distance=1.0):
    """
    Associate person detections across views using pelvis proximity.

    Returns:
        groups: list of dicts, each mapping view_idx -> person_idx
    """
    view_persons = {}
    for v in range(n_views):
        persons = predictions.get((v, frame_idx), [])
        pelvis_positions = []
        for p in persons:
            pelvis = get_pelvis_world(p, extrinsics[v])
            pelvis_positions.append(pelvis)
        if pelvis_positions:
            view_persons[v] = {
                "persons": persons,
                "pelvis": np.stack(pelvis_positions),
            }

    if not view_persons:
        return []

    # Use the view with most detections as reference
    ref_view = max(view_persons.keys(), key=lambda v: len(view_persons[v]["persons"]))
    n_ref = len(view_persons[ref_view]["persons"])

    groups = [{ref_view: i} for i in range(n_ref)]
    ref_pelvis = view_persons[ref_view]["pelvis"]

    for v in sorted(view_persons.keys()):
        if v == ref_view:
            continue
        other_pelvis = view_persons[v]["pelvis"]
        n_other = len(other_pelvis)

        cost = np.linalg.norm(
            ref_pelvis[:, None, :] - other_pelvis[None, :, :], axis=-1
        )  # [n_ref, n_other]

        row_ind, col_ind = linear_sum_assignment(cost)
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < max_distance:
                groups[r][v] = c

    # Sort groups by number of supporting views (most views first) so the
    # best-supported person is always at index 0. This prevents spurious
    # detections from a single view from displacing the real person.
    groups.sort(key=lambda g: len(g), reverse=True)

    return groups


def fuse_meshes_multiview(predictions, extrinsics, groups, frame_idx):
    """
    For each person group, fuse mesh predictions across views.

    Returns:
        list of dicts with fused world-space keypoints and vertices.
    """
    fused = []
    for group in groups:
        all_kp_world = []
        all_verts_world = []

        for view_idx, person_idx in group.items():
            person = predictions[(view_idx, frame_idx)][person_idx]
            extr = extrinsics[view_idx]
            cam_t = person["pred_cam_t"]               # [3] body-centered -> camera

            kp_body = person["pred_keypoints_3d"]      # [70, 3] body-centered
            verts_body = person["pred_vertices"]        # [V, 3] body-centered

            kp_cam = kp_body + cam_t[None, :]
            verts_cam = verts_body + cam_t[None, :]

            kp_world = cam_to_world(kp_cam, extr)
            verts_world = cam_to_world(verts_cam, extr)

            all_kp_world.append(kp_world)
            all_verts_world.append(verts_world)

        fused_kp = np.mean(all_kp_world, axis=0)           # [70, 3]
        fused_verts = np.mean(all_verts_world, axis=0)      # [V, 3]

        fused.append({
            "keypoints_3d": fused_kp,
            "vertices": fused_verts,
            "n_views_fused": len(group),
        })

    return fused


def temporal_smooth(trajectories, kernel_size=3):
    """
    Apply temporal smoothing to trajectories.

    Args:
        trajectories: [T, N, 3]
        kernel_size: smoothing window

    Returns:
        smoothed: [T, N, 3]
    """
    if kernel_size <= 1:
        return trajectories
    smoothed = np.copy(trajectories)
    for dim in range(3):
        smoothed[:, :, dim] = uniform_filter1d(
            trajectories[:, :, dim], size=kernel_size, axis=0, mode="nearest"
        )
    return smoothed


def sample_vertices(vertices, n_samples=500, seed=42):
    """
    Uniformly subsample mesh vertices.

    Args:
        vertices: [V, 3]
        n_samples: number of vertices to sample

    Returns:
        indices: [n_samples] selected vertex indices
    """
    rng = np.random.RandomState(seed)
    n_verts = vertices.shape[0]
    if n_verts <= n_samples:
        return np.arange(n_verts)
    return rng.choice(n_verts, size=n_samples, replace=False)


def compute_visibility(traj3d_world, extrinsics, intrinsics, img_h, img_w,
                       depth_maps=None, depth_tolerance=0.1):
    """
    Compute per-view visibility for 3D trajectory points.

    Args:
        traj3d_world: [T, N, 3] world-space 3D trajectories
        extrinsics: [n_views, 3, 4] or [n_views, 4, 4] world-to-cam
        intrinsics: [n_views, 3, 3]
        img_h, img_w: image dimensions
        depth_maps: optional [n_views, T, H, W] depth maps for occlusion checking
        depth_tolerance: relative depth tolerance for occlusion

    Returns:
        visibility: [n_views, T, N] boolean
        traj2d: [n_views, T, N, 2] pixel coordinates
        traj_z: [n_views, T, N] camera-space z
    """
    n_views = extrinsics.shape[0]
    T, N, _ = traj3d_world.shape

    points_homo = np.concatenate([
        traj3d_world,
        np.ones((*traj3d_world.shape[:-1], 1))
    ], axis=-1)  # [T, N, 4]

    visibility = np.zeros((n_views, T, N), dtype=bool)
    traj2d = np.zeros((n_views, T, N, 2), dtype=np.float32)
    traj_z = np.zeros((n_views, T, N), dtype=np.float32)

    for v in range(n_views):
        extr = extrinsics[v]
        if extr.shape[0] == 4:
            extr = extr[:3, :]
        intr = intrinsics[v]

        # Project to camera space: [T, N, 3]
        points_cam = np.einsum("ij,TNj->TNi", extr, points_homo)
        z_cam = points_cam[:, :, 2]

        # Project to pixel space
        points_2d_homo = np.einsum("ij,TNj->TNi", intr, points_cam)
        xy = points_2d_homo[:, :, :2] / (points_2d_homo[:, :, 2:] + 1e-8)

        traj2d[v] = xy
        traj_z[v] = z_cam

        # Visibility checks
        in_front = z_cam > 0.01
        in_bounds_x = (xy[:, :, 0] >= 0) & (xy[:, :, 0] < img_w)
        in_bounds_y = (xy[:, :, 1] >= 0) & (xy[:, :, 1] < img_h)
        vis = in_front & in_bounds_x & in_bounds_y

        # Depth-based occlusion check
        if depth_maps is not None and v < depth_maps.shape[0]:
            for t in range(T):
                if not vis[t].any():
                    continue
                visible_mask = vis[t]
                px = np.clip(xy[t, visible_mask, 0].astype(int), 0, img_w - 1)
                py = np.clip(xy[t, visible_mask, 1].astype(int), 0, img_h - 1)
                scene_depth = depth_maps[v, t, py, px]
                point_depth = z_cam[t, visible_mask]

                # Point is occluded if it's significantly behind the scene depth
                occluded = point_depth > scene_depth * (1 + depth_tolerance)
                valid_depth = scene_depth > 0.01  # only check where depth is valid
                vis_update = visible_mask.copy()
                vis_update[visible_mask] = ~(occluded & valid_depth)
                vis[t] = vis_update

        visibility[v] = vis

    return visibility, traj2d, traj_z


def process_sequence(seq_path, n_vertex_samples=500, smooth_kernel=3,
                     skip_existing=True, max_persons=None):
    """
    Full pipeline: load predictions, fuse, sample tracks, compute visibility.
    """
    output_file = os.path.join(seq_path, "human_tracks.npz")
    if skip_existing and os.path.exists(output_file):
        print(f"  [SKIP] {output_file} already exists")
        return

    pred_dir = os.path.join(seq_path, "sam3d_predictions")
    if not os.path.isdir(pred_dir):
        warnings.warn(f"No SAM3D predictions found at {pred_dir}")
        return

    ann = load_annotations(seq_path)
    extrinsics = ann["extrinsics"]  # [n_views, 4, 4] or [n_views, 3, 4]
    intrinsics = ann["intrinsics"]  # [n_views, 3, 3]
    n_views = extrinsics.shape[0]

    # Determine number of frames from the image directory
    ims_path = os.path.join(seq_path, "ims")
    view_folders = sorted(os.listdir(ims_path), key=lambda x: int(x))
    first_view_dir = os.path.join(ims_path, view_folders[0])
    n_frames = len(os.listdir(first_view_dir))

    # Get image dimensions from the first image
    first_img = cv2.imread(os.path.join(first_view_dir, sorted(os.listdir(first_view_dir))[0]))
    img_h, img_w = first_img.shape[:2]

    print(f"  Views: {n_views}, Frames: {n_frames}, Resolution: {img_w}x{img_h}")

    predictions = load_sam3d_predictions(pred_dir, n_views, n_frames)
    if not predictions:
        warnings.warn(f"No predictions loaded for {seq_path}")
        return

    # Load depth maps if available
    depths_path = os.path.join(seq_path, "dynamic3dgs_depth")
    depth_maps = None
    if os.path.isdir(depths_path):
        depth_list = []
        for v_str in view_folders:
            v = int(v_str)
            depth_file = os.path.join(depths_path, f"depths_{v:02d}.npy")
            if os.path.exists(depth_file):
                depth_list.append(np.load(depth_file))  # [T, H, W]
        if depth_list:
            depth_maps = np.stack(depth_list)  # [n_views, T, H, W]

    # Step 1: Associate persons across views and fuse per frame
    all_frame_persons = []
    for frame_idx in range(n_frames):
        groups = associate_persons_across_views(
            predictions, extrinsics, n_views, frame_idx,
            max_distance=1.0,
        )
        if not groups:
            all_frame_persons.append([])
            continue
        fused = fuse_meshes_multiview(predictions, extrinsics, groups, frame_idx)
        all_frame_persons.append(fused)

    # Determine consistent number of persons (use mode across frames)
    person_counts = [len(fp) for fp in all_frame_persons if len(fp) > 0]
    if not person_counts:
        warnings.warn(f"No persons detected in any frame of {seq_path}")
        np.savez_compressed(output_file, n_persons=0, n_tracks=0)
        return

    n_persons_target = max(set(person_counts), key=person_counts.count)
    if max_persons is not None:
        n_persons_target = min(n_persons_target, max_persons)

    # Select vertex indices to sample (consistent across frames)
    first_valid = next(fp for fp in all_frame_persons if len(fp) > 0)
    n_mesh_verts = first_valid[0]["vertices"].shape[0]
    vertex_indices = sample_vertices(
        first_valid[0]["vertices"], n_samples=n_vertex_samples
    )

    n_keypoints = 70
    n_sampled_verts = len(vertex_indices)
    n_tracks_per_person = n_keypoints + n_sampled_verts
    n_tracks_total = n_persons_target * n_tracks_per_person

    # Step 2: Build trajectory arrays
    traj3d_world = np.zeros((n_frames, n_tracks_total, 3), dtype=np.float32)
    track_valid = np.zeros((n_frames, n_tracks_total), dtype=bool)

    # Track type labels: 0=keypoint, 1=vertex
    track_types = np.zeros(n_tracks_total, dtype=np.int32)
    track_person_ids = np.zeros(n_tracks_total, dtype=np.int32)
    track_point_ids = np.zeros(n_tracks_total, dtype=np.int32)

    for person_idx in range(n_persons_target):
        base = person_idx * n_tracks_per_person
        # Keypoints
        track_types[base:base + n_keypoints] = 0
        track_person_ids[base:base + n_keypoints] = person_idx
        track_point_ids[base:base + n_keypoints] = np.arange(n_keypoints)
        # Vertices
        track_types[base + n_keypoints:base + n_tracks_per_person] = 1
        track_person_ids[base + n_keypoints:base + n_tracks_per_person] = person_idx
        track_point_ids[base + n_keypoints:base + n_tracks_per_person] = vertex_indices

    min_views_for_valid = 3
    n_views_per_frame = np.zeros(n_frames, dtype=int)

    for frame_idx in range(n_frames):
        frame_persons = all_frame_persons[frame_idx]
        for person_idx in range(min(len(frame_persons), n_persons_target)):
            person = frame_persons[person_idx]
            base = person_idx * n_tracks_per_person
            n_fused = person["n_views_fused"]
            n_views_per_frame[frame_idx] = max(n_views_per_frame[frame_idx], n_fused)

            if n_fused < min_views_for_valid:
                continue

            # Keypoints
            traj3d_world[frame_idx, base:base + n_keypoints] = person["keypoints_3d"]
            track_valid[frame_idx, base:base + n_keypoints] = True

            # Sampled vertices
            verts = person["vertices"][vertex_indices]
            traj3d_world[frame_idx, base + n_keypoints:base + n_tracks_per_person] = verts
            track_valid[frame_idx, base + n_keypoints:base + n_tracks_per_person] = True

    # Interpolate frames that were rejected due to low view count
    n_rejected = (n_views_per_frame < min_views_for_valid).sum()
    if n_rejected > 0:
        print(f"  Rejected {n_rejected} frames with <{min_views_for_valid} views, interpolating...")
        for track_idx in range(n_tracks_total):
            valid_frames = np.where(track_valid[:, track_idx])[0]
            if len(valid_frames) == 0:
                continue
            invalid_frames = np.where(~track_valid[:, track_idx])[0]
            for dim in range(3):
                traj3d_world[invalid_frames, track_idx, dim] = np.interp(
                    invalid_frames,
                    valid_frames,
                    traj3d_world[valid_frames, track_idx, dim],
                )
            track_valid[invalid_frames, track_idx] = True

    # Step 3: Temporal smoothing
    valid_mask = track_valid.all(axis=0)
    if valid_mask.any():
        traj3d_world[:, valid_mask] = temporal_smooth(
            traj3d_world[:, valid_mask], kernel_size=smooth_kernel
        )

    # Step 4: Compute visibility
    visibility, traj2d, traj_z = compute_visibility(
        traj3d_world, extrinsics, intrinsics, img_h, img_w,
        depth_maps=depth_maps,
    )

    # Mask visibility where tracks are not valid
    visibility = visibility & track_valid[None, :, :]

    # Build 2D+Z trajectory: [n_views, T, N, 3] (x_px, y_px, z_cam)
    traj2d_w_z = np.concatenate([traj2d, traj_z[:, :, :, None]], axis=-1)

    print(f"  Tracks: {n_tracks_total} ({n_persons_target} persons x "
          f"{n_tracks_per_person} points [{n_keypoints} kp + {n_sampled_verts} verts])")
    print(f"  Visibility mean: {visibility.mean():.3f}")

    np.savez_compressed(
        output_file,
        # Core trajectory data
        trajectories=traj3d_world.astype(np.float32),           # [T, N, 3]
        trajectories_pixelspace=traj2d.astype(np.float32),      # [V, T, N, 2]
        per_view_visibilities=visibility,                        # [V, T, N]
        traj2d_w_z=traj2d_w_z.astype(np.float32),             # [V, T, N, 3]

        # Camera parameters (same as in annotations)
        extrinsics=extrinsics,
        intrinsics=intrinsics,

        # Metadata
        n_persons=n_persons_target,
        n_tracks=n_tracks_total,
        n_keypoints=n_keypoints,
        n_sampled_verts=n_sampled_verts,
        vertex_indices=vertex_indices,
        track_types=track_types,           # 0=keypoint, 1=vertex
        track_person_ids=track_person_ids,
        track_point_ids=track_point_ids,
        track_valid=track_valid,           # [T, N] whether this track has data at this frame
        img_h=img_h,
        img_w=img_w,
    )
    print(f"  Saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate human tracks from SAM3D predictions")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory of panoptic-multiview dataset")
    parser.add_argument("--n_vertex_samples", type=int, default=500,
                        help="Number of mesh vertices to sample per person")
    parser.add_argument("--smooth_kernel", type=int, default=3,
                        help="Temporal smoothing kernel size (1=no smoothing)")
    parser.add_argument("--seq_index", type=int, default=None,
                        help="Process only this sequence index (for SLURM array jobs)")
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--no_skip_existing", action="store_true", default=False)
    parser.add_argument("--max_persons", type=int, default=None,
                        help="Maximum number of persons to track per sequence")
    args = parser.parse_args()

    skip_existing = args.skip_existing and not args.no_skip_existing

    seq_index = args.seq_index
    if seq_index is None and "SLURM_ARRAY_TASK_ID" in os.environ:
        seq_index = int(os.environ["SLURM_ARRAY_TASK_ID"])

    print("=== Generate Human Tracks from SAM3D Predictions ===")
    print(f"Data root:        {args.data_root}")
    print(f"Vertex samples:   {args.n_vertex_samples}")
    print(f"Smooth kernel:    {args.smooth_kernel}")
    print(f"Skip existing:    {skip_existing}")

    sequences = get_sequences(args.data_root)
    print(f"Found {len(sequences)} sequences")

    if seq_index is not None:
        if seq_index >= len(sequences):
            print(f"Sequence index {seq_index} out of range (max {len(sequences) - 1})")
            return
        sequences = [sequences[seq_index]]
        print(f"Processing sequence index {seq_index}: {sequences[0]}")

    for seq_name in sequences:
        seq_path = os.path.join(args.data_root, seq_name)
        print(f"\nSequence: {seq_name}")
        process_sequence(
            seq_path,
            n_vertex_samples=args.n_vertex_samples,
            smooth_kernel=args.smooth_kernel,
            skip_existing=skip_existing,
            max_persons=args.max_persons,
        )

    print("\n=== Track generation complete ===")


if __name__ == "__main__":
    main()
