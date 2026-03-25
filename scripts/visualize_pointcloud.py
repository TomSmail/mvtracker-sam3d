"""
Visualize MVTracker point clouds: baseline (depth-only) vs SAM3D-augmented.

Loads one sample from the Panoptic human dataset, constructs the depth-based
point cloud, optionally augments with SAM3D mesh vertices, and renders two
side-by-side 3D scatter plots saved as PNG images.

Usage:
    python scripts/visualize_pointcloud.py
"""

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mvtracker.datasets.panoptic_human_trajectory_dataset import PanopticHumanTrajectoryDataset
from mvtracker.models.core.model_utils import init_pointcloud_from_rgbd, augment_pointcloud_with_mesh_vertices


def subsample(xyz, max_pts=20000, rng=None):
    """Randomly subsample points for visualization."""
    N = xyz.shape[0]
    if N <= max_pts:
        return xyz
    if rng is None:
        rng = np.random.default_rng(42)
    idx = rng.choice(N, max_pts, replace=False)
    return xyz[idx]


def color_by_height(xyz):
    """Color points by z-coordinate (height)."""
    z = xyz[:, 2]
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-8)
    return plt.cm.viridis(z_norm)


def render_pointcloud(ax, xyz, color, title, point_size=0.3, alpha=0.4):
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=color, s=point_size, alpha=alpha, rasterized=True)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # Set consistent view angle
    ax.view_init(elev=25, azim=-60)


def main():
    dataset_root = os.environ.get("DATASET_ROOT", "./datasets")
    output_dir = "./logs/pointcloud_vis"
    os.makedirs(output_dir, exist_ok=True)

    # Load one sample from the training dataset
    print("Loading dataset...")
    dataset = PanopticHumanTrajectoryDataset(
        data_root=os.path.join(dataset_root, "panoptic-multiview"),
        views_to_return=[1, 7, 14, 20],
        traj_per_sample=384,
        seed=42,
        max_videos=4,
        use_cached_tracks=False,
        crop_size=[384, 512],
        seq_len=24,
    )

    print(f"Dataset has {len(dataset)} sequences")
    datapoint, _ = dataset[0]
    seq_name = datapoint.seq_name[0] if datapoint.seq_name else "unknown"
    print(f"Loaded sequence: {seq_name}")

    # Extract tensors for a single frame (frame 0)
    # video: (V, T, C, H, W), depth: (V, T, 1, H, W)
    rgbs = datapoint.video.unsqueeze(0)       # (1, V, T, C, H, W)
    depths = datapoint.videodepth.unsqueeze(0) # (1, V, T, 1, H, W)
    intrs = datapoint.intrs.unsqueeze(0)       # (1, V, T, 3, 3)
    extrs = datapoint.extrs.unsqueeze(0)       # (1, V, T, 3, 4)

    B, V, T, C, H, W = rgbs.shape
    print(f"Shape: B={B}, V={V}, T={T}, C={C}, H={H}, W={W}")

    # Use frame 0 only: reshape to (B, V, S=1, ...)
    frame_idx = 0
    rgbs_f = rgbs[:, :, frame_idx:frame_idx+1]
    depths_f = depths[:, :, frame_idx:frame_idx+1]
    intrs_f = intrs[:, :, frame_idx:frame_idx+1]
    extrs_f = extrs[:, :, frame_idx:frame_idx+1]

    # Create dummy feature maps (use RGB as stand-in, we only need xyz for vis)
    # Real features are 128-d from BasicEncoder, but for point cloud xyz we just need depth+camera
    # Use a simple 3-channel "feature" from RGB for coloring
    fmaps = rgbs_f  # (B, V, 1, C, H, W) - C=3 here

    print("Building baseline point cloud (depth-only)...")
    pc_xyz, pc_fvec, pc_valid = init_pointcloud_from_rgbd(
        fmaps=fmaps,
        depths=depths_f,
        intrs=intrs_f,
        extrs=extrs_f,
        stride=1,  # stride=1 since we're using full-res
        level=0,
        return_validity_mask=True,
    )
    # pc_xyz: (B*S, V*H*W, 3), pc_valid: (B*S, V*H*W)
    print(f"Baseline point cloud: {pc_xyz.shape[1]} points")

    # Filter valid points (depth > 0)
    valid_mask = pc_valid[0].bool()
    baseline_xyz = pc_xyz[0, valid_mask].numpy()
    baseline_rgb = pc_fvec[0, valid_mask].numpy()  # RGB features for coloring
    print(f"Valid baseline points: {baseline_xyz.shape[0]}")

    # SAM3D mesh vertices
    sam3d_verts = datapoint.sam3d_vertices_world  # (T, M, 3) or None
    if sam3d_verts is not None:
        mesh_verts_f = sam3d_verts[frame_idx:frame_idx+1]  # (1, M, 3)
        M = mesh_verts_f.shape[1]
        print(f"SAM3D mesh vertices: {M} points")

        # Build augmented point cloud
        aug_xyz, aug_fvec, aug_valid = augment_pointcloud_with_mesh_vertices(
            pointcloud_xyz=pc_xyz.clone(),
            pointcloud_fvec=pc_fvec.clone(),
            mesh_vertices=mesh_verts_f,
            fmaps=fmaps,
            intrs=intrs_f,
            extrs=extrs_f,
            stride=1,
            level=0,
            pointcloud_valid=pc_valid.clone(),
        )
        print(f"Augmented point cloud: {aug_xyz.shape[1]} points ({pc_xyz.shape[1]} depth + {M} mesh)")

        aug_valid_mask = aug_valid[0].bool()
        aug_all_xyz = aug_xyz[0, aug_valid_mask].numpy()

        # Separate depth points and mesh points for coloring
        n_depth = pc_xyz.shape[1]
        depth_mask = aug_valid_mask[:n_depth]
        mesh_mask = aug_valid_mask[n_depth:]

        depth_pts = aug_xyz[0, :n_depth][depth_mask].numpy()
        mesh_pts = aug_xyz[0, n_depth:][mesh_mask].numpy()
    else:
        print("WARNING: No SAM3D vertices in this sample!")
        mesh_pts = None

    # --- Render ---
    rng = np.random.default_rng(42)

    # Subsample for rendering
    baseline_sub = subsample(baseline_xyz, max_pts=15000, rng=rng)
    if mesh_pts is not None:
        depth_sub = subsample(depth_pts, max_pts=15000, rng=rng)

    # Compute axis limits from baseline for consistency
    all_pts = baseline_sub
    center = all_pts.mean(axis=0)
    spread = float(np.percentile(np.abs(all_pts - center), 95, axis=0).max())
    xlim = (float(center[0]) - spread * 1.3, float(center[0]) + spread * 1.3)
    ylim = (float(center[1]) - spread * 1.3, float(center[1]) + spread * 1.3)
    zlim = (float(center[2]) - spread * 1.3, float(center[2]) + spread * 1.3)

    # --- Figure 1: Baseline only ---
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 8), subplot_kw={"projection": "3d"})
    colors_baseline = color_by_height(baseline_sub)
    render_pointcloud(ax1, baseline_sub, colors_baseline,
                      f"MVTracker Point Cloud (depth-only)\n{seq_name} — frame {frame_idx} — {baseline_xyz.shape[0]:,} pts")
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_zlim(zlim)
    fig1.tight_layout()
    path1 = os.path.join(output_dir, f"pointcloud_baseline_{seq_name}_f{frame_idx}.png")
    fig1.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved: {path1}")

    # --- Figure 2: With SAM3D mesh ---
    if mesh_pts is not None:
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8), subplot_kw={"projection": "3d"})

        # Depth points in gray/muted
        colors_depth = color_by_height(depth_sub)
        colors_depth[:, 3] = 0.15  # very transparent
        ax2.scatter(depth_sub[:, 0], depth_sub[:, 1], depth_sub[:, 2],
                    c=colors_depth, s=0.3, rasterized=True)

        # SAM3D mesh points highlighted in red
        ax2.scatter(mesh_pts[:, 0], mesh_pts[:, 1], mesh_pts[:, 2],
                    c="red", s=3.0, alpha=0.8, label=f"SAM3D mesh ({mesh_pts.shape[0]:,} verts)",
                    rasterized=True)

        ax2.set_title(f"MVTracker + SAM3D Augmented Point Cloud\n{seq_name} — frame {frame_idx} — "
                       f"{depth_pts.shape[0]:,} depth + {mesh_pts.shape[0]:,} mesh pts",
                       fontsize=14, fontweight='bold')
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z")
        ax2.view_init(elev=25, azim=-60)
        ax2.set_xlim(xlim)
        ax2.set_ylim(ylim)
        ax2.set_zlim(zlim)
        ax2.legend(loc="upper right", fontsize=11)
        fig2.tight_layout()
        path2 = os.path.join(output_dir, f"pointcloud_sam3d_{seq_name}_f{frame_idx}.png")
        fig2.savefig(path2, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved: {path2}")

    # --- Figure 3: Side by side ---
    if mesh_pts is not None:
        fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(20, 8),
                                           subplot_kw={"projection": "3d"})

        # Left: baseline
        colors_left = color_by_height(baseline_sub)
        render_pointcloud(ax3a, baseline_sub, colors_left,
                          f"Baseline (depth-only)\n{baseline_xyz.shape[0]:,} pts")
        ax3a.set_xlim(xlim)
        ax3a.set_ylim(ylim)
        ax3a.set_zlim(zlim)

        # Right: augmented
        colors_depth_r = color_by_height(depth_sub)
        colors_depth_r[:, 3] = 0.15
        ax3b.scatter(depth_sub[:, 0], depth_sub[:, 1], depth_sub[:, 2],
                     c=colors_depth_r, s=0.3, rasterized=True)
        ax3b.scatter(mesh_pts[:, 0], mesh_pts[:, 1], mesh_pts[:, 2],
                     c="red", s=3.0, alpha=0.8, label=f"SAM3D mesh ({mesh_pts.shape[0]:,} verts)",
                     rasterized=True)
        ax3b.set_title(f"+ SAM3D Augmentation\n{depth_pts.shape[0]:,} depth + {mesh_pts.shape[0]:,} mesh pts",
                        fontsize=14, fontweight='bold')
        ax3b.set_xlabel("X")
        ax3b.set_ylabel("Y")
        ax3b.set_zlabel("Z")
        ax3b.view_init(elev=25, azim=-60)
        ax3b.set_xlim(xlim)
        ax3b.set_ylim(ylim)
        ax3b.set_zlim(zlim)
        ax3b.legend(loc="upper right", fontsize=11)

        fig3.suptitle(f"Point Cloud Comparison — {seq_name}, frame {frame_idx}", fontsize=16, fontweight="bold")
        fig3.tight_layout()
        path3 = os.path.join(output_dir, f"pointcloud_comparison_{seq_name}_f{frame_idx}.png")
        fig3.savefig(path3, dpi=150, bbox_inches="tight")
        plt.close(fig3)
        print(f"Saved: {path3}")

    print("\nDone! Images saved to:", output_dir)


if __name__ == "__main__":
    main()
