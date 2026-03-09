"""
Visualize fused world-space human tracks from human_tracks.npz overlaid on frames.

Shows the multi-view-fused, temporally-smoothed trajectories projected back
onto any camera view. Color-coded by person and track type (keypoint vs vertex).

Usage:
    # Single frame:
    python scripts/data_engine/visualize_tracks.py \
        --data_root /cluster/scratch/tsmail/datasets/panoptic-multiview \
        --seq basketball --view 1 --frame 0

    # Video:
    python scripts/data_engine/visualize_tracks.py \
        --data_root /cluster/scratch/tsmail/datasets/panoptic-multiview \
        --seq basketball --view 1 --output tracks_basketball_v1.mp4

    # Side-by-side with raw SAM3D predictions:
    python scripts/data_engine/visualize_tracks.py \
        --data_root /cluster/scratch/tsmail/datasets/panoptic-multiview \
        --seq basketball --view 1 --frame 0 --side_by_side
"""

import argparse
import os
import sys

import cv2
import numpy as np


PERSON_COLORS = [
    (46, 204, 113),   # green
    (231, 76, 60),    # red
    (52, 152, 219),   # blue
    (241, 196, 15),   # yellow
    (155, 89, 182),   # purple
    (230, 126, 34),   # orange
    (26, 188, 156),   # teal
    (192, 57, 43),    # dark red
]

SKELETON = [
    (13, 11), (11, 9), (14, 12), (12, 10),
    (9, 10), (5, 9), (6, 10), (5, 6),
    (5, 7), (6, 8), (7, 62), (8, 41),
    (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
    (13, 15), (13, 16), (13, 17),
    (14, 18), (14, 19), (14, 20),
]

HAND_SKELETON_RIGHT = [
    (41, 24), (24, 23), (23, 22), (22, 21),
    (41, 28), (28, 27), (27, 26), (26, 25),
    (41, 32), (32, 31), (31, 30), (30, 29),
    (41, 36), (36, 35), (35, 34), (34, 33),
    (41, 40), (40, 39), (39, 38), (38, 37),
]

HAND_SKELETON_LEFT = [
    (62, 45), (45, 44), (44, 43), (43, 42),
    (62, 49), (49, 48), (48, 47), (47, 46),
    (62, 53), (53, 52), (52, 51), (51, 50),
    (62, 57), (57, 56), (56, 55), (55, 54),
    (62, 61), (61, 60), (60, 59), (59, 58),
]


def brighten(color, factor=1.4):
    return tuple(min(int(c * factor), 255) for c in color)


def darken(color, factor=0.5):
    return tuple(int(c * factor) for c in color)


def render_tracks_frame(img_rgb, tracks_data, view_idx, frame_idx, show_vertices=True,
                        show_trails=False, trail_length=5):
    """Render fused tracks on an RGB image for a given view and frame."""
    vis = img_rgb.copy()
    h, w = vis.shape[:2]

    traj2d = tracks_data["trajectories_pixelspace"]   # [V, T, N, 2]
    visibility = tracks_data["per_view_visibilities"]  # [V, T, N]
    track_types = tracks_data["track_types"]           # [N]
    track_person_ids = tracks_data["track_person_ids"] # [N]
    track_point_ids = tracks_data["track_point_ids"]   # [N]
    track_valid = tracks_data["track_valid"]           # [T, N]
    n_persons = int(tracks_data["n_persons"])
    n_keypoints = int(tracks_data["n_keypoints"])
    n_tracks_per_person = n_keypoints + int(tracks_data["n_sampled_verts"])

    pts = traj2d[view_idx, frame_idx]     # [N, 2]
    vis_mask = visibility[view_idx, frame_idx]  # [N]
    valid = track_valid[frame_idx]        # [N]
    active = vis_mask & valid

    for person_idx in range(n_persons):
        base_color = PERSON_COLORS[person_idx % len(PERSON_COLORS)]
        vert_color = darken(base_color, 0.6)
        kp_color = base_color
        hand_color = brighten(base_color, 1.3)

        person_mask = track_person_ids == person_idx
        base_offset = person_idx * n_tracks_per_person

        # Draw vertex tracks (dimmer, smaller)
        if show_vertices:
            vert_mask = person_mask & (track_types == 1) & active
            vert_pts = pts[vert_mask]
            for pt in vert_pts:
                x, y = int(pt[0]), int(pt[1])
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(vis, (x, y), 1, vert_color, -1)

        # Draw motion trails
        if show_trails and frame_idx > 0:
            kp_mask = person_mask & (track_types == 0)
            kp_indices = np.where(kp_mask)[0]
            for idx in kp_indices:
                if not active[idx]:
                    continue
                trail_pts = []
                for t in range(max(0, frame_idx - trail_length), frame_idx + 1):
                    if visibility[view_idx, t, idx] and track_valid[t, idx]:
                        trail_pts.append(traj2d[view_idx, t, idx].astype(int))
                if len(trail_pts) >= 2:
                    for k in range(len(trail_pts) - 1):
                        alpha = (k + 1) / len(trail_pts)
                        c = tuple(int(c * alpha) for c in darken(base_color, 0.7))
                        cv2.line(vis, tuple(trail_pts[k]), tuple(trail_pts[k + 1]),
                                 c, 1, cv2.LINE_AA)

        # Draw skeleton on the keypoint subset for this person
        kp_start = base_offset
        kp_end = base_offset + n_keypoints
        person_kp_pts = pts[kp_start:kp_end]        # [70, 2]
        person_kp_active = active[kp_start:kp_end]  # [70]

        for i, j in SKELETON:
            if person_kp_active[i] and person_kp_active[j]:
                pt1 = tuple(person_kp_pts[i].astype(int))
                pt2 = tuple(person_kp_pts[j].astype(int))
                cv2.line(vis, pt1, pt2, kp_color, 2, cv2.LINE_AA)

        for skel, col in [(HAND_SKELETON_RIGHT, hand_color), (HAND_SKELETON_LEFT, hand_color)]:
            for i, j in skel:
                if person_kp_active[i] and person_kp_active[j]:
                    pt1 = tuple(person_kp_pts[i].astype(int))
                    pt2 = tuple(person_kp_pts[j].astype(int))
                    cv2.line(vis, pt1, pt2, col, 1, cv2.LINE_AA)

        # Draw keypoint dots
        for i in range(n_keypoints):
            if person_kp_active[i]:
                x, y = int(person_kp_pts[i, 0]), int(person_kp_pts[i, 1])
                if 0 <= x < w and 0 <= y < h:
                    r = 3 if i < 21 or i >= 63 else 2  # body/feet bigger, hands smaller
                    cv2.circle(vis, (x, y), r, kp_color, -1, cv2.LINE_AA)
                    cv2.circle(vis, (x, y), r, (0, 0, 0), 1, cv2.LINE_AA)

    # Stats overlay
    n_visible = active.sum()
    n_total = len(active)
    label = (f"seq={os.path.basename(tracks_data['_seq_path'])} view={view_idx} "
             f"frame={frame_idx} tracks={n_visible}/{n_total}")
    cv2.putText(vis, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.putText(vis, "[FUSED TRACKS]", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (46, 204, 113), 1, cv2.LINE_AA)

    return vis


def render_raw_sam3d_frame(img_rgb, seq_path, view_idx, frame_idx, intrinsics):
    """Render raw SAM3D predictions for side-by-side comparison."""
    from visualize_sam3d import render_frame, load_intrinsics

    pred_dir = os.path.join(seq_path, "sam3d_predictions", f"view_{view_idx:02d}")
    pred_files = sorted(os.listdir(pred_dir))
    if frame_idx >= len(pred_files):
        return img_rgb.copy()

    pred = np.load(os.path.join(pred_dir, pred_files[frame_idx]))
    intr = intrinsics[view_idx] if intrinsics is not None else None
    if intr is None:
        h, w = img_rgb.shape[:2]
        fx = fy = max(h, w)
        intr = np.array([[fx, 0, w / 2], [0, fy, h / 2], [0, 0, 1]])

    vis = render_frame(img_rgb, pred, intr)

    h = vis.shape[0]
    cv2.putText(vis, "[RAW SAM3D]", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 128, 255), 1, cv2.LINE_AA)
    return vis


def main():
    parser = argparse.ArgumentParser(description="Visualize fused human tracks on Panoptic frames")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--seq", type=str, required=True, help="Sequence name")
    parser.add_argument("--view", type=int, required=True, help="View index")
    parser.add_argument("--frame", type=int, default=None, help="Single frame to render")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--output", type=str, default=None, help="Output .jpg or .mp4")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--no_vertices", action="store_true")
    parser.add_argument("--trails", action="store_true", help="Show motion trails")
    parser.add_argument("--trail_length", type=int, default=5)
    parser.add_argument("--side_by_side", action="store_true",
                        help="Show fused tracks next to raw SAM3D predictions")
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()

    seq_path = os.path.join(args.data_root, args.seq)
    tracks_file = os.path.join(seq_path, "human_tracks.npz")
    rgb_dir = os.path.join(seq_path, "ims", str(args.view))

    if not os.path.exists(tracks_file):
        print(f"Error: {tracks_file} not found. Run generate_human_tracks.py first.")
        sys.exit(1)
    if not os.path.isdir(rgb_dir):
        print(f"Error: images not found at {rgb_dir}")
        sys.exit(1)

    tracks_data = dict(np.load(tracks_file))
    tracks_data["_seq_path"] = seq_path

    rgb_files = sorted(os.listdir(rgb_dir))
    n_frames = tracks_data["trajectories"].shape[0]

    if args.frame is not None:
        start, end = args.frame, args.frame + 1
    else:
        start = args.start
        end = args.end or n_frames
    end = min(end, n_frames, len(rgb_files))

    n_views = tracks_data["extrinsics"].shape[0]
    if args.view >= n_views:
        print(f"Error: view {args.view} out of range (max {n_views - 1})")
        sys.exit(1)

    print(f"Sequence: {args.seq}, View: {args.view}")
    print(f"Frames: {start}-{end - 1} ({end - start} total)")
    print(f"Tracks: {int(tracks_data['n_tracks'])} "
          f"({int(tracks_data['n_persons'])} persons, "
          f"{int(tracks_data['n_keypoints'])} kp + "
          f"{int(tracks_data['n_sampled_verts'])} verts each)")

    if args.side_by_side:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    writer = None
    for frame_idx in range(start, end):
        img_bgr = cv2.imread(os.path.join(rgb_dir, rgb_files[frame_idx]))
        img_rgb = img_bgr[:, :, ::-1]

        vis_tracks = render_tracks_frame(
            img_rgb, tracks_data, args.view, frame_idx,
            show_vertices=not args.no_vertices,
            show_trails=args.trails,
            trail_length=args.trail_length,
        )

        if args.side_by_side:
            vis_raw = render_raw_sam3d_frame(
                img_rgb, seq_path, args.view, frame_idx,
                tracks_data["intrinsics"],
            )
            vis_combined = np.concatenate([vis_raw, vis_tracks], axis=1)
        else:
            vis_combined = vis_tracks

        vis_bgr = vis_combined[:, :, ::-1]

        if args.output and args.output.endswith(".mp4"):
            if writer is None:
                h_out, w_out = vis_bgr.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.output, fourcc, args.fps, (w_out, h_out))
            writer.write(vis_bgr)
        elif args.output_dir:
            out_path = os.path.join(args.output_dir, f"frame_{frame_idx:05d}.jpg")
            cv2.imwrite(out_path, vis_bgr)
        elif args.output and args.frame is not None:
            cv2.imwrite(args.output, vis_bgr)
            print(f"Saved to {args.output}")

        if frame_idx % 10 == 0:
            n_vis = (tracks_data["per_view_visibilities"][args.view, frame_idx]
                     & tracks_data["track_valid"][frame_idx]).sum()
            print(f"  Frame {frame_idx}: {n_vis} visible tracks")

    if writer is not None:
        writer.release()
        print(f"Video saved to {args.output}")

    if args.output_dir:
        print(f"Frames saved to {args.output_dir}/")

    if args.frame is not None and not args.output:
        out_path = f"tracks_viz_{args.seq}_v{args.view}_f{args.frame}.jpg"
        cv2.imwrite(out_path, vis_bgr)
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
