"""
Visualize SAM3D predictions overlaid on source frames.

Projects predicted 3D keypoints and a subset of mesh vertices onto the original
RGB frame, drawing skeleton connections and color-coded points.

Usage:
    python scripts/data_engine/visualize_sam3d.py \
        --data_root /cluster/scratch/tsmail/datasets/panoptic-multiview \
        --seq boxes --view 1 --frame 0

    # Render a range of frames to a video:
    python scripts/data_engine/visualize_sam3d.py \
        --data_root /cluster/scratch/tsmail/datasets/panoptic-multiview \
        --seq boxes --view 1 --start 0 --end 30 --output video.mp4

    # Render all frames as individual images:
    python scripts/data_engine/visualize_sam3d.py \
        --data_root /cluster/scratch/tsmail/datasets/panoptic-multiview \
        --seq boxes --view 1 --output_dir viz_output/
"""

import argparse
import os
import sys

import cv2
import numpy as np


SKELETON = [
    (13, 11), (11, 9), (14, 12), (12, 10),       # legs
    (9, 10), (5, 9), (6, 10), (5, 6),             # torso
    (5, 7), (6, 8), (7, 62), (8, 41),             # arms
    (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),  # head
    (13, 15), (13, 16), (13, 17),                  # left foot
    (14, 18), (14, 19), (14, 20),                  # right foot
]

HAND_SKELETON_RIGHT = [
    (41, 24), (24, 23), (23, 22), (22, 21),       # thumb
    (41, 28), (28, 27), (27, 26), (26, 25),       # index
    (41, 32), (32, 31), (31, 30), (30, 29),       # middle
    (41, 36), (36, 35), (35, 34), (34, 33),       # ring
    (41, 40), (40, 39), (39, 38), (38, 37),       # pinky
]

HAND_SKELETON_LEFT = [
    (62, 45), (45, 44), (44, 43), (43, 42),       # thumb
    (62, 49), (49, 48), (48, 47), (47, 46),       # index
    (62, 53), (53, 52), (52, 51), (51, 50),       # middle
    (62, 57), (57, 56), (56, 55), (55, 54),       # ring
    (62, 61), (61, 60), (60, 59), (59, 58),       # pinky
]

BODY_COLOR = (0, 255, 128)
RIGHT_HAND_COLOR = (255, 128, 0)
LEFT_HAND_COLOR = (0, 128, 255)
VERTEX_COLOR = (180, 180, 180)

BODY_INDICES = list(range(21)) + [41, 62, 63, 64, 65, 66, 67, 68, 69]
RIGHT_HAND_INDICES = list(range(21, 42))
LEFT_HAND_INDICES = list(range(42, 63))


def project_to_2d(points_3d, cam_t, intrinsics, img_h, img_w):
    """
    Project camera-space 3D points to 2D pixel coordinates.

    SAM3D outputs points relative to the person, translated by cam_t to get
    camera-space coordinates. We then project using intrinsics.
    """
    points_cam = points_3d + cam_t[None, :]
    points_2d_h = (intrinsics @ points_cam.T).T
    z = points_2d_h[:, 2:3]
    valid = z[:, 0] > 0.01
    points_2d = points_2d_h[:, :2] / (z + 1e-8)
    in_bounds = (
        valid
        & (points_2d[:, 0] >= 0) & (points_2d[:, 0] < img_w)
        & (points_2d[:, 1] >= 0) & (points_2d[:, 1] < img_h)
    )
    return points_2d, in_bounds


def draw_skeleton(img, kp2d, valid, skeleton, color, thickness=2):
    for i, j in skeleton:
        if valid[i] and valid[j]:
            pt1 = tuple(kp2d[i].astype(int))
            pt2 = tuple(kp2d[j].astype(int))
            cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)


def draw_keypoints(img, kp2d, valid, indices, color, radius=4):
    for i in indices:
        if valid[i]:
            pt = tuple(kp2d[i].astype(int))
            cv2.circle(img, pt, radius, color, -1, cv2.LINE_AA)
            cv2.circle(img, pt, radius, (0, 0, 0), 1, cv2.LINE_AA)


def draw_vertices(img, vert2d, valid, color, radius=1, subsample=10):
    for i in range(0, len(vert2d), subsample):
        if valid[i]:
            pt = tuple(vert2d[i].astype(int))
            cv2.circle(img, pt, radius, color, -1)


def render_frame(img_rgb, pred_data, intrinsics, show_vertices=True, vertex_subsample=10):
    """Render SAM3D predictions on top of an RGB image."""
    vis = img_rgb.copy()
    h, w = vis.shape[:2]
    n_persons = int(pred_data["n_persons"])

    for p in range(n_persons):
        prefix = f"person_{p}"
        kp3d = pred_data[f"{prefix}_pred_keypoints_3d"]
        verts = pred_data[f"{prefix}_pred_vertices"]
        cam_t = pred_data[f"{prefix}_pred_cam_t"]
        bbox = pred_data[f"{prefix}_bbox"]

        kp2d, kp_valid = project_to_2d(kp3d, cam_t, intrinsics, h, w)
        vert2d, vert_valid = project_to_2d(verts, cam_t, intrinsics, h, w)

        if show_vertices:
            draw_vertices(vis, vert2d, vert_valid, VERTEX_COLOR, radius=1,
                          subsample=vertex_subsample)

        draw_skeleton(vis, kp2d, kp_valid, SKELETON, BODY_COLOR, thickness=2)
        draw_skeleton(vis, kp2d, kp_valid, HAND_SKELETON_RIGHT, RIGHT_HAND_COLOR, thickness=1)
        draw_skeleton(vis, kp2d, kp_valid, HAND_SKELETON_LEFT, LEFT_HAND_COLOR, thickness=1)

        draw_keypoints(vis, kp2d, kp_valid, BODY_INDICES, BODY_COLOR, radius=4)
        draw_keypoints(vis, kp2d, kp_valid, RIGHT_HAND_INDICES, RIGHT_HAND_COLOR, radius=3)
        draw_keypoints(vis, kp2d, kp_valid, LEFT_HAND_INDICES, LEFT_HAND_COLOR, radius=3)

        # Draw bounding box
        x1, y1, x2, y2 = bbox.astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(vis, f"P{p}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 0), 1, cv2.LINE_AA)

    return vis


def load_intrinsics(seq_path, view_idx):
    ann_file = os.path.join(seq_path, "tapvid3d_annotations.npz")
    if os.path.exists(ann_file):
        ann = np.load(ann_file)
        return ann["intrinsics"][view_idx]
    return None


def get_default_intrinsics(h, w):
    """Fallback intrinsics if no calibration available."""
    fx = fy = max(h, w)
    cx, cy = w / 2, h / 2
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Visualize SAM3D predictions on Panoptic frames")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--seq", type=str, required=True, help="Sequence name (e.g. boxes)")
    parser.add_argument("--view", type=int, required=True, help="View index")
    parser.add_argument("--frame", type=int, default=None, help="Single frame index to display")
    parser.add_argument("--start", type=int, default=0, help="Start frame for range/video")
    parser.add_argument("--end", type=int, default=None, help="End frame (exclusive)")
    parser.add_argument("--output", type=str, default=None, help="Output image or .mp4 video path")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for per-frame images")
    parser.add_argument("--no_vertices", action="store_true", help="Hide mesh vertices")
    parser.add_argument("--vertex_subsample", type=int, default=10)
    args = parser.parse_args()

    seq_path = os.path.join(args.data_root, args.seq)
    pred_dir = os.path.join(seq_path, "sam3d_predictions", f"view_{args.view:02d}")
    rgb_dir = os.path.join(seq_path, "ims", str(args.view))

    if not os.path.isdir(pred_dir):
        print(f"Error: predictions not found at {pred_dir}")
        sys.exit(1)
    if not os.path.isdir(rgb_dir):
        print(f"Error: images not found at {rgb_dir}")
        sys.exit(1)

    intrinsics = load_intrinsics(seq_path, args.view)
    rgb_files = sorted(os.listdir(rgb_dir))
    pred_files = sorted(os.listdir(pred_dir))

    if args.frame is not None:
        start, end = args.frame, args.frame + 1
    else:
        start = args.start
        end = args.end or len(rgb_files)
    end = min(end, len(rgb_files), len(pred_files))

    print(f"Sequence: {args.seq}, View: {args.view}")
    print(f"Frames: {start}-{end - 1} ({end - start} total)")
    print(f"Predictions: {pred_dir}")
    print(f"Images: {rgb_dir}")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    writer = None
    for frame_idx in range(start, end):
        img_bgr = cv2.imread(os.path.join(rgb_dir, rgb_files[frame_idx]))
        img_rgb = img_bgr[:, :, ::-1]
        h, w = img_rgb.shape[:2]

        if intrinsics is None:
            intrinsics = get_default_intrinsics(h, w)

        pred = np.load(os.path.join(pred_dir, pred_files[frame_idx]))
        vis = render_frame(img_rgb, pred, intrinsics,
                           show_vertices=not args.no_vertices,
                           vertex_subsample=args.vertex_subsample)

        label = f"seq={args.seq} view={args.view} frame={frame_idx} persons={int(pred['n_persons'])}"
        cv2.putText(vis, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 1, cv2.LINE_AA)

        vis_bgr = vis[:, :, ::-1]

        if args.output and args.output.endswith(".mp4"):
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.output, fourcc, 15, (w, h))
            writer.write(vis_bgr)
        elif args.output_dir:
            out_path = os.path.join(args.output_dir, f"frame_{frame_idx:05d}.jpg")
            cv2.imwrite(out_path, vis_bgr)
        elif args.output and args.frame is not None:
            cv2.imwrite(args.output, vis_bgr)
            print(f"Saved to {args.output}")

        if frame_idx % 10 == 0:
            print(f"  Frame {frame_idx}: {int(pred['n_persons'])} person(s)")

    if writer is not None:
        writer.release()
        print(f"Video saved to {args.output}")

    if args.output_dir:
        print(f"Frames saved to {args.output_dir}/")

    if args.frame is not None and not args.output:
        out_path = f"sam3d_viz_{args.seq}_v{args.view}_f{args.frame}.jpg"
        cv2.imwrite(out_path, vis_bgr)
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
