"""
Create tapvid3d_annotations.npz from Panoptic Studio calibration JSONs.

Parses the raw calibration JSON, extracts HD camera intrinsics and extrinsics,
scales intrinsics to 640x360, converts translations from cm to meters, and
saves in the format expected by the SAM3D inference and track generation pipeline.

Usage:
    python scripts/data_engine/create_calibration.py --seq_index 0
    python scripts/data_engine/create_calibration.py  # all sequences
"""

import argparse
import json
import os
import sys

import numpy as np


def get_sequences(raw_root):
    """List sequence directories sorted alphabetically."""
    return sorted([
        d for d in os.listdir(raw_root)
        if os.path.isdir(os.path.join(raw_root, d))
        and not d.startswith(".")
        and not d.startswith("_")
    ])


def create_calibration(raw_root, output_root, seq_name, target_w, target_h,
                        native_w, native_h, skip_existing):
    """Create tapvid3d_annotations.npz for one sequence."""
    output_dir = os.path.join(output_root, seq_name)
    output_path = os.path.join(output_dir, "tapvid3d_annotations.npz")

    if skip_existing and os.path.exists(output_path):
        print(f"  [SKIP] {output_path} already exists")
        return True

    # Find calibration JSON
    cal_path = os.path.join(raw_root, seq_name, f"calibration_{seq_name}.json")
    if not os.path.exists(cal_path):
        print(f"  [FAIL] No calibration file: {cal_path}")
        return False

    # Check that ims/ directory exists (frames must be extracted first)
    ims_dir = os.path.join(output_dir, "ims")
    if not os.path.isdir(ims_dir):
        print(f"  [FAIL] No ims/ directory at {ims_dir} — extract frames first")
        return False

    # Parse calibration JSON
    with open(cal_path) as f:
        cal = json.load(f)

    # Filter to HD cameras, sorted by node number
    hd_cams = sorted(
        [c for c in cal["cameras"] if c["type"] == "hd"],
        key=lambda c: c["node"]
    )

    if not hd_cams:
        print(f"  [FAIL] No HD cameras in calibration")
        return False

    # Only include cameras that have extracted frames
    available_views = set(
        int(d) for d in os.listdir(ims_dir)
        if os.path.isdir(os.path.join(ims_dir, d)) and d.isdigit()
    )

    scale_x = target_w / native_w
    scale_y = target_h / native_h

    intrinsics_list = []
    extrinsics_list = []
    included_nodes = []

    for cam in hd_cams:
        node = cam["node"]
        if node not in available_views:
            continue

        # Intrinsics: scale from native to target resolution
        K = np.array(cam["K"]).reshape(3, 3)
        K_scaled = K.copy()
        K_scaled[0, :] *= scale_x  # fx, skew, cx
        K_scaled[1, :] *= scale_y  # fy, cy
        intrinsics_list.append(K_scaled)

        # Extrinsics: R as-is, t from cm to meters
        R = np.array(cam["R"]).reshape(3, 3)
        t = np.array(cam["t"]).flatten() / 100.0  # cm -> meters

        extr = np.eye(4)
        extr[:3, :3] = R
        extr[:3, 3] = t
        extrinsics_list.append(extr)
        included_nodes.append(node)

    if not intrinsics_list:
        print(f"  [FAIL] No HD cameras match available views")
        return False

    intrinsics = np.stack(intrinsics_list)  # (V, 3, 3)
    extrinsics = np.stack(extrinsics_list)  # (V, 4, 4)

    # Save with empty trajectory placeholders (downstream scripts may check for these keys)
    n_views = len(intrinsics_list)
    os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(
        output_path,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        # Empty trajectory placeholders
        trajectories=np.zeros((0, 0, 3), dtype=np.float64),
        trajectories_pixelspace=np.zeros((n_views, 0, 0, 2), dtype=np.float64),
        per_view_visibilities=np.zeros((n_views, 0, 0), dtype=bool),
        query_points_3d=np.zeros((0, 4), dtype=np.float64),
    )

    print(f"  [OK] {n_views} cameras (nodes: {included_nodes})")
    print(f"  Intrinsics: {intrinsics.shape}, Extrinsics: {extrinsics.shape}")
    print(f"  Saved: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Create calibration from Panoptic JSON")
    parser.add_argument("--raw_root", default=os.path.join(
        os.environ.get("SCRATCH", "/cluster/scratch/tsmail"), "datasets", "panoptic-raw"))
    parser.add_argument("--output_root", default=os.path.join(
        os.environ.get("SCRATCH", "/cluster/scratch/tsmail"), "datasets", "panoptic-multiview"))
    parser.add_argument("--seq_index", type=int, default=None,
                        help="Process only this sequence index. Reads SLURM_ARRAY_TASK_ID if not set.")
    parser.add_argument("--target_width", type=int, default=640)
    parser.add_argument("--target_height", type=int, default=360)
    parser.add_argument("--native_width", type=int, default=1920)
    parser.add_argument("--native_height", type=int, default=1080)
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    args = parser.parse_args()

    seq_index = args.seq_index
    if seq_index is None:
        seq_index = os.environ.get("SLURM_ARRAY_TASK_ID")
        if seq_index is not None:
            seq_index = int(seq_index)

    sequences = get_sequences(args.raw_root)
    if not sequences:
        print(f"No sequences found in {args.raw_root}")
        sys.exit(1)

    if seq_index is not None:
        if seq_index >= len(sequences):
            print(f"seq_index {seq_index} out of range (have {len(sequences)} sequences)")
            sys.exit(1)
        sequences = [sequences[seq_index]]

    print(f"=== Creating calibration for {len(sequences)} sequence(s) ===")
    print(f"Raw root:    {args.raw_root}")
    print(f"Output root: {args.output_root}")
    print(f"Resolution:  {args.target_width}x{args.target_height} (from {args.native_width}x{args.native_height})")
    print()

    n_ok = 0
    n_fail = 0
    for seq in sequences:
        print(f"[{seq}]")
        if create_calibration(args.raw_root, args.output_root, seq,
                              args.target_width, args.target_height,
                              args.native_width, args.native_height,
                              args.skip_existing):
            n_ok += 1
        else:
            n_fail += 1
        print()

    print(f"=== Done: {n_ok} ok, {n_fail} failed ===")


if __name__ == "__main__":
    main()
