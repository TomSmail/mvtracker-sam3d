"""
Extract JPEG frames from Panoptic HD MP4 videos into the panoptic-multiview format.

Resizes from 1920x1080 to 640x360 and outputs 0-indexed frames.
Designed to be called per-sequence from a SLURM array job.

Usage:
    python scripts/data_engine/extract_frames.py --seq_index 0
    python scripts/data_engine/extract_frames.py  # all sequences
"""

import argparse
import os
import subprocess
import sys


def get_sequences(raw_root):
    """List sequence directories sorted alphabetically."""
    return sorted([
        d for d in os.listdir(raw_root)
        if os.path.isdir(os.path.join(raw_root, d))
        and not d.startswith(".")
        and not d.startswith("_")
    ])


def extract_sequence(raw_root, output_root, seq_name, max_frames, target_w, target_h):
    """Extract frames for one sequence. Returns (n_views_ok, n_views_skipped, n_views_failed)."""
    hd_dir = os.path.join(raw_root, seq_name, "hdVideos")
    if not os.path.isdir(hd_dir):
        print(f"  [SKIP] No hdVideos/ directory")
        return 0, 0, 0

    ims_dir = os.path.join(output_root, seq_name, "ims")
    os.makedirs(ims_dir, exist_ok=True)

    mp4_files = sorted([f for f in os.listdir(hd_dir) if f.endswith(".mp4")])
    if not mp4_files:
        print(f"  [SKIP] No MP4 files in hdVideos/")
        return 0, 0, 0

    n_ok = 0
    n_skipped = 0
    n_failed = 0

    for mp4 in mp4_files:
        # Parse node number from filename: hd_00_XX.mp4
        parts = mp4.replace(".mp4", "").split("_")
        if len(parts) != 3:
            print(f"  [WARN] Unexpected filename: {mp4}, skipping")
            continue
        node = int(parts[2])

        mp4_path = os.path.join(hd_dir, mp4)
        view_dir = os.path.join(ims_dir, str(node))
        os.makedirs(view_dir, exist_ok=True)

        # Check if already extracted
        existing_frames = [f for f in os.listdir(view_dir) if f.endswith(".jpg")]
        if len(existing_frames) >= max_frames:
            n_ok += 1
            n_skipped += 1
            continue

        # Extract with ffmpeg
        # -start_number 0 for 0-indexed output
        # -vf scale for resizing
        # -frames:v for frame limit
        # -qscale:v 2 for high quality JPEG
        cmd = [
            "ffmpeg", "-y",
            "-i", mp4_path,
            "-vf", f"scale={target_w}:{target_h}",
            "-qscale:v", "2",
            "-frames:v", str(max_frames),
            "-start_number", "0",
            os.path.join(view_dir, "%06d.jpg"),
        ]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=300,  # 5 min per video
            )
            # Count extracted frames
            extracted = len([f for f in os.listdir(view_dir) if f.endswith(".jpg")])
            if result.returncode == 0 and extracted > 0:
                n_ok += 1
                print(f"  [OK] view {node}: {extracted} frames")
            else:
                n_failed += 1
                stderr_tail = result.stderr.decode()[-200:] if result.stderr else ""
                print(f"  [FAIL] view {node}: ffmpeg returned {result.returncode}. {stderr_tail}")
        except subprocess.TimeoutExpired:
            n_failed += 1
            print(f"  [FAIL] view {node}: ffmpeg timeout")
        except Exception as e:
            n_failed += 1
            print(f"  [FAIL] view {node}: {e}")

    return n_ok, n_skipped, n_failed


def main():
    parser = argparse.ArgumentParser(description="Extract frames from Panoptic HD videos")
    parser.add_argument("--raw_root", default=os.path.join(
        os.environ.get("SCRATCH", "/cluster/scratch/tsmail"), "datasets", "panoptic-raw"))
    parser.add_argument("--output_root", default=os.path.join(
        os.environ.get("SCRATCH", "/cluster/scratch/tsmail"), "datasets", "panoptic-multiview"))
    parser.add_argument("--seq_index", type=int, default=None,
                        help="Process only this sequence index. Reads SLURM_ARRAY_TASK_ID if not set.")
    parser.add_argument("--max_frames", type=int, default=150)
    parser.add_argument("--target_width", type=int, default=640)
    parser.add_argument("--target_height", type=int, default=360)
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

    print(f"=== Extracting frames for {len(sequences)} sequence(s) ===")
    print(f"Raw root:    {args.raw_root}")
    print(f"Output root: {args.output_root}")
    print(f"Max frames:  {args.max_frames}")
    print(f"Resolution:  {args.target_width}x{args.target_height}")
    print()

    for seq in sequences:
        print(f"[{seq}] Extracting frames...")
        n_ok, n_skipped, n_failed = extract_sequence(
            args.raw_root, args.output_root, seq,
            args.max_frames, args.target_width, args.target_height)
        total = n_ok + n_failed
        status = "DONE" if n_failed == 0 else "PARTIAL"
        print(f"[{seq}] {status}: {n_ok}/{total} views ({n_skipped} skipped, {n_failed} failed)")
        print()


if __name__ == "__main__":
    main()
