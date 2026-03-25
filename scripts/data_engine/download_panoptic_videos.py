"""
Download HD MP4 videos for Panoptic Studio sequences from the CMU server.

Supports resumable downloads via wget -c, and skips already-downloaded files.
Designed to be called per-sequence from a SLURM array job.

Usage:
    python scripts/data_engine/download_panoptic_videos.py --seq_index 0
    python scripts/data_engine/download_panoptic_videos.py  # all sequences
"""

import argparse
import os
import subprocess
import sys


ENDPOINT = "http://domedb.perception.cs.cmu.edu"
NUM_HD_NODES = 31


def get_sequences(raw_root):
    """List sequence directories sorted alphabetically."""
    return sorted([
        d for d in os.listdir(raw_root)
        if os.path.isdir(os.path.join(raw_root, d))
        and not d.startswith(".")
        and not d.startswith("_")
    ])


def download_sequence(raw_root, seq_name, endpoint, num_nodes, retries=3):
    """Download HD videos for one sequence. Returns (n_ok, n_total)."""
    hd_dir = os.path.join(raw_root, seq_name, "hdVideos")
    os.makedirs(hd_dir, exist_ok=True)

    n_ok = 0
    n_skipped = 0
    n_failed = 0

    for node in range(num_nodes):
        filename = f"hd_00_{node:02d}.mp4"
        target = os.path.join(hd_dir, filename)
        url = f"{endpoint}/webdata/dataset/{seq_name}/videos/hd_shared_crf20/{filename}"

        # Skip if already downloaded (non-zero size)
        if os.path.exists(target) and os.path.getsize(target) > 0:
            n_ok += 1
            n_skipped += 1
            continue

        # Download with retries
        success = False
        for attempt in range(1, retries + 1):
            try:
                result = subprocess.run(
                    ["wget", "-c", "-q", "--show-progress", "-O", target, url],
                    timeout=1800,  # 30 min per video
                )
                if result.returncode == 0 and os.path.exists(target) and os.path.getsize(target) > 0:
                    success = True
                    break
                else:
                    print(f"  [WARN] wget returned {result.returncode} for {filename} (attempt {attempt}/{retries})")
            except subprocess.TimeoutExpired:
                print(f"  [WARN] Timeout downloading {filename} (attempt {attempt}/{retries})")
            except Exception as e:
                print(f"  [WARN] Error downloading {filename}: {e} (attempt {attempt}/{retries})")

        if success:
            n_ok += 1
            print(f"  [OK] {filename}")
        else:
            n_failed += 1
            # Remove zero-byte files from failed downloads
            if os.path.exists(target) and os.path.getsize(target) == 0:
                os.remove(target)
            print(f"  [FAIL] {filename}")

    return n_ok, n_skipped, n_failed, num_nodes


def main():
    parser = argparse.ArgumentParser(description="Download Panoptic HD videos")
    parser.add_argument("--raw_root", default=os.path.join(
        os.environ.get("SCRATCH", "/cluster/scratch/tsmail"), "datasets", "panoptic-raw"))
    parser.add_argument("--seq_index", type=int, default=None,
                        help="Process only this sequence index. Reads SLURM_ARRAY_TASK_ID if not set.")
    parser.add_argument("--endpoint", default=ENDPOINT)
    parser.add_argument("--num_hd_views", type=int, default=NUM_HD_NODES)
    parser.add_argument("--retries", type=int, default=3)
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

    print(f"=== Downloading HD videos for {len(sequences)} sequence(s) ===")
    print(f"Raw root: {args.raw_root}")
    print(f"Endpoint: {args.endpoint}")
    print()

    for seq in sequences:
        print(f"[{seq}] Downloading {args.num_hd_views} HD videos...")
        n_ok, n_skipped, n_failed, n_total = download_sequence(
            args.raw_root, seq, args.endpoint, args.num_hd_views, args.retries)
        status = "DONE" if n_failed == 0 else "PARTIAL"
        print(f"[{seq}] {status}: {n_ok}/{n_total} ok ({n_skipped} skipped, {n_failed} failed)")
        print()


if __name__ == "__main__":
    main()
