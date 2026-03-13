#!/usr/bin/env python3
"""
Download missing HD videos from CMU Panoptic Studio dataset.

CMU Panoptic Studio: http://domedb.perception.cs.cmu.edu/
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import List, Tuple


# CMU Panoptic Studio base URLs
CMU_BASE_URL = "http://domedb.perception.cs.cmu.edu"


def get_sequences_needing_videos(data_root: Path) -> List[Tuple[str, Path]]:
    """Find sequences with empty hdVideos directories."""
    sequences = []

    for seq_dir in sorted(data_root.iterdir()):
        if not seq_dir.is_dir() or seq_dir.name.startswith('.'):
            continue

        hd_videos_dir = seq_dir / "hdVideos"
        if not hd_videos_dir.exists():
            continue

        # Check if directory is empty or has very few files
        try:
            video_files = list(hd_videos_dir.glob("*.mp4"))
            if len(video_files) < 10:  # Most sequences have 30+ HD cameras
                sequences.append((seq_dir.name, seq_dir))
        except Exception as e:
            print(f"Warning: Could not check {seq_dir.name}: {e}")

    return sequences


def download_sequence_videos(seq_name: str, seq_path: Path, dry_run: bool = False) -> bool:
    """
    Download HD videos for a single sequence using wget.

    CMU Panoptic provides videos at:
    http://domedb.perception.cs.cmu.edu/dataset/{seq_name}/hdVideos/
    """

    hd_videos_dir = seq_path / "hdVideos"
    hd_videos_dir.mkdir(exist_ok=True)

    # CMU Panoptic video URL pattern
    base_url = f"{CMU_BASE_URL}/dataset/{seq_name}/hdVideos/"

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Downloading videos for: {seq_name}")
    print(f"  URL: {base_url}")
    print(f"  Destination: {hd_videos_dir}")

    if dry_run:
        return True

    # Use wget to download with:
    # -r: recursive
    # -np: no parent directories
    # -nH: no host directories
    # -nc: no clobber (skip existing files)
    # -c: continue partial downloads
    # --cut-dirs=3: strip dataset/{seq}/hdVideos from path
    # -A: accept only .mp4 files
    # -P: output directory
    cmd = [
        "wget",
        "-r",  # recursive
        "-np",  # no parent
        "-nH",  # no host directories
        "-nc",  # skip existing (resumable)
        "-c",  # continue partial downloads
        "--cut-dirs=3",  # strip first 3 directory levels
        "-A", "*.mp4",  # accept only mp4 files
        "-P", str(seq_path),  # save to sequence directory
        "--progress=bar:force",  # show progress
        "--show-progress",
        base_url
    ]

    print(f"  Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True)
        print(f"  ✓ Download complete for {seq_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Download failed for {seq_name}: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n  Interrupted. Partial downloads can be resumed.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download missing HD videos from CMU Panoptic Studio"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/cluster/scratch/tsmail/datasets/panoptic-raw",
        help="Path to panoptic-raw directory"
    )
    parser.add_argument(
        "--seq",
        type=str,
        help="Download specific sequence only"
    )
    parser.add_argument(
        "--max_sequences",
        type=int,
        default=None,
        help="Maximum number of sequences to download (for testing)"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be downloaded without actually downloading"
    )
    parser.add_argument(
        "--seq_index",
        type=int,
        help="Process only this sequence index (for array jobs)"
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)

    if not data_root.exists():
        print(f"Error: Data root does not exist: {data_root}")
        return 1

    print("=" * 70)
    print("CMU Panoptic Studio Video Downloader")
    print("=" * 70)
    print(f"Data root: {data_root}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'DOWNLOAD'}")
    print()

    # Find sequences needing videos
    if args.seq:
        sequences = [(args.seq, data_root / args.seq)]
    else:
        sequences = get_sequences_needing_videos(data_root)

    print(f"Found {len(sequences)} sequences needing videos")

    if args.seq_index is not None:
        if args.seq_index >= len(sequences):
            print(f"Error: Sequence index {args.seq_index} out of range (max {len(sequences) - 1})")
            return 1
        sequences = [sequences[args.seq_index]]
        print(f"Processing sequence index {args.seq_index}: {sequences[0][0]}")

    if args.max_sequences:
        sequences = sequences[:args.max_sequences]
        print(f"Limited to first {args.max_sequences} sequences")

    print()

    # Estimate download size
    if not args.dry_run:
        est_size_gb = len(sequences) * 3.0  # Rough estimate: 3GB per sequence
        print(f"Estimated download size: ~{est_size_gb:.0f} GB")
        print(f"This will take several hours to days depending on network speed.")
        print()

        response = input("Continue with download? [y/N] ")
        if response.lower() != 'y':
            print("Aborted.")
            return 0
        print()

    # Download sequences
    successful = 0
    failed = 0

    for i, (seq_name, seq_path) in enumerate(sequences, 1):
        print(f"\n[{i}/{len(sequences)}] {seq_name}")

        success = download_sequence_videos(seq_name, seq_path, dry_run=args.dry_run)

        if success:
            successful += 1
        else:
            failed += 1

    print()
    print("=" * 70)
    print("Download Summary")
    print("=" * 70)
    print(f"Total sequences: {len(sequences)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print()

    if not args.dry_run:
        print("Next steps:")
        print("  1. Verify downloads with: python scripts/data_engine/verify_panoptic_raw.py")
        print("  2. Preprocess sequences: python scripts/panoptic_studio_preprocessing.py")
        print("  3. Run SAM3D inference: bash scripts/data_engine/submit_slurm_sam3d.sh")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
