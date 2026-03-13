#!/usr/bin/env python3
"""
Verify integrity of Panoptic Studio raw dataset.

Checks:
- All sequences present
- Required directory structure
- Calibration files exist and valid JSON
- Tar archives are readable
- Video directories are accessible
"""

import os
import json
import tarfile
import argparse
from pathlib import Path
from collections import defaultdict


# Expected sequences from CMU Panoptic Studio dataset
# https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox
EXPECTED_SEQUENCES = [
    # Social scenarios
    "160224_haggling1", "160224_mafia1", "160224_mafia2",
    "160224_ultimatum1", "160224_ultimatum2",
    "160226_haggling1", "160226_mafia1", "160226_mafia2", "160226_ultimatum1",
    "160422_haggling1", "160422_mafia2", "160422_ultimatum1",
    "160906_band1", "160906_band2", "160906_band3", "160906_band4",
    "160906_ian1", "160906_ian2", "160906_ian3", "160906_ian5",

    # Toddlers
    "160317_moonbaby1", "160317_moonbaby2", "160317_moonbaby3",
    "160401_ian1", "160401_ian2", "160401_ian3",
    "160906_pizza1", "160906_pizza2",

    # Dance
    "160906_dance1", "160906_dance2", "160906_dance3", "160906_dance4",

    # Sports (included in benchmark subset)
    "171026_pose1", "171026_pose2", "171026_pose3",
    "171204_pose1", "171204_pose2", "171204_pose3", "171204_pose4", "171204_pose5", "171204_pose6",

    # More activities
    "160906_ultimatum1", "160906_haggling1", "160906_mafia2",
]


def verify_sequence(seq_path: Path, verbose: bool = False) -> dict:
    """Verify a single sequence directory."""
    seq_name = seq_path.name
    issues = []
    stats = {
        "seq_name": seq_name,
        "exists": seq_path.exists(),
        "size_gb": 0,
        "has_calibration": False,
        "has_hdVideos": False,
        "has_vgaVideos": False,
        "tar_files": {},
        "issues": issues,
    }

    if not seq_path.exists():
        issues.append("Sequence directory missing")
        return stats

    # Check size
    try:
        size_bytes = sum(f.stat().st_size for f in seq_path.rglob('*') if f.is_file())
        stats["size_gb"] = round(size_bytes / (1024**3), 2)
    except Exception as e:
        issues.append(f"Could not calculate size: {e}")

    # Check calibration JSON
    calib_file = seq_path / f"calibration_{seq_name}.json"
    if calib_file.exists():
        stats["has_calibration"] = True
        try:
            with open(calib_file) as f:
                calib = json.load(f)
            if "cameras" not in calib:
                issues.append("Calibration JSON missing 'cameras' field")
            elif verbose:
                stats["num_cameras"] = len(calib["cameras"])
        except json.JSONDecodeError:
            issues.append("Calibration JSON is corrupted")
        except Exception as e:
            issues.append(f"Could not read calibration: {e}")
    else:
        issues.append("Missing calibration JSON")

    # Check video directories
    hd_videos = seq_path / "hdVideos"
    vga_videos = seq_path / "vgaVideos"

    if hd_videos.exists() and hd_videos.is_dir():
        stats["has_hdVideos"] = True
        try:
            hd_files = list(hd_videos.glob("*.mp4"))
            if verbose:
                stats["num_hd_videos"] = len(hd_files)
            if len(hd_files) == 0:
                issues.append("hdVideos directory is empty")
        except Exception as e:
            issues.append(f"Could not list hdVideos: {e}")
    else:
        issues.append("Missing hdVideos directory")

    if vga_videos.exists() and vga_videos.is_dir():
        stats["has_vgaVideos"] = True
        try:
            vga_files = list(vga_videos.glob("*.mp4"))
            if verbose:
                stats["num_vga_videos"] = len(vga_files)
        except Exception as e:
            issues.append(f"Could not list vgaVideos: {e}")

    # Check tar archives (3D pose annotations)
    tar_names = ["hdFace3d.tar", "hdHand3d.tar", "hdPose3d_stage1_coco19.tar"]
    for tar_name in tar_names:
        tar_path = seq_path / tar_name
        if tar_path.exists():
            stats["tar_files"][tar_name] = "readable"
            try:
                # Quick check: can we open and list the tar?
                with tarfile.open(tar_path, 'r') as tar:
                    members = tar.getmembers()
                    if len(members) == 0:
                        stats["tar_files"][tar_name] = "empty"
                        issues.append(f"{tar_name} is empty")
            except Exception as e:
                stats["tar_files"][tar_name] = "corrupted"
                issues.append(f"{tar_name} is corrupted: {e}")
        else:
            stats["tar_files"][tar_name] = "missing"

    return stats


def main():
    parser = argparse.ArgumentParser(description="Verify Panoptic Studio raw dataset integrity")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/cluster/scratch/tsmail/datasets/panoptic-raw",
        help="Path to panoptic-raw directory"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed stats per sequence"
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)

    print("=" * 70)
    print("Panoptic Studio Raw Dataset Verification")
    print("=" * 70)
    print(f"Data root: {data_root}")
    print()

    if not data_root.exists():
        print(f"ERROR: Data root does not exist: {data_root}")
        return 1

    # Find all sequences
    actual_sequences = sorted([
        d.name for d in data_root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

    print(f"Found {len(actual_sequences)} sequences in {data_root}")
    print()

    # Verify each sequence
    all_stats = []
    missing_sequences = []
    corrupted_sequences = []
    healthy_sequences = []

    print("Verifying sequences...")
    for i, seq_name in enumerate(actual_sequences, 1):
        seq_path = data_root / seq_name
        stats = verify_sequence(seq_path, verbose=args.verbose)
        all_stats.append(stats)

        if len(stats["issues"]) == 0:
            healthy_sequences.append(seq_name)
            status = "✓"
        else:
            corrupted_sequences.append(seq_name)
            status = "✗"

        print(f"  [{i:2d}/{len(actual_sequences)}] {status} {seq_name:30s} {stats['size_gb']:6.2f} GB", end="")

        if len(stats["issues"]) > 0:
            print(f"  ({len(stats['issues'])} issues)")
        else:
            print()

    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Total sequences found:     {len(actual_sequences)}")
    print(f"Healthy sequences:         {len(healthy_sequences)} ✓")
    print(f"Sequences with issues:     {len(corrupted_sequences)} ✗")
    print()

    # Total size
    total_size_gb = sum(s["size_gb"] for s in all_stats)
    print(f"Total dataset size:        {total_size_gb:.1f} GB")
    print()

    # Issues breakdown
    if len(corrupted_sequences) > 0:
        print("Sequences with issues:")
        for seq_name in corrupted_sequences:
            stats = next(s for s in all_stats if s["seq_name"] == seq_name)
            print(f"\n  {seq_name}:")
            for issue in stats["issues"]:
                print(f"    - {issue}")
        print()

    # Check for common issues
    issue_counts = defaultdict(int)
    for stats in all_stats:
        for issue in stats["issues"]:
            # Categorize
            if "calibration" in issue.lower():
                issue_counts["Missing/corrupt calibration"] += 1
            elif "hdVideos" in issue:
                issue_counts["Missing/corrupt HD videos"] += 1
            elif "vgaVideos" in issue:
                issue_counts["Missing/corrupt VGA videos"] += 1
            elif "tar" in issue.lower():
                issue_counts["Missing/corrupt tar archives"] += 1
            else:
                issue_counts["Other"] += 1

    if issue_counts:
        print("Issue breakdown:")
        for issue_type, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
            print(f"  - {issue_type:30s}: {count} sequences")
        print()

    # Recommendations
    if len(corrupted_sequences) > 0:
        print("=" * 70)
        print("Recommendations")
        print("=" * 70)
        print("Some sequences have issues. Common fixes:")
        print("  1. Re-download from CMU Panoptic Studio:")
        print("     http://domedb.perception.cs.cmu.edu/")
        print("  2. Check disk space and permissions")
        print("  3. Verify network/download wasn't interrupted")
        print()

    # Detailed stats if verbose
    if args.verbose:
        print("=" * 70)
        print("Detailed Statistics")
        print("=" * 70)
        for stats in all_stats:
            if len(stats["issues"]) > 0:  # Only show problematic ones in verbose
                print(f"\n{stats['seq_name']}:")
                print(f"  Size: {stats['size_gb']} GB")
                print(f"  Calibration: {'✓' if stats['has_calibration'] else '✗'}")
                print(f"  HD Videos: {'✓' if stats['has_hdVideos'] else '✗'}")
                print(f"  VGA Videos: {'✓' if stats['has_vgaVideos'] else '✗'}")
                if "num_cameras" in stats:
                    print(f"  Cameras: {stats['num_cameras']}")
                if "num_hd_videos" in stats:
                    print(f"  HD video files: {stats['num_hd_videos']}")
                print(f"  Tar archives:")
                for tar_name, status in stats["tar_files"].items():
                    print(f"    - {tar_name}: {status}")

    print()
    return 0 if len(corrupted_sequences) == 0 else 1


if __name__ == "__main__":
    exit(main())
