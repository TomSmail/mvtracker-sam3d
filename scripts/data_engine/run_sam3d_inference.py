"""
SAM3D Body inference on Panoptic Studio sequences.

For each sequence, for each camera view, for each frame:
  1. Load the RGB image
  2. Detect people and run SAM3D Body inference
  3. Save per-frame predictions (vertices, keypoints, camera params) as .npz

Usage:
    python scripts/data_engine/run_sam3d_inference.py \
        --data_root /cluster/scratch/tsmail/datasets/panoptic-multiview \
        --checkpoint_path ~/checkpoints/sam-3d-body-dinov3/model.ckpt \
        --mhr_path ~/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt

    # Process a single sequence (for SLURM array jobs):
    python scripts/data_engine/run_sam3d_inference.py \
        --data_root ... --seq_index 0
"""

import argparse
import os
import sys
import warnings

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "sam-3d-body"))

from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from sam_3d_body.data.transforms import (
    Compose,
    GetBBoxCenterScale,
    TopdownAffine,
    VisionTransformWrapper,
)
from torchvision.transforms import ToTensor


def get_sequences(data_root):
    """Return sorted list of valid sequence directories."""
    seqs = [
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
        and not d.startswith(".")
        and not d.startswith("_")
    ]
    return sorted(seqs)


def get_view_folders(seq_path):
    """Return sorted list of view folder indices."""
    ims_path = os.path.join(seq_path, "ims")
    if not os.path.isdir(ims_path):
        return []
    return sorted(os.listdir(ims_path), key=lambda x: int(x))


def load_intrinsics(seq_path):
    """Load camera intrinsics from the annotations file."""
    ann_file = os.path.join(seq_path, "tapvid3d_annotations.npz")
    if not os.path.exists(ann_file):
        return None
    ann = np.load(ann_file)
    return ann["intrinsics"]  # [n_views, 3, 3]


def process_sequence(estimator, seq_path, output_dir, skip_existing=True,
                     inference_type="body", device="cuda"):
    """
    Run SAM3D inference on all frames of all views in a sequence.

    Saves per-view, per-frame predictions as compressed npz.
    """
    view_folders = get_view_folders(seq_path)
    if not view_folders:
        warnings.warn(f"No view folders found in {seq_path}/ims/")
        return

    intrinsics = load_intrinsics(seq_path)
    os.makedirs(output_dir, exist_ok=True)

    for view_idx_str in view_folders:
        view_idx = int(view_idx_str)
        view_output_dir = os.path.join(output_dir, f"view_{view_idx:02d}")
        os.makedirs(view_output_dir, exist_ok=True)

        rgb_folder = os.path.join(seq_path, "ims", view_idx_str)
        rgb_files = sorted(os.listdir(rgb_folder))

        cam_int = None
        if intrinsics is not None and view_idx < len(intrinsics):
            cam_int = torch.from_numpy(intrinsics[view_idx]).float().unsqueeze(0).to(device)

        for frame_idx, rgb_file in enumerate(rgb_files):
            out_file = os.path.join(view_output_dir, f"frame_{frame_idx:05d}.npz")
            if skip_existing and os.path.exists(out_file):
                continue

            img_path = os.path.join(rgb_folder, rgb_file)
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                warnings.warn(f"Failed to read image: {img_path}")
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            try:
                outputs = estimator.process_one_image(
                    img_rgb,
                    cam_int=cam_int,
                    inference_type=inference_type,
                    bbox_thr=0.5,
                )
            except Exception as e:
                warnings.warn(f"SAM3D failed on {img_path}: {e}")
                np.savez_compressed(out_file, n_persons=0)
                continue

            if len(outputs) == 0:
                np.savez_compressed(out_file, n_persons=0)
                continue

            save_dict = {"n_persons": len(outputs)}
            for person_idx, out in enumerate(outputs):
                prefix = f"person_{person_idx}"
                save_dict[f"{prefix}_pred_keypoints_3d"] = out["pred_keypoints_3d"]
                save_dict[f"{prefix}_pred_vertices"] = out["pred_vertices"]
                save_dict[f"{prefix}_pred_cam_t"] = out["pred_cam_t"]
                save_dict[f"{prefix}_bbox"] = out["bbox"]
                save_dict[f"{prefix}_pred_keypoints_2d"] = out["pred_keypoints_2d"]
                if "focal_length" in out:
                    save_dict[f"{prefix}_focal_length"] = out["focal_length"]

            np.savez_compressed(out_file, **save_dict)

        print(f"  View {view_idx:2d}: {len(rgb_files)} frames processed -> {view_output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run SAM3D Body inference on Panoptic Studio sequences")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory of panoptic-multiview dataset")
    parser.add_argument("--checkpoint_path", type=str,
                        default=os.path.expanduser("~/checkpoints/sam-3d-body-dinov3/model.ckpt"))
    parser.add_argument("--mhr_path", type=str,
                        default=os.path.expanduser("~/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"))
    parser.add_argument("--detector_name", type=str, default="vitdet")
    parser.add_argument("--detector_path", type=str, default="",
                        help="Path to directory containing detector checkpoint (avoids download)")
    parser.add_argument("--inference_type", type=str, default="body",
                        choices=["full", "body", "hand"],
                        help="SAM3D inference type (body is faster, full includes hand refinement)")
    parser.add_argument("--seq_index", type=int, default=None,
                        help="Process only this sequence index (for SLURM array jobs)")
    parser.add_argument("--skip_existing", action="store_true", default=True,
                        help="Skip frames that already have predictions")
    parser.add_argument("--no_skip_existing", action="store_true", default=False)
    parser.add_argument("--bbox_thr", type=float, default=0.3)
    args = parser.parse_args()

    skip_existing = args.skip_existing and not args.no_skip_existing

    # Handle SLURM array task ID
    seq_index = args.seq_index
    if seq_index is None and "SLURM_ARRAY_TASK_ID" in os.environ:
        seq_index = int(os.environ["SLURM_ARRAY_TASK_ID"])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=== SAM3D Body Inference on Panoptic Studio ===")
    print(f"Data root:       {args.data_root}")
    print(f"Checkpoint:      {args.checkpoint_path}")
    print(f"Inference type:  {args.inference_type}")
    print(f"Device:          {device}")
    print(f"Skip existing:   {skip_existing}")

    model, model_cfg = load_sam_3d_body(
        args.checkpoint_path, device=device, mhr_path=args.mhr_path
    )

    human_detector = None
    if args.detector_name:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "sam-3d-body"))
        from tools.build_detector import HumanDetector
        human_detector = HumanDetector(name=args.detector_name, device=device,
                                       path=args.detector_path)

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
    )

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
        output_dir = os.path.join(seq_path, "sam3d_predictions")
        print(f"\nSequence: {seq_name}")
        process_sequence(
            estimator, seq_path, output_dir,
            skip_existing=skip_existing,
            inference_type=args.inference_type,
            device=device,
        )

    print("\n=== Inference complete ===")


if __name__ == "__main__":
    main()
