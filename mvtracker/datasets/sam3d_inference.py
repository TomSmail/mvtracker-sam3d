"""
On-the-fly SAM3D inference for multi-view datasets.

This module provides utilities to run SAM3D Body inference during evaluation
to compute human body fits that guide MVTracker's kNN search.
"""

import logging
import os
import sys
from typing import Optional, Tuple

import numpy as np
import torch

# Add SAM3D to path
SAM3D_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "sam-3d-body")
if os.path.exists(SAM3D_PATH) and SAM3D_PATH not in sys.path:
    sys.path.insert(0, SAM3D_PATH)


class SAM3DInferenceWrapper:
    """
    Wrapper for running SAM3D Body inference on multi-view datasets.

    Handles:
    - Lazy model initialization
    - Per-view, per-frame inference
    - Camera-space to world-space transformation
    - Multi-person handling
    """

    _instance = None  # Singleton pattern to avoid reloading model

    def __init__(
        self,
        checkpoint_path: str,
        mhr_path: str,
        device: str = "cuda",
        detector_name: str = "vitdet",
        detector_path: str = "",
    ):
        """
        Initialize SAM3D inference wrapper.

        Args:
            checkpoint_path: Path to SAM3D checkpoint
            mhr_path: Path to MHR model
            device: Device to run inference on
            detector_name: Name of human detector
            detector_path: Path to detector checkpoint
        """
        self.checkpoint_path = checkpoint_path
        self.mhr_path = mhr_path
        self.device = device
        self.detector_name = detector_name
        self.detector_path = detector_path

        self.estimator = None
        self.model = None
        self.cfg = None

    def _lazy_init(self):
        """Lazy initialization of SAM3D model."""
        if self.estimator is not None:
            return

        logging.info(f"Initializing SAM3D model from {self.checkpoint_path}")

        try:
            from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
        except ImportError as e:
            raise ImportError(
                f"Failed to import SAM3D: {e}\n"
                "Make sure sam-3d-body is installed and in Python path."
            )

        # Load SAM3D model
        self.model, self.cfg = load_sam_3d_body(
            checkpoint_path=self.checkpoint_path,
            device=self.device,
            mhr_path=self.mhr_path,
        )

        # Load detector (optional, uses default if not specified)
        detector = None
        if self.detector_name:
            try:
                from tools.build_detector import HumanDetector
                detector = HumanDetector(
                    name=self.detector_name,
                    device=self.device,
                    path=self.detector_path,
                )
            except Exception as e:
                logging.warning(f"Failed to load detector {self.detector_name}: {e}")
                logging.warning("Continuing with SAM3D built-in detector")

        # Create estimator (segmentor and fov_estimator are optional)
        self.estimator = SAM3DBodyEstimator(
            sam_3d_body_model=self.model,
            model_cfg=self.cfg,
            human_detector=detector,
            human_segmentor=None,  # Optional, will use default if needed
            fov_estimator=None,    # Optional, uses default FOV
        )

        logging.info("SAM3D model initialized successfully")

    @torch.no_grad()
    def process_multiview_sequence(
        self,
        rgbs: torch.Tensor,
        intrs: torch.Tensor,
        extrs: torch.Tensor,
        inference_type: str = "body",
        bbox_thr: float = 0.5,
    ) -> Optional[torch.Tensor]:
        """
        Run SAM3D on a multi-view sequence and convert to world coordinates.

        Args:
            rgbs: RGB images (V, T, 3, H, W) - torch tensor, 0-1 range
            intrs: Intrinsics (V, T, 3, 3)
            extrs: Extrinsics (V, T, 3, 4) - camera-to-world [R|t]
            inference_type: "body" or "full"
            bbox_thr: Detection confidence threshold

        Returns:
            joints_3d_world: (T, n_persons, 70, 3) in world coordinates, or None if no detections
        """
        self._lazy_init()

        n_views, n_frames = rgbs.shape[:2]
        device = rgbs.device

        # Storage for per-view detections
        all_view_detections = []

        for view_idx in range(n_views):
            view_detections = []

            for frame_idx in range(n_frames):
                # Extract frame
                rgb_frame = rgbs[view_idx, frame_idx]  # (3, H, W)
                intr_frame = intrs[view_idx, frame_idx]  # (3, 3)
                extr_frame = extrs[view_idx, frame_idx]  # (3, 4)

                # Convert to numpy (H, W, 3) in 0-255 range
                rgb_np = (rgb_frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                # SAM3D expects cam_int as a tensor (calls .to() on it)
                intr_tensor = intr_frame  # Keep as tensor, don't convert to numpy

                # Run SAM3D inference
                try:
                    outputs = self.estimator.process_one_image(
                        rgb_np,
                        cam_int=intr_tensor,
                        inference_type=inference_type,
                        bbox_thr=bbox_thr,
                    )
                except Exception as e:
                    logging.warning(f"SAM3D failed on view {view_idx}, frame {frame_idx}: {e}")
                    outputs = []

                # Convert to world coordinates
                frame_detections = []
                for person_output in outputs:
                    # SAM3D outputs joints in camera coordinates
                    joints_cam = person_output["pred_keypoints_3d"]  # (70, 3)

                    # Transform to world coordinates
                    joints_world = self._camera_to_world(
                        joints_cam, extr_frame
                    )

                    frame_detections.append(joints_world)

                view_detections.append(frame_detections)

            all_view_detections.append(view_detections)

        # Aggregate detections across views
        # For each frame, merge detections from all views
        joints_3d_world = self._aggregate_multiview_detections(
            all_view_detections, n_frames, device
        )

        return joints_3d_world

    def _camera_to_world(
        self,
        joints_cam: np.ndarray,
        extr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Transform joints from camera space to world space.

        Args:
            joints_cam: (70, 3) numpy array in camera coordinates
            extr: (3, 4) camera-to-world transformation [R|t]

        Returns:
            joints_world: (70, 3) tensor in world coordinates
        """
        # Convert to tensor if needed
        if isinstance(joints_cam, np.ndarray):
            joints_cam_t = torch.from_numpy(joints_cam).float()
        else:
            joints_cam_t = joints_cam.float()

        # Extract rotation and translation
        R = extr[:3, :3]  # (3, 3)
        t = extr[:3, 3]   # (3,)

        # Apply transformation: X_world = R @ X_cam + t
        joints_world = (R @ joints_cam_t.T).T + t

        return joints_world

    def _aggregate_multiview_detections(
        self,
        all_view_detections: list,
        n_frames: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """
        Aggregate person detections across multiple views.

        Uses simple first-view priority for now. Could be improved with:
        - Hungarian matching across views
        - Confidence-based selection
        - Triangulation from multiple views

        Args:
            all_view_detections: [n_views][n_frames][n_persons] -> (70, 3) tensors
            n_frames: Number of frames
            device: Target device

        Returns:
            joints_3d_world: (T, n_persons, 70, 3) or None
        """
        n_views = len(all_view_detections)

        # Find maximum number of persons across all frames/views
        max_persons = 0
        for view_idx in range(n_views):
            for frame_idx in range(n_frames):
                max_persons = max(max_persons, len(all_view_detections[view_idx][frame_idx]))

        if max_persons == 0:
            logging.warning("No person detections found in any frame")
            return None

        # Allocate output tensor
        joints_3d_world = torch.zeros(
            (n_frames, max_persons, 70, 3),
            dtype=torch.float32,
            device=device,
        )

        # Simple strategy: use detections from the first view that has them
        for frame_idx in range(n_frames):
            for view_idx in range(n_views):
                frame_detections = all_view_detections[view_idx][frame_idx]
                if len(frame_detections) > 0:
                    # Use detections from this view
                    for person_idx, joints in enumerate(frame_detections):
                        if person_idx < max_persons:
                            # Ensure joints is a tensor and move to correct device
                            if isinstance(joints, np.ndarray):
                                joints = torch.from_numpy(joints).float()
                            joints_3d_world[frame_idx, person_idx] = joints.to(device)
                    break  # Stop after finding detections in first view

        return joints_3d_world


def get_sam3d_wrapper(
    checkpoint_path: Optional[str] = None,
    mhr_path: Optional[str] = None,
    device: str = "cuda",
) -> Optional[SAM3DInferenceWrapper]:
    """
    Get or create SAM3D inference wrapper (singleton pattern).

    Args:
        checkpoint_path: Path to SAM3D checkpoint
        mhr_path: Path to MHR model
        device: Device to run on

    Returns:
        SAM3DInferenceWrapper instance or None if paths not provided
    """
    if checkpoint_path is None or mhr_path is None:
        return None

    if SAM3DInferenceWrapper._instance is None:
        SAM3DInferenceWrapper._instance = SAM3DInferenceWrapper(
            checkpoint_path=checkpoint_path,
            mhr_path=mhr_path,
            device=device,
        )

    return SAM3DInferenceWrapper._instance
