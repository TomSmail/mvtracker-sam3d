"""
MVTracker dataset loader for SAM3D-generated human trajectories on Panoptic Studio.

Loads human body keypoint + vertex tracks produced by the AnthroTAP data engine
(scripts/data_engine/generate_human_tracks.py) and outputs standard Datapoints.

Follows the same conventions as PanopticStudioMultiViewDataset:
  - Same scene normalization (scale=2.5, rot_x=-90)
  - Same view selection syntax
  - Same Datapoint format

Dataset name format:
  panoptic-human-views1_7_14_20
  panoptic-human-views1_7_14_20-novelviews24
  panoptic-human-views1_7_14_20-single
  panoptic-human-views1_7_14_20-2dpt
  panoptic-human-views1_7_14_20-cached
"""

import logging
import os
import re
import time
import warnings

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset

from mvtracker.datasets.utils import Datapoint, transform_scene


class PanopticHumanTrajectoryDataset(Dataset):
    @staticmethod
    def from_name(dataset_name: str, dataset_root: str):
        """
        Factory method supporting names like:
        - panoptic-human-views1_7_14_20
        - panoptic-human-views27_16_14_8-novelviews1_4
        - panoptic-human-views1_7_14_20-single
        - panoptic-human-views1_7_14_20-2dpt
        - panoptic-human-views1_7_14_20-cached
        """
        non_parsed = dataset_name.replace("panoptic-human", "", 1)

        if non_parsed.startswith("-views"):
            match = re.match(r"-views((?:\d+_?)+)", non_parsed)
            assert match is not None, f"Failed to parse views from: {non_parsed}"
            views = list(map(int, match.group(1).split("_")))
            non_parsed = non_parsed.replace(match.group(0), "", 1)
        else:
            views = None

        if non_parsed.startswith("-novelviews"):
            match = re.match(r"-novelviews((?:\d+_?)+)", non_parsed)
            assert match is not None
            novel_views = list(map(int, match.group(1).split("_")))
            non_parsed = non_parsed.replace(match.group(0), "", 1)
        else:
            novel_views = None

        single_point = False
        if non_parsed.startswith("-single"):
            single_point = True
            non_parsed = non_parsed.replace("-single", "", 1)

        eval_2dpt = False
        if non_parsed.startswith("-2dpt"):
            eval_2dpt = True
            non_parsed = non_parsed.replace("-2dpt", "", 1)

        use_cached_tracks = False
        if non_parsed.startswith("-cached"):
            use_cached_tracks = True
            non_parsed = non_parsed.replace("-cached", "", 1)

        assert non_parsed == "", f"Unparsed part of the dataset name: {non_parsed}"

        return PanopticHumanTrajectoryDataset(
            data_root=os.path.join(dataset_root, "panoptic-multiview"),
            views_to_return=views,
            novel_views=novel_views,
            traj_per_sample=384,
            seed=72,
            max_videos=6,
            use_cached_tracks=use_cached_tracks,
            crop_size=None,
            seq_len=None,
        )

    def __init__(
        self,
        data_root,
        views_to_return=None,
        novel_views=None,
        traj_per_sample=512,
        seed=None,
        max_videos=None,
        use_cached_tracks=False,
        crop_size=None,
        seq_len=None,
    ):
        super().__init__()
        self.data_root = data_root
        self.views_to_return = views_to_return
        self.novel_views = novel_views
        self.traj_per_sample = traj_per_sample
        self.seed = seed
        self.use_cached_tracks = use_cached_tracks
        self.crop_size = crop_size  # (H, W) tuple or None
        self.seq_len = seq_len  # max frames to use, or None for all
        self.cache_name = self._cache_key()
        self.seq_names = self._get_sequence_names(max_videos)
        self.getitem_calls = 0

    def _get_sequence_names(self, max_videos):
        seq_names = [
            fname
            for fname in os.listdir(self.data_root)
            if os.path.isdir(os.path.join(self.data_root, fname))
            and not fname.startswith(".")
            and not fname.startswith("_")
        ]
        seq_names = sorted(seq_names)
        valid_seqs = []

        for seq_name in seq_names:
            scene_path = os.path.join(self.data_root, seq_name)
            tracks_file = os.path.join(scene_path, "human_tracks.npz")
            if not os.path.exists(tracks_file):
                warnings.warn(f"Skipping {scene_path}: no human_tracks.npz")
                continue
            valid_seqs.append(seq_name)

        if max_videos is not None:
            valid_seqs = valid_seqs[:max_videos]

        print(f"PanopticHumanTrajectory: using {len(valid_seqs)} videos from {self.data_root}")
        return valid_seqs

    def _cache_key(self):
        name = f"human_cachedtracks--seed{self.seed}"
        if self.views_to_return is not None:
            name += f"-views{'_'.join(map(str, self.views_to_return))}"
        if self.traj_per_sample is not None:
            name += f"-n{self.traj_per_sample}"
        return name + "--v1"

    def __len__(self):
        return len(self.seq_names)

    def __getitem__(self, index):
        start_time = time.time()
        for attempt in range(len(self.seq_names)):
            try:
                sample = self._getitem_helper((index + attempt) % len(self.seq_names))
                break
            except Exception as e:
                if attempt == 0:
                    import warnings
                    warnings.warn(f"Skipping seq index {index} ({self.seq_names[index]}): {e}")
        self.getitem_calls += 1
        if self.getitem_calls < 10:
            print(f"Loading {index:>06d} took {time.time() - start_time:.3f} sec. "
                  f"Getitem calls: {self.getitem_calls}")
        return sample, True

    def _getitem_helper(self, index):
        if self.seed is None:
            seed = torch.randint(0, 2 ** 32 - 1, (1,)).item()
        else:
            seed = self.seed
        rnd_torch = torch.Generator().manual_seed(seed)

        datapoint_path = os.path.join(self.data_root, self.seq_names[index])
        ims_path = os.path.join(datapoint_path, "ims")
        depths_path = os.path.join(datapoint_path, "dynamic3dgs_depth")

        # Load human tracks
        tracks = np.load(os.path.join(datapoint_path, "human_tracks.npz"))
        traj3d_world = tracks["trajectories"]         # [T, N, 3]
        traj2d = tracks["trajectories_pixelspace"]    # [V, T, N, 2]
        visibility = tracks["per_view_visibilities"]  # [V, T, N]
        track_valid = tracks["track_valid"]           # [T, N]
        extrs = tracks["extrinsics"]                  # [V, 3, 4] or [V, 4, 4]
        intrs = tracks["intrinsics"]                  # [V, 3, 3]

        # Load RGB images
        view_folders = sorted([f for f in os.listdir(ims_path)], key=lambda x: int(x))
        if self.views_to_return is not None:
            views_to_return = self.views_to_return
        else:
            views_to_return = sorted(list(range(len(view_folders))))
        views_to_load = views_to_return.copy()
        if self.novel_views is not None:
            views_to_load = list(set(views_to_load + self.novel_views))

        views = {}
        for v in views_to_load:
            rgb_folder = os.path.join(ims_path, str(v))
            rgb_files = sorted(os.listdir(rgb_folder))
            rgb_images = []
            for f in rgb_files:
                img = cv2.imread(os.path.join(rgb_folder, f))
                if img is None:
                    raise RuntimeError(f"Failed to read image: {os.path.join(rgb_folder, f)}")
                rgb_images.append(img[:, :, ::-1])
            depth_file = os.path.join(depths_path, f"depths_{v:02d}.npy")
            depth = np.load(depth_file) if os.path.exists(depth_file) else None
            views[v] = {"rgb": np.stack(rgb_images), "depth": depth}

        rgbs = np.stack([views[v]["rgb"] for v in views_to_return])
        n_views, n_frames_total, h, w, _ = rgbs.shape

        # Load depths (from dynamic3dgs or zeros)
        depth_list = []
        for v in views_to_return:
            if views[v]["depth"] is not None:
                depth_list.append(views[v]["depth"])
            else:
                depth_list.append(np.zeros((n_frames_total, h, w)))
        depths = np.stack(depth_list)[..., None].astype(np.float32)

        # Frame subsampling: pick a random contiguous window of seq_len frames
        if self.seq_len is not None and self.seq_len < n_frames_total:
            max_start = n_frames_total - self.seq_len
            t0 = torch.randint(0, max_start + 1, (1,), generator=rnd_torch).item()
            t1 = t0 + self.seq_len
            frame_inds = slice(t0, t1)
            rgbs = rgbs[:, frame_inds]
            depths = depths[:, frame_inds]
            traj3d_world = traj3d_world[frame_inds]
            traj2d = traj2d[:, frame_inds]
            visibility = visibility[:, frame_inds]
            track_valid = track_valid[frame_inds]
            if self.novel_views is not None:
                for v in self.novel_views:
                    views[v]["rgb"] = views[v]["rgb"][frame_inds]
        n_frames = rgbs.shape[1]

        # Image resizing
        if self.crop_size is not None:
            target_h, target_w = self.crop_size
            scale_h = target_h / h
            scale_w = target_w / w
            # Resize rgbs: (V, T, H, W, 3) -> (V, T, target_h, target_w, 3)
            rgbs_resized = np.zeros((n_views, n_frames, target_h, target_w, 3), dtype=rgbs.dtype)
            depths_resized = np.zeros((n_views, n_frames, target_h, target_w, 1), dtype=depths.dtype)
            for vi in range(n_views):
                for ti in range(n_frames):
                    rgbs_resized[vi, ti] = cv2.resize(rgbs[vi, ti], (target_w, target_h))
                    depths_resized[vi, ti, :, :, 0] = cv2.resize(
                        depths[vi, ti, :, :, 0], (target_w, target_h), interpolation=cv2.INTER_NEAREST
                    )
            rgbs = rgbs_resized
            depths = depths_resized
            # Scale intrinsics
            intrs = intrs.copy()
            intrs[:, 0, :] *= scale_w  # fx, cx
            intrs[:, 1, :] *= scale_h  # fy, cy
            # Scale 2D trajectories
            traj2d = traj2d.copy()
            traj2d[..., 0] *= scale_w
            traj2d[..., 1] *= scale_h
            h, w = target_h, target_w

        intrs_sel = np.stack([intrs[v] for v in views_to_return])[:, None, :, :].repeat(n_frames, axis=1)

        extrs_for_views = []
        for v in views_to_return:
            extr_v = extrs[v]
            if extr_v.shape[0] == 4:
                extr_v = extr_v[:3, :]
            extrs_for_views.append(extr_v)
        extrs_sel = np.stack(extrs_for_views)[:, None, :, :].repeat(n_frames, axis=1)

        visibility_sel = visibility[views_to_return]
        traj2d_sel = traj2d[views_to_return]

        # Novel views
        novel_rgbs = None
        novel_intrs = None
        novel_extrs = None
        if self.novel_views is not None:
            novel_rgbs = np.stack([views[v]["rgb"] for v in self.novel_views])
            novel_intrs = np.stack([intrs[v] for v in self.novel_views])[:, None, :, :].repeat(n_frames, axis=1)
            novel_extrs_list = []
            for v in self.novel_views:
                extr_v = extrs[v]
                if extr_v.shape[0] == 4:
                    extr_v = extr_v[:3, :]
                novel_extrs_list.append(extr_v)
            novel_extrs = np.stack(novel_extrs_list)[:, None, :, :].repeat(n_frames, axis=1)

        # Compute camera-space z for 2D+Z trajectory
        point_3d_world = traj3d_world
        point_4d_world_homo = np.concatenate(
            [point_3d_world, np.ones_like(point_3d_world[..., :1])], axis=-1
        )
        point_3d_camera = np.einsum("ABij,BCj->ABCi", extrs_sel, point_4d_world_homo)
        traj2d_w_z_sel = np.concatenate([traj2d_sel, point_3d_camera[..., 2:]], axis=-1)

        # Convert to tensors
        rgbs = torch.from_numpy(rgbs).permute(0, 1, 4, 2, 3).float()
        depths = torch.from_numpy(depths).permute(0, 1, 4, 2, 3).float()
        intrs_t = torch.from_numpy(intrs_sel).float()
        extrs_t = torch.from_numpy(extrs_sel).float()
        traj2d_t = torch.from_numpy(traj2d_sel)
        traj2d_w_z_t = torch.from_numpy(traj2d_w_z_sel)
        traj3d_world_t = torch.from_numpy(traj3d_world)
        visibility_t = torch.from_numpy(visibility_sel)
        track_valid_t = torch.from_numpy(track_valid)

        if novel_rgbs is not None:
            novel_rgbs = torch.from_numpy(novel_rgbs).permute(0, 1, 4, 2, 3).float()
            novel_intrs = torch.from_numpy(novel_intrs).float()
            novel_extrs = torch.from_numpy(novel_extrs).float()

        # Track selection
        n_tracks = traj3d_world_t.shape[1]
        cache_root = os.path.join(datapoint_path, "cache")
        os.makedirs(cache_root, exist_ok=True)
        cache_file = os.path.join(cache_root, f"{self.cache_name}.npz")

        use_cache = bool(self.use_cached_tracks) and os.path.isfile(cache_file)
        if use_cache:
            cache = np.load(cache_file)
            inds_sampled = torch.from_numpy(cache["track_indices"])
            traj2d_w_z_t = torch.from_numpy(cache["traj2d_w_z"])
            traj3d_world_t = torch.from_numpy(cache["traj3d_world"])
            visibility_t = torch.from_numpy(cache["visibility"])
            valids = torch.from_numpy(cache["valids"])
            query_points = torch.from_numpy(cache["query_points"])
        else:
            # Filter to tracks visible in at least two frames
            visible_for_at_least_two_frames = visibility_t.any(0).sum(0) >= 2
            # Also require track to have valid data
            valid_in_most_frames = track_valid_t.sum(0) >= (n_frames * 0.5)
            valid_tracks = (visible_for_at_least_two_frames & valid_in_most_frames).nonzero(as_tuple=False)[:, 0]

            if len(valid_tracks) == 0:
                warnings.warn(f"No valid tracks in {self.seq_names[index]}, using all tracks")
                valid_tracks = torch.arange(n_tracks)

            point_inds = torch.randperm(len(valid_tracks), generator=rnd_torch)
            traj_per_sample = min(self.traj_per_sample or len(point_inds), len(valid_tracks))
            point_inds = point_inds[:traj_per_sample]
            inds_sampled = valid_tracks[point_inds]

            n_tracks = len(inds_sampled)
            traj2d_t = traj2d_t[:, :, inds_sampled].float()
            traj2d_w_z_t = traj2d_w_z_t[:, :, inds_sampled].float()
            traj3d_world_t = traj3d_world_t[:, inds_sampled].float()
            visibility_t = visibility_t[:, :, inds_sampled]

            valids = ~torch.isnan(traj2d_t).any(dim=-1).any(dim=0)
            valids = valids & track_valid_t[:, inds_sampled]

            # Create query points
            gt_vis_any_view = visibility_t.any(dim=0)
            vis_count = gt_vis_any_view.sum(dim=0)
            has_enough = vis_count >= 2
            if not has_enough.all():
                # For tracks with <2 visible frames, mark them visible everywhere they have data
                for i in range(n_tracks):
                    if not has_enough[i]:
                        gt_vis_any_view[:, i] = track_valid_t[:, inds_sampled[i]]

            last_visible_index = (torch.arange(n_frames).unsqueeze(-1) * gt_vis_any_view).max(0).values
            gt_vis_for_query = gt_vis_any_view.clone()
            gt_vis_for_query[last_visible_index[None, :].long(), torch.arange(n_tracks)] = False
            has_query = gt_vis_for_query.sum(dim=0) >= 1
            if not has_query.all():
                gt_vis_for_query = gt_vis_any_view.clone()

            query_points_t = torch.argmax(gt_vis_for_query.float(), dim=0)
            query_points_xyz = traj3d_world_t[query_points_t, torch.arange(n_tracks)]
            query_points = torch.cat([query_points_t[:, None].float(), query_points_xyz], dim=1)

            # Replace nans with zeros
            traj2d_t[torch.isnan(traj2d_t)] = 0
            traj2d_w_z_t[torch.isnan(traj2d_w_z_t)] = 0
            traj3d_world_t[torch.isnan(traj3d_world_t)] = 0

            if self.use_cached_tracks:
                logging.warning(f"Caching tracks for {self.seq_names[index]} at {os.path.abspath(cache_file)}")
                np.savez_compressed(
                    cache_file,
                    track_indices=inds_sampled.numpy(),
                    traj2d_w_z=traj2d_w_z_t.numpy(),
                    traj3d_world=traj3d_world_t.numpy(),
                    visibility=visibility_t.numpy(),
                    valids=valids.numpy(),
                    query_points=query_points.numpy(),
                )

        # Scene normalization (same as PanopticStudioMultiViewDataset)
        scale = 2.5
        rot_x = R.from_euler("x", -90, degrees=True).as_matrix()
        rot_y = R.from_euler("y", 0, degrees=True).as_matrix()
        rot_z = R.from_euler("z", 0, degrees=True).as_matrix()
        rot = torch.from_numpy(rot_z @ rot_y @ rot_x)
        translate = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

        (
            depths_trans, extrs_trans, query_points_trans,
            traj3d_world_trans, traj2d_w_z_trans
        ) = transform_scene(scale, rot, translate, depths, extrs_t, query_points, traj3d_world_t, traj2d_w_z_t)

        novel_extrs_trans = None
        if novel_extrs is not None:
            novel_extrs_trans = transform_scene(scale, rot, translate, None, novel_extrs, None, None, None)[1]

        segs = torch.ones((n_frames, 1, h, w))
        n_sel_views = rgbs.shape[0]

        # Extract SAM3D joints (first 70 keypoints per person) from human_tracks.npz
        # These are used to guide MVTracker's attention and point cloud
        n_keypoints = int(tracks["n_keypoints"])
        n_persons = int(tracks["n_persons"])
        track_types = tracks["track_types"]  # 0=keypoint, 1=vertex

        # Extract joint trajectories (world-space 3D)
        joint_mask = track_types == 0  # keypoints only
        joints_3d_world = traj3d_world[..., joint_mask, :]  # [T, 70*n_persons, 3]
        joints_3d_world = joints_3d_world.reshape(n_frames, n_persons, n_keypoints, 3)  # [T, P, 70, 3]

        # Apply same scene transformation as trajectories
        joints_3d_world_homo = torch.cat([
            torch.from_numpy(joints_3d_world),
            torch.ones(n_frames, n_persons, n_keypoints, 1)
        ], dim=-1).float()  # [T, P, 70, 4]

        # Transform: scale, rotate
        joints_3d_world_trans = joints_3d_world_homo[..., :3] * scale
        joints_3d_world_trans = torch.einsum('ij,TPKj->TPKi', rot.float(), joints_3d_world_trans)
        joints_3d_world_trans = joints_3d_world_trans + translate

        # Extract vertex trajectories (world-space 3D) for point cloud augmentation
        vertex_mask = track_types == 1  # vertices only
        if vertex_mask.any():
            vertices_3d_world = traj3d_world[..., vertex_mask, :]  # [T, n_verts_total, 3]
            vertices_3d_world_t = torch.from_numpy(vertices_3d_world).float()
            # Apply same scene transformation
            vertices_3d_world_trans = vertices_3d_world_t * scale
            vertices_3d_world_trans = torch.einsum('ij,TNj->TNi', rot.float(), vertices_3d_world_trans)
            vertices_3d_world_trans = vertices_3d_world_trans + translate
        else:
            vertices_3d_world_trans = None

        datapoint = Datapoint(
            video=rgbs,
            videodepth=depths_trans,
            feats=None,
            segmentation=segs,
            trajectory=traj2d_w_z_trans,
            trajectory_3d=traj3d_world_trans,
            trajectory_category=None,
            visibility=visibility_t,
            valid=valids,
            seq_name=self.seq_names[index],
            intrs=intrs_t,
            extrs=extrs_trans,
            query_points=None,
            query_points_3d=query_points_trans,
            track_upscaling_factor=1 / scale,
            novel_video=novel_rgbs,
            novel_intrs=novel_intrs,
            novel_extrs=novel_extrs_trans,
            sam3d_joints_world=joints_3d_world_trans,  # [T, n_persons, 70, 3]
            sam3d_vertices_world=vertices_3d_world_trans,  # [T, n_verts_total, 3] or None
        )
        return datapoint
