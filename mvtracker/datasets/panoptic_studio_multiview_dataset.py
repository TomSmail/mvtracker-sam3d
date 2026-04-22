import logging
import os
import pathlib
import re
import time
import warnings

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset

from mvtracker.datasets.utils import Datapoint, transform_scene


class PanopticStudioMultiViewDataset(Dataset):
    @staticmethod
    def from_name(dataset_name: str, dataset_root: str):
        """
        Examples of datasets supported by this factory method:
        - panoptic-multiview
        - panoptic-multiview-views27_16_14_8
        - panoptic-multiview-duster27_16_14_8
        - panoptic-multiview-duster27_16_14_8cleaned
        - panoptic-multiview-duster27_16_14_8cleaned-views27_16
        - panoptic-multiview-duster27_16_14_8cleaned-views27_16-novelviews1_4
        - panoptic-multiview-duster27_16_14_8cleaned-views27_16-novelviews1_4-single
        - panoptic-multiview-duster27_16_14_8cleaned-views27_16-novelviews1_4-single-2dpt
        - panoptic-multiview-duster27_16_14_8cleaned-views27_16-novelviews1_4-single-2dpt-cached
        """
        # Parse the dataset name, chunk by chunk
        non_parsed = dataset_name.replace("panoptic-multiview", "", 1)

        if non_parsed.startswith("-duster"):
            match = re.match(r"-duster((?:\d+_?)+)(cleaned)?", non_parsed)
            assert match is not None
            duster_views = list(map(int, match.group(1).split("_")))
            use_duster = True
            use_duster_cleaned = match.group(2) is not None
            non_parsed = non_parsed.replace(match.group(0), "", 1)
        else:
            use_duster = False
            use_duster_cleaned = False
            duster_views = None

        if non_parsed.startswith("-views"):
            match = re.match(r"-views((?:\d+_?)+)", non_parsed)
            assert match is not None
            views = list(map(int, match.group(1).split("_")))
            if duster_views is not None:
                assert all(v in duster_views for v in views)
            non_parsed = non_parsed.replace(match.group(0), "", 1)
        else:
            views = duster_views

        if non_parsed.startswith("-novelviews"):
            match = re.match(r"-novelviews((?:\d+_?)+)", non_parsed)
            assert match is not None
            novel_views = list(map(int, match.group(1).split("_")))
            non_parsed = non_parsed.replace(match.group(0), "", 1)
        else:
            novel_views = None

        if non_parsed.startswith("-single"):
            single_point = True
            non_parsed = non_parsed.replace("-single", "", 1)
        else:
            single_point = False

        if non_parsed.startswith("-2dpt"):
            eval_2dpt = True
            non_parsed = non_parsed.replace("-2dpt", "", 1)
        else:
            eval_2dpt = False

        if non_parsed.startswith("-cached"):
            use_cached_tracks = True
            non_parsed = non_parsed.replace("-cached", "", 1)
        else:
            use_cached_tracks = False

        assert non_parsed == "", f"Unparsed part of the dataset name: {non_parsed}"

        return PanopticStudioMultiViewDataset(
            data_root=os.path.join(dataset_root, "panoptic-multiview"),
            views_to_return=views,
            novel_views=novel_views,
            use_duster_depths=use_duster,
            clean_duster_depths=use_duster_cleaned,
            traj_per_sample=384,
            seed=72,
            max_videos=6,
            perform_sanity_checks=False,
            use_cached_tracks=use_cached_tracks,
        )

    def __init__(
            self,
            data_root,
            views_to_return=None,
            novel_views=None,
            use_duster_depths=False,
            clean_duster_depths=False,
            traj_per_sample=512,
            seed=None,
            max_videos=None,
            perform_sanity_checks=False,
            use_cached_tracks=False,
    ):
        super().__init__()
        self.data_root = data_root
        self.views_to_return = views_to_return
        self.novel_views = novel_views
        self.use_duster_depths = use_duster_depths
        self.clean_duster_depths = clean_duster_depths
        self.traj_per_sample = traj_per_sample
        self.seed = seed
        self.perform_sanity_checks = perform_sanity_checks
        self.use_cached_tracks = use_cached_tracks
        self.cache_name = self._cache_key()
        self.seq_names = self._get_sequence_names(max_videos)
        self.getitem_calls = 0

    def _get_sequence_names(self, max_videos):
        """
        Fetch all valid sequence names from the dataset root.

        Args:
            max_videos (int): Limit the number of sequences to load.

        Returns:
            List[str]: Sorted list of valid sequence names.
        """
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
            if not os.path.exists(os.path.join(scene_path, "tapvid3d_annotations.npz")):
                warnings.warn(f"Skipping {scene_path} because it has no tapvid3d_annotations.npz labels file.")
                continue

            if self.use_duster_depths and self.views_to_return is not None:
                views_str = '-'.join(str(v) for v in self.views_to_return)
                duster_dir = os.path.join(scene_path, f'duster-views-{views_str}')
                if not os.path.isdir(duster_dir):
                    warnings.warn(f"Skipping {seq_name}: no DUSt3R depths at {duster_dir}")
                    continue
            elif not self.use_duster_depths:
                d3dgs_dir = os.path.join(scene_path, "dynamic3dgs_depth")
                if not os.path.isdir(d3dgs_dir):
                    warnings.warn(f"Skipping {seq_name}: no dynamic3dgs_depth directory")
                    continue

            valid_seqs.append(seq_name)

        if max_videos is not None:
            valid_seqs = valid_seqs[:max_videos]

        print(f"Using {len(valid_seqs)} videos from {self.data_root}")
        return valid_seqs

    def _cache_key(self):
        name = f"cachedtracks--seed{self.seed}"
        if self.views_to_return is not None:
            name += f"-views{'_'.join(map(str, self.views_to_return))}"
        if self.traj_per_sample is not None:
            name += f"-n{self.traj_per_sample}"
        return name + "--v1"  # bump this if you change the selection policy

    def __len__(self):
        return len(self.seq_names)

    def __getitem__(self, index):
        start_time = time.time()
        sample = self._getitem_helper(index)

        self.getitem_calls += 1
        if self.getitem_calls < 10:
            print(f"Loading {index:>06d} took  {time.time() - start_time:.3f} sec. Getitem calls: {self.getitem_calls}")

        return sample, True

    def _getitem_helper(self, index):
        """
        Helper function to load a single sample.

        Args:
            index (int): Index of the sample to load.

        Returns:
            CoTrackerData, bool: Sample data and success flag.
        """
        if self.seed is None:
            seed = torch.randint(0, 2 ** 32 - 1, (1,)).item()
        else:
            seed = self.seed
        rnd_torch = torch.Generator().manual_seed(seed)
        rnd_np = np.random.RandomState(seed=seed)

        datapoint_path = os.path.join(self.data_root, self.seq_names[index])
        ims_path = os.path.join(datapoint_path, "ims")
        depths_path = os.path.join(datapoint_path, "dynamic3dgs_depth")

        tapvid3d_merged_annotations = np.load(os.path.join(datapoint_path, "tapvid3d_annotations.npz"))
        traj3d_world = tapvid3d_merged_annotations["trajectories"]
        traj2d = tapvid3d_merged_annotations["trajectories_pixelspace"]
        visibility = tapvid3d_merged_annotations["per_view_visibilities"]
        query_points_3d = tapvid3d_merged_annotations["query_points_3d"]
        extrs = tapvid3d_merged_annotations["extrinsics"]
        intrs = tapvid3d_merged_annotations["intrinsics"]

        views = {}
        view_folders = sorted([f for f in os.listdir(ims_path)], key=lambda x: int(x))
        if self.views_to_return is not None:
            views_to_return = self.views_to_return
        else:
            views_to_return = sorted(list(range(len(view_folders))))
        views_to_load = views_to_return.copy()
        if self.novel_views is not None:
            views_to_load = list(set(views_to_load + self.novel_views))
        for v in views_to_load:
            rgb_folder = os.path.join(ims_path, str(v))
            rgb_files = sorted(os.listdir(rgb_folder))
            rgb_images = [cv2.imread(os.path.join(rgb_folder, f))[:, :, ::-1] for f in rgb_files]
            depth = np.load(os.path.join(depths_path, f"depths_{v:02d}.npy"))
            views[v] = {
                "rgb": np.stack(rgb_images),
                "depth": depth,
            }

        rgbs = np.stack([views[v]["rgb"] for v in views_to_return])
        n_views, n_frames, h, w, _ = rgbs.shape
        depths = np.stack([views[v]["depth"] for v in views_to_return])[..., None].astype(np.float32)
        intrs = np.stack([intrs[v] for v in views_to_return])[:, None, :, :].repeat(n_frames, axis=1)
        extrs = np.stack([extrs[v][:3, :] for v in views_to_return])[:, None, :, :].repeat(n_frames, axis=1)
        visibility = visibility[views_to_return]
        traj2d = traj2d[views_to_return]

        # Load novel views if they exist
        novel_rgbs = None
        novel_intrs = None
        novel_extrs = None
        if self.novel_views is not None:
            novel_rgbs = np.stack([views[v]["rgb"]
                                   for v in self.novel_views])
            novel_intrs = np.stack([tapvid3d_merged_annotations["intrinsics"][v]
                                    for v in self.novel_views])[:, None, :, :].repeat(n_frames, axis=1)
            novel_extrs = np.stack([tapvid3d_merged_annotations["extrinsics"][v][:3, :]
                                    for v in self.novel_views])[:, None, :, :].repeat(n_frames, axis=1)

        # Load Duster's features and estimated depths if they exist
        views_selection_str = '-'.join(str(v) for v in self.views_to_return)
        duster_root = pathlib.Path(datapoint_path) / f'duster-views-{views_selection_str}'
        if self.use_duster_depths:
            assert duster_root.exists(), f"Duster root {duster_root} does not exist."
            last_frame_scene_file = duster_root / f"3d_model__{n_frames - 1:05d}__scene.npz"
            assert last_frame_scene_file.exists(), f"Duster scene file {last_frame_scene_file} does not exist."

        feats = None
        feat_dim = None
        feat_stride = None
        if duster_root.exists() and (duster_root / f"3d_model__{n_frames - 1:05d}__scene.npz").exists():
            duster_depths = []
            duster_feats = []
            for frame_idx in range(n_frames):
                scene = np.load(duster_root / f"3d_model__{frame_idx:05d}__scene.npz")
                duster_depth = torch.from_numpy(scene["depths"])
                duster_conf = torch.from_numpy(scene["confs"])
                duster_msk = torch.from_numpy(scene["cleaned_mask"])
                duster_feat = torch.from_numpy(scene["feats"])

                if self.clean_duster_depths:
                    duster_depth = duster_depth * duster_msk

                duster_depth = F.interpolate(duster_depth[:, None], (h, w), mode='nearest')
                duster_depths.append(duster_depth[:, 0, :, :, None])
                duster_feats.append(duster_feat)

            feats = torch.stack(duster_feats, dim=1).numpy()
            assert feats.ndim == 4
            assert feats.shape[0] == n_views
            assert feats.shape[1] == n_frames
            feat_stride = np.round(np.sqrt(h * w / feats.shape[2])).astype(int)
            feat_dim = feats.shape[3]
            feats = feats.reshape(n_views, n_frames, h // feat_stride, w // feat_stride, feat_dim)

            # Replace the depths with the Duster depths, if configured so
            if self.use_duster_depths:
                depths = torch.stack(duster_depths, dim=1).numpy()

        n_tracks = traj3d_world.shape[1]
        assert rgbs.shape == (n_views, n_frames, h, w, 3)
        assert depths.shape == (n_views, n_frames, h, w, 1)
        assert feats is None or feats.shape == (n_views, n_frames, h // feat_stride, w // feat_stride, feat_dim)
        assert intrs.shape == (n_views, n_frames, 3, 3)
        assert extrs.shape == (n_views, n_frames, 3, 4)
        assert traj2d.shape == (n_views, n_frames, n_tracks, 2)
        assert visibility.shape == (n_views, n_frames, n_tracks)
        assert traj3d_world.shape == (n_frames, n_tracks, 3)

        if novel_rgbs is not None:
            assert novel_rgbs.shape == (len(self.novel_views), n_frames, h, w, 3)
            assert novel_intrs.shape == (len(self.novel_views), n_frames, 3, 3)
            assert novel_extrs.shape == (len(self.novel_views), n_frames, 3, 4)

        # Make sure our intrinsics and extrinsics work correctly
        point_3d_world = traj3d_world
        point_4d_world_homo = np.concatenate([point_3d_world, np.ones_like(point_3d_world[..., :1])], axis=-1)
        point_3d_camera = np.einsum('ABij,BCj->ABCi', extrs, point_4d_world_homo)
        if self.perform_sanity_checks:
            point_2d_pixel_homo = np.einsum('ABij,ABCj->ABCi', intrs, point_3d_camera)
            point_2d_pixel = point_2d_pixel_homo[..., :2] / point_2d_pixel_homo[..., 2:]
            point_2d_pixel_gt = traj2d

            point_2d_pixel_no_nan = np.nan_to_num(point_2d_pixel, nan=0)
            point_2d_pixel_gt_no_nan = np.nan_to_num(point_2d_pixel_gt, nan=0)

            assert np.allclose(point_2d_pixel_no_nan[0, :, 0, :], point_2d_pixel_no_nan[0, :, 0, :], atol=.01)
            assert np.allclose(point_2d_pixel_gt_no_nan, point_2d_pixel_gt_no_nan, atol=.01), f"Point projection failed"
        traj2d_w_z = np.concatenate([traj2d, point_3d_camera[..., 2:]], axis=-1)

        rgbs = torch.from_numpy(rgbs).permute(0, 1, 4, 2, 3).float()
        depths = torch.from_numpy(depths).permute(0, 1, 4, 2, 3).float()
        feats = torch.from_numpy(feats).permute(0, 1, 4, 2, 3).float() if feats is not None else None
        intrs = torch.from_numpy(intrs).float()
        extrs = torch.from_numpy(extrs).float()
        traj2d = torch.from_numpy(traj2d)
        traj2d_w_z = torch.from_numpy(traj2d_w_z)
        traj3d_world = torch.from_numpy(traj3d_world)
        visibility = torch.from_numpy(visibility)
        if novel_rgbs is not None:
            novel_rgbs = torch.from_numpy(novel_rgbs).permute(0, 1, 4, 2, 3).float()
            novel_intrs = torch.from_numpy(novel_intrs).float()
            novel_extrs = torch.from_numpy(novel_extrs).float()

        # Track selection
        cache_root = os.path.join(self.data_root, self.seq_names[index], "cache")
        os.makedirs(cache_root, exist_ok=True)
        cache_file = os.path.join(cache_root, f"{self.cache_name}.npz")

        # Check if we can use cached tracks
        use_cache = bool(self.use_cached_tracks) and os.path.isfile(cache_file)
        if use_cache:
            cache = np.load(cache_file)
            inds_sampled = torch.from_numpy(cache["track_indices"])
            traj2d_w_z = torch.from_numpy(cache["traj2d_w_z"])
            traj3d_world = torch.from_numpy(cache["traj3d_world"])
            visibility = torch.from_numpy(cache["visibility"])
            valids = torch.from_numpy(cache["valids"])
            query_points = torch.from_numpy(cache["query_points"])

        # Otherwise, sample the tracks and create query points
        else:
            # Prefer TAPVid-3D's merged query points when selecting the query points
            # First, denote the points in time before the query points appeared as non-visible
            # Second, choose the query points as the first appearance of the points in the selected views (which might be
            # later than in the TAPVid-3D annotations because the query might not be visible in the selected views)
            tapvid3d_merged_query_point_timestep = query_points_3d[:, 0].round().astype(int)
            visibility *= (np.arange(n_frames)[None, :, None] >= tapvid3d_merged_query_point_timestep[None, None, :])

            # Sample the points to track
            visible_for_at_least_two_frames = visibility.any(0).sum(0) >= 2
            valid_tracks = visible_for_at_least_two_frames
            valid_tracks = valid_tracks.nonzero(as_tuple=False)[:, 0]

            point_inds = torch.randperm(len(valid_tracks), generator=rnd_torch)
            traj_per_sample = self.traj_per_sample if self.traj_per_sample is not None else len(point_inds)
            assert len(point_inds) >= traj_per_sample
            point_inds = point_inds[:traj_per_sample]
            inds_sampled = valid_tracks[point_inds]

            n_tracks = len(inds_sampled)
            traj2d = traj2d[:, :, inds_sampled].float()
            traj2d_w_z = traj2d_w_z[:, :, inds_sampled].float()
            traj3d_world = traj3d_world[:, inds_sampled].float()
            visibility = visibility[:, :, inds_sampled]

            valids = ~torch.isnan(traj2d).any(dim=-1).any(dim=0)

            # Create the query points
            gt_visibilities_any_view = visibility.any(dim=0)
            assert (gt_visibilities_any_view.sum(dim=0) >= 2).all(), "All points should be visible in least two frames."
            last_visible_index = (torch.arange(n_frames).unsqueeze(-1) * gt_visibilities_any_view).max(0).values
            assert gt_visibilities_any_view[last_visible_index[None, :], torch.arange(n_tracks)].all()
            gt_visibilities_any_view[last_visible_index[None, :], torch.arange(n_tracks)] = False
            assert (gt_visibilities_any_view.sum(dim=0) >= 1).all()

            query_points_t = torch.argmax(gt_visibilities_any_view.float(), dim=0)
            query_points_xyz_worldspace = traj3d_world[query_points_t, torch.arange(n_tracks)]
            query_points = torch.cat([query_points_t[:, None], query_points_xyz_worldspace], dim=1)
            assert gt_visibilities_any_view[query_points_t, torch.arange(n_tracks)].all()

            # Replace nans with zeros
            traj2d[torch.isnan(traj2d)] = 0
            traj2d_w_z[torch.isnan(traj2d_w_z)] = 0
            traj3d_world[torch.isnan(traj3d_world)] = 0
            assert torch.isnan(visibility).sum() == 0

            # Cache the selected tracks and query points
            if self.use_cached_tracks:
                logging.warn(f"Caching tracks for {self.seq_names[index]} at {os.path.abspath(cache_file)}")
                np.savez_compressed(
                    cache_file,
                    track_indices=inds_sampled.numpy(),
                    traj2d_w_z=traj2d_w_z.numpy(),
                    traj3d_world=traj3d_world.numpy(),
                    visibility=visibility.numpy(),
                    valids=valids.numpy(),
                    query_points=query_points.numpy(),
                )

        # Normalize the scene to be similar to Kubric's scene
        scale = 2.5
        rot_x = R.from_euler('x', -90, degrees=True).as_matrix()
        rot_y = R.from_euler('y', 0, degrees=True).as_matrix()
        rot_z = R.from_euler('z', 0, degrees=True).as_matrix()
        rot = torch.from_numpy(rot_z @ rot_y @ rot_x)
        translate = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        (
            depths_trans, extrs_trans, query_points_trans, traj3d_world_trans, traj2d_w_z_trans
        ) = transform_scene(scale, rot, translate, depths, extrs, query_points, traj3d_world, traj2d_w_z)
        novel_extrs_trans = transform_scene(scale, rot, translate, None, novel_extrs, None, None, None)[1]

        # # Use the auto scene normalization of generic scenes
        # from mvtracker.datasets.generic_scene_dataset import compute_auto_scene_normalization
        # scale, rot, translation = compute_auto_scene_normalization(depths, torch.ones_like(depths) * 100, extrs_trans, intrs)
        # scale = scale * T[0, 0].item()
        # print(f"{scale=}")
        # (depths_trans, extrs_trans, query_points_trans, traj3d_world_trans, traj2d_w_z_trans
        # ) = transform_scene(scale, rot, translation, depths_trans, extrs_trans, query_points_trans, traj3d_world_trans, traj2d_w_z_trans)
        # _, novel_extrs_trans, _, _, _ = transform_scene(scale, rot, translation, None, novel_extrs_trans, None, None, None)
        # 85.7 94.5 92.3 --> 86.0 94.8 92.2

        # from mvtracker.datasets.dexycb_multiview_dataset import rerun_viz_scene
        # rerun_viz_scene("nane/pc__no_transform/", rgbs[:, ::20], depths[:, ::20], intrs[:, ::20], extrs[:, ::20], traj3d_world[:, ::20], 0.1)
        # rerun_viz_scene("nane/pc_transformed/", rgbs[:, ::20], depths[:, ::20], intrs[:, ::20], extrs_trans[:, ::20], traj3d_world_trans[:, ::20], 1)

        segs = torch.ones((n_frames, 1, h, w))  # Dummy segmentation masks
        datapoint = Datapoint(
            video=rgbs,
            videodepth=depths_trans,
            feats=feats,
            segmentation=segs,
            trajectory=traj2d_w_z_trans,
            trajectory_3d=traj3d_world_trans,
            trajectory_category=None,
            visibility=visibility,
            valid=valids,
            seq_name=self.seq_names[index],
            intrs=intrs,
            extrs=extrs_trans,
            query_points=None,
            query_points_3d=query_points_trans,
            track_upscaling_factor=1 / scale,

            novel_video=novel_rgbs,
            novel_intrs=novel_intrs,
            novel_extrs=novel_extrs_trans,
        )
        return datapoint
