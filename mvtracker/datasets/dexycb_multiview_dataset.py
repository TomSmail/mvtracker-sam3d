import logging
import os
import pathlib
import re
import time
import warnings

import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset

from mvtracker.datasets.utils import Datapoint, transform_scene


class DexYCBMultiViewDataset(Dataset):

    @staticmethod
    def from_name(dataset_name: str, dataset_root: str, use_sam3d: bool = False,
                  sam3d_checkpoint_path: str = None, sam3d_mhr_path: str = None):
        """
        Examples of datasets supported by this factory method:
        - "dex-ycb-multiview",
        - "dex-ycb-multiview-single",
        - "dex-ycb-multiview-removehand",
        - "dex-ycb-multiview-duster0123",
        - "dex-ycb-multiview-duster0123cleaned",
        - "dex-ycb-multiview-duster0123cleaned-views0123",
        - "dex-ycb-multiview-duster0123cleaned-views0123-novelviews45",
        - "dex-ycb-multiview-duster0123cleaned-views0123-novelviews45-removehand",
        - "dex-ycb-multiview-duster0123cleaned-views0123-novelviews45-removehand-single",
        - "dex-ycb-multiview-duster0123cleaned-views0123-novelviews45-removehand-2dpt-single",
        - "dex-ycb-multiview-duster0123cleaned-views0123-novelviews45-removehand-2dpt-single-cached",
        """
        # Parse the dataset name, chunk by chunk
        non_parsed = dataset_name.replace("dex-ycb-multiview", "", 1)

        if non_parsed.startswith("-duster"):
            match = re.match(r"-duster(\d+)(cleaned)?", non_parsed)
            assert match is not None
            duster_views = list(map(int, match.group(1)))
            use_duster = True
            use_duster_cleaned = match.group(2) is not None
            non_parsed = non_parsed.replace(match.group(0), "", 1)
        else:
            use_duster = False
            use_duster_cleaned = False
            duster_views = None

        if non_parsed.startswith("-views"):
            match = re.match(r"-views(\d+)", non_parsed)
            assert match is not None
            views = list(map(int, match.group(1)))
            if duster_views is not None:
                assert all(v in duster_views for v in views)
            non_parsed = non_parsed.replace(match.group(0), "", 1)
        else:
            views = duster_views

        if non_parsed.startswith("-novelviews"):
            match = re.match(r"-novelviews(\d+)", non_parsed)
            assert match is not None
            novel_views = list(map(int, match.group(1)))
            non_parsed = non_parsed.replace(match.group(0), "", 1)
        else:
            novel_views = None

        if non_parsed.startswith("-removehand"):
            remove_hand = True
            non_parsed = non_parsed.replace("-removehand", "", 1)
        else:
            remove_hand = False

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

        if views is None and duster_views is None:
            views = [0, 1, 2, 3]  # Make the legacy "dex-ycb-multiview" name take the first 4 views (not all 8)

        return DexYCBMultiViewDataset(
            data_root=os.path.join(dataset_root, "dex-ycb-multiview"),
            views_to_return=views,
            novel_views=novel_views,
            remove_hand=remove_hand,
            use_duster_depths=use_duster,
            duster_views=duster_views,
            clean_duster_depths=use_duster_cleaned,
            traj_per_sample=384,
            seed=72,
            max_videos=10,
            perform_sanity_checks=False,
            use_cached_tracks=use_cached_tracks,
            use_sam3d=use_sam3d,
            sam3d_checkpoint_path=sam3d_checkpoint_path,
            sam3d_mhr_path=sam3d_mhr_path,
        )

    def __init__(
            self,
            data_root,
            remove_hand=False,
            views_to_return=None,
            novel_views=None,
            use_duster_depths=False,
            clean_duster_depths=False,
            duster_views=None,
            traj_per_sample=768,
            seed=None,
            max_videos=None,
            perform_sanity_checks=False,
            use_cached_tracks=False,
            use_sam3d=False,
            sam3d_checkpoint_path=None,
            sam3d_mhr_path=None,
    ):
        super().__init__()
        self.data_root = data_root
        self.remove_hand = remove_hand
        self.views_to_return = views_to_return
        self.novel_views = novel_views
        self.use_duster_depths = use_duster_depths
        self.clean_duster_depths = clean_duster_depths
        self.duster_views = duster_views
        self.traj_per_sample = traj_per_sample
        self.seed = seed
        self.perform_sanity_checks = perform_sanity_checks
        self.use_cached_tracks = use_cached_tracks
        self.use_sam3d = use_sam3d
        self.sam3d_checkpoint_path = sam3d_checkpoint_path
        self.sam3d_mhr_path = sam3d_mhr_path
        self.cache_name = self._cache_key()
        self.seq_names = self._get_sequence_names(max_videos)
        self.getitem_calls = 0

        # Initialize SAM3D wrapper if enabled
        self.sam3d_wrapper = None
        if self.use_sam3d:
            from mvtracker.datasets.sam3d_inference import get_sam3d_wrapper
            self.sam3d_wrapper = get_sam3d_wrapper(
                checkpoint_path=sam3d_checkpoint_path,
                mhr_path=sam3d_mhr_path,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )

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
            view_folders = [
                d for d in os.listdir(scene_path)
                if os.path.isdir(os.path.join(scene_path, d)) and d.startswith("view_")
            ]
            if not view_folders:
                warnings.warn(f"Skipping {scene_path} because it has no views.")
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
        if self.remove_hand:
            name += "-removehand"
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

        views = {}
        view_folders = sorted([f for f in os.listdir(datapoint_path) if f.startswith("view_")])
        if self.views_to_return is not None:
            views_to_return = self.views_to_return
        else:
            views_to_return = sorted(list(range(len(view_folders))))
        views_to_load = views_to_return.copy()
        if self.novel_views is not None:
            views_to_load = list(set(views_to_load + self.novel_views))
        for v in views_to_load:
            view_path = os.path.join(datapoint_path, view_folders[v])

            # Load RGB images
            rgb_folder = os.path.join(view_path, "rgb")
            rgb_files = sorted(os.listdir(rgb_folder))
            rgb_images = [cv2.imread(os.path.join(rgb_folder, f))[:, :, ::-1] for f in rgb_files]

            # Load depth maps
            depth_folder = os.path.join(view_path, "depth")
            depth_files = sorted(os.listdir(depth_folder))
            depth_images = [cv2.imread(os.path.join(depth_folder, f), cv2.IMREAD_ANYDEPTH) for f in depth_files]

            # Load camera parameters
            camera_params_file = os.path.join(view_path, "intrinsics_extrinsics.npz")
            params = np.load(camera_params_file)
            intrinsics = params["intrinsics"][:3, :3]  # Extract K
            extrinsics = params["extrinsics"][:3, :]  # Extract R|t (world to camera)

            views[v] = {
                "rgb": np.stack(rgb_images),
                "depth": np.stack(depth_images),
                "intrinsics": intrinsics,
                "extrinsics": extrinsics,
            }

        rgbs = np.stack([views[v]["rgb"] for v in views_to_return])
        n_views, n_frames, h, w, _ = rgbs.shape
        depths = np.stack([views[v]["depth"] for v in views_to_return])[..., None].astype(np.float32) / 1000
        intrs = np.stack([views[v]["intrinsics"] for v in views_to_return])[:, None, :, :].repeat(n_frames, axis=1)
        extrs = np.stack([views[v]["extrinsics"] for v in views_to_return])[:, None, :, :].repeat(n_frames, axis=1)

        # Load novel views if they exist
        novel_rgbs = None
        novel_intrs = None
        novel_extrs = None
        if self.novel_views is not None:
            novel_rgbs = np.stack([views[v]["rgb"] for v in self.novel_views])
            novel_intrs = np.stack([views[v]["intrinsics"] for v in self.novel_views])[:, None, :, :].repeat(n_frames,
                                                                                                             axis=1)
            novel_extrs = np.stack([views[v]["extrinsics"] for v in self.novel_views])[:, None, :, :].repeat(n_frames,
                                                                                                             axis=1)

        # Load Duster's features and estimated depths if they exist
        duster_views = self.duster_views if self.duster_views is not None else views_to_return
        duster_views_str = ''.join(str(v) for v in duster_views)
        duster_root = pathlib.Path(datapoint_path) / f'duster-views-{duster_views_str}'
        if self.use_duster_depths:
            assert duster_root.exists() and (duster_root / f"3d_model__{n_frames - 1:05d}__scene.npz").exists(), \
                f"Duster root {duster_root} does not exist."

        feats = None
        feat_dim = None
        feat_stride = None
        depth_confs = None
        if duster_root.exists() and (duster_root / f"3d_model__{n_frames - 1:05d}__scene.npz").exists():
            duster_depths = []
            duster_confs = []
            duster_feats = []
            for frame_idx in range(n_frames):
                scene = np.load(duster_root / f"3d_model__{frame_idx:05d}__scene.npz")
                duster_depth = torch.from_numpy(scene["depths"])
                duster_conf = torch.from_numpy(scene["confs"])
                duster_msk = torch.from_numpy(scene["cleaned_mask"])

                if self.clean_duster_depths:
                    duster_depth = duster_depth * duster_msk

                duster_depth = F.interpolate(duster_depth[:, None], (h, w), mode='nearest')
                duster_depths.append(duster_depth[:, 0, :, :, None])

                duster_conf = F.interpolate(duster_conf[:, None], (h, w), mode='nearest')
                duster_confs.append(duster_conf[:, 0, :, :, None])

                if "feats" in scene:
                    duster_feats.append(torch.from_numpy(scene["feats"]))

            duster_depths = torch.stack(duster_depths, dim=1).numpy()
            duster_confs = torch.stack(duster_confs, dim=1).numpy()
            if duster_feats:
                feats = torch.stack(duster_feats, dim=1).numpy()

            # Extract the correct views
            assert duster_depths.shape[0] == len(duster_views)
            duster_depths = duster_depths[[duster_views.index(v) for v in views_to_return]]
            duster_confs = duster_confs[[duster_views.index(v) for v in views_to_return]]
            if feats is not None:
                assert feats.shape[0] == len(duster_views)
                feats = feats[[duster_views.index(v) for v in views_to_return]]

            # Reshape the features
            if feats is not None:
                assert feats.ndim == 4
                assert feats.shape[0] == n_views
                assert feats.shape[1] == n_frames
                feat_stride = np.round(np.sqrt(h * w / feats.shape[2])).astype(int)
                feat_dim = feats.shape[3]
                feats = feats.reshape(n_views, n_frames, h // feat_stride, w // feat_stride, feat_dim)

            # Replace the depths with the Duster depths, if configured so
            if self.use_duster_depths:
                depths = duster_depths
                depth_confs = duster_confs

        tracks_3d_file = os.path.join(datapoint_path, "tracks_3d.npz")
        tracks_3d_data = np.load(tracks_3d_file, allow_pickle=True)
        traj3d_world = tracks_3d_data["tracks_3d"]
        traj2d = tracks_3d_data["tracks_2d"][views_to_return]
        traj2d_w_z = np.concatenate((traj2d, tracks_3d_data["tracks_2d_z"][views_to_return][:, :, :, None]), axis=-1)
        visibility = tracks_3d_data["tracks_2d_visibilities"][views_to_return]

        # Label the trajectories according to: 0: hand, 1: moving ycb object, 2: static ycb objects
        object_id_to_name = tracks_3d_data["object_id_to_name"].item()
        traj_object_id = tracks_3d_data["object_ids"]
        for object_name in object_id_to_name.values():
            assert object_name == "mano-right-hand" or object_name.startswith("ycb")
        avg_movement_per_object_id = {}
        for object_id in np.unique(traj_object_id):
            object_mask = traj_object_id == object_id
            object_traj = traj3d_world[:, object_mask]
            avg_movement_per_object_id[object_id] = np.linalg.norm(object_traj[1:] - object_traj[:-1], axis=-1).mean()
        hand_id = {v: k for k, v in object_id_to_name.items()}["mano-right-hand"]
        dynamic_ycb_object_ids = [k for k, v in avg_movement_per_object_id.items() if v >= 1e-4 and k != hand_id]
        assert len(dynamic_ycb_object_ids) == 1
        dynamic_ycb_object_id = dynamic_ycb_object_ids[0]
        static_ycb_object_ids = [k for k, v in avg_movement_per_object_id.items() if v < 1e-4 and k != hand_id]
        assert 1 + 1 + len(static_ycb_object_ids) == len(object_id_to_name)
        # remap object ids to 0: hand, 1: dynamic ycb object, 2: static ycb objects
        traj_object_id = (
                0 * (traj_object_id == hand_id) +
                1 * (traj_object_id == dynamic_ycb_object_id) +
                2 * np.isin(traj_object_id, static_ycb_object_ids)
        )

        if self.remove_hand:
            traj3d_world = traj3d_world[:, traj_object_id > 0]
            traj2d = traj2d[:, :, traj_object_id > 0]
            traj2d_w_z = traj2d_w_z[:, :, traj_object_id > 0]
            visibility = visibility[:, :, traj_object_id > 0]
            traj_object_id = traj_object_id[traj_object_id > 0]

        n_tracks = traj3d_world.shape[1]
        assert rgbs.shape == (n_views, n_frames, h, w, 3)
        assert depths.shape == (n_views, n_frames, h, w, 1)
        assert depth_confs is None or depth_confs.shape == (n_views, n_frames, h, w, 1)
        assert feats is None or feats.shape == (n_views, n_frames, h // feat_stride, w // feat_stride, feat_dim)
        assert intrs.shape == (n_views, n_frames, 3, 3)
        assert extrs.shape == (n_views, n_frames, 3, 4)
        assert traj2d.shape == (n_views, n_frames, n_tracks, 2)
        assert visibility.shape == (n_views, n_frames, n_tracks)
        assert traj3d_world.shape == (n_frames, n_tracks, 3)
        assert traj_object_id.shape == (n_tracks,)

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

            assert np.allclose(point_2d_pixel_no_nan[0, :, 0, :], point_2d_pixel_no_nan[0, :, 0, :],
                               atol=1), f"Proj. failed"
            assert np.allclose(point_2d_pixel_gt_no_nan, point_2d_pixel_gt_no_nan, atol=1), f"Point projection failed"

            assert np.allclose(point_3d_camera[..., 2:], traj2d_w_z[..., -1:], atol=1)

        # Convert everything to torch tensors
        rgbs = torch.from_numpy(rgbs).permute(0, 1, 4, 2, 3).float()
        depths = torch.from_numpy(depths).permute(0, 1, 4, 2, 3).float()
        depth_confs = torch.from_numpy(depth_confs).permute(0, 1, 4, 2, 3).float() if depth_confs is not None else None
        feats = torch.from_numpy(feats).permute(0, 1, 4, 2, 3).float() if feats is not None else None
        intrs = torch.from_numpy(intrs).float()
        extrs = torch.from_numpy(extrs).float()
        traj2d = torch.from_numpy(traj2d)
        traj2d_w_z = torch.from_numpy(traj2d_w_z)
        traj3d_world = torch.from_numpy(traj3d_world)
        traj_object_id = torch.from_numpy(traj_object_id)
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
            traj_object_id = torch.from_numpy(cache["traj_object_id"])
            visibility = torch.from_numpy(cache["visibility"])
            valids = torch.from_numpy(cache["valids"])
            query_points = torch.from_numpy(cache["query_points"])

        # Otherwise, sample the tracks and create query points
        else:
            # Force query points on hand to appear later
            # This avoids querying when the GT hand reconstruction is severely lacking
            # Identify tracks that are invisible in the first frame across all views (as they are probably on the hand)
            invisible_at_first_frame = visibility[:, 0, :] == 0
            invisible_at_first_frame = invisible_at_first_frame.unsqueeze(1).expand(-1, 5, -1)
            # Set visibility to 0 for the first 5 frames where the first frame was invisible
            visibility[:, 0:5, :] *= ~invisible_at_first_frame  # Keep visible ones, set others to 0

            # Sample the points to track
            visible_for_at_least_two_frames = visibility.any(0).sum(0) >= 2
            hectic_visibility = ((visibility[:, :-1] & ~visibility[:, 1:]).sum(0) >= 3).any(0)
            valid_tracks = visible_for_at_least_two_frames & ~hectic_visibility
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
            traj_object_id = traj_object_id[inds_sampled]
            visibility = visibility[:, :, inds_sampled]

            valids = ~torch.isnan(traj2d).any(dim=-1).any(dim=0)

            # Create the query points
            gt_visibilities_any_view = visibility.any(dim=0)
            assert (gt_visibilities_any_view.sum(dim=0) >= 2).all(), "All points should be visible in least two frames."
            last_visible_index = (torch.arange(n_frames).unsqueeze(-1) * gt_visibilities_any_view).max(0).values
            assert gt_visibilities_any_view[last_visible_index[None, :], torch.arange(n_tracks)].all()
            gt_visibilities_any_view[last_visible_index[None, :], torch.arange(n_tracks)] = False
            assert (gt_visibilities_any_view.sum(dim=0) >= 1).all()

            n_non_first_point_appearance_queries = n_tracks // 4
            n_first_point_appearance_queries = n_tracks - n_non_first_point_appearance_queries

            first_point_appearances = torch.argmax(
                gt_visibilities_any_view[..., -n_first_point_appearance_queries:].float(), dim=0)
            non_first_point_appearances = first_point_appearances.new_zeros((n_non_first_point_appearance_queries,))
            for track_idx in range(n_tracks)[:n_non_first_point_appearance_queries]:
                # Randomly take a timestep where the point is visible
                non_zero_timesteps = torch.nonzero(gt_visibilities_any_view[:, track_idx] == 1)
                random_timestep = non_zero_timesteps[rnd_np.randint(len(non_zero_timesteps))].item()
                non_first_point_appearances[track_idx] = random_timestep

            query_points_t = torch.cat([non_first_point_appearances, first_point_appearances], dim=0)
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
                    traj_object_id=traj_object_id.numpy(),
                    visibility=visibility.numpy(),
                    valids=valids.numpy(),
                    query_points=query_points.numpy(),
                )

        # Normalize the scene to be similar to Kubric's scene
        scale = 6
        rot_x = R.from_euler('x', 220, degrees=True).as_matrix()
        rot_y = R.from_euler('y', 3, degrees=True).as_matrix()
        rot_z = R.from_euler('z', -30, degrees=True).as_matrix()
        rot = torch.from_numpy(rot_z @ rot_y @ rot_x)
        translation = torch.tensor([0.0, 0.0, 0.5], dtype=torch.float32)
        (
            depths_trans, extrs_trans, query_points_trans, traj3d_world_trans, traj2d_w_z_trans
        ) = transform_scene(scale, rot, translation, depths, extrs, query_points, traj3d_world, traj2d_w_z)
        novel_extrs_trans = transform_scene(scale, rot, translation, None, novel_extrs, None, None, None)[1]

        # rerun_viz_scene("nane/scene__no_transform/", rgbs, depths, intrs, extrs, traj3d_world, 0.1)
        # rerun_viz_scene("nane/scene_transformed/", rgbs, depths_trans, intrs, extrs_trans, traj3d_world_trans, 1)

        # # Use the auto scene normalization of generic scenes
        # from mvtracker.datasets.generic_scene_dataset import compute_auto_scene_normalization
        # scale, rot, translation = compute_auto_scene_normalization(depths, torch.ones_like(depths) * 100, extrs_trans, intrs)
        # scale = scale * T[0, 0].item()
        # print(f"{scale=}")
        # (depths_trans, extrs_trans, query_points_trans, traj3d_world_trans, traj2d_w_z_trans
        # ) = transform_scene(scale, rot, translation, depths_trans, extrs_trans, query_points_trans, traj3d_world_trans, traj2d_w_z_trans)
        # _, novel_extrs_trans, _, _, _ = transform_scene(scale, rot, translation, None, novel_extrs_trans, None, None, None)
        # 82.7 91.1 --> 80.8 89.1

        segs = torch.ones((n_frames, 1, h, w))  # Dummy segmentation masks

        # Run SAM3D inference if enabled
        sam3d_joints_world = None
        if self.use_sam3d and self.sam3d_wrapper is not None:
            logging.info(f"Running SAM3D inference on {self.seq_names[index]}...")
            try:
                # Run SAM3D on the sequence (before scene transformation)
                sam3d_joints_world = self.sam3d_wrapper.process_multiview_sequence(
                    rgbs=rgbs,  # (V, T, 3, H, W)
                    intrs=intrs,  # (V, T, 3, 3)
                    extrs=extrs,  # (V, T, 3, 4)
                    inference_type="body",
                    bbox_thr=0.5,
                )

                if sam3d_joints_world is not None:
                    # Apply same scene transformation as trajectories
                    # sam3d_joints_world is [T, n_persons, 70, 3]
                    T_sam, P_sam, K_sam = sam3d_joints_world.shape[:3]
                    sam3d_flat = sam3d_joints_world.reshape(T_sam * P_sam * K_sam, 3)

                    # Apply scale, rotation, translation
                    sam3d_flat_trans = sam3d_flat * scale
                    sam3d_flat_trans = torch.einsum('ij,nj->ni', rot.float(), sam3d_flat_trans)
                    sam3d_flat_trans = sam3d_flat_trans + translation

                    sam3d_joints_world = sam3d_flat_trans.reshape(T_sam, P_sam, K_sam, 3)
                    logging.info(f"SAM3D inference succeeded: {P_sam} persons detected")
                else:
                    logging.warning(f"SAM3D inference returned no detections for {self.seq_names[index]}")
            except Exception as e:
                logging.error(f"SAM3D inference failed on {self.seq_names[index]}: {e}")
                sam3d_joints_world = None

        datapoint = Datapoint(
            video=rgbs,
            videodepth=depths_trans,
            videodepthconf=depth_confs.float() if depth_confs is not None else None,
            feats=feats,
            segmentation=segs,
            trajectory=traj2d_w_z_trans,
            trajectory_3d=traj3d_world_trans,
            trajectory_category=traj_object_id,
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
            sam3d_joints_world=sam3d_joints_world,
        )
        return datapoint


def rerun_viz_scene(entity_prefix, rgbs, depths, intrs, extrs, tracks, radii_scale,
                    viz_camera=False, viz_point_cloud=True, fps=12):
    import rerun as rr

    # Initialize Rerun
    rr.init(f"3dpt", recording_id="v0.16")
    rr.connect_tcp()

    V, T, _, H, W = rgbs.shape
    _, N, _ = tracks.shape
    assert rgbs.shape == (V, T, 3, H, W)
    assert depths.shape == (V, T, 1, H, W)
    assert intrs.shape == (V, T, 3, 3)
    assert extrs.shape == (V, T, 3, 4)
    assert tracks.shape == (T, N, 3)

    # Compute inverse intrinsics and extrinsics
    intrs_inv = torch.inverse(intrs.float()).type(intrs.dtype)
    extrs_square = torch.eye(4).to(extrs.device).repeat(V, T, 1, 1)
    extrs_square[:, :, :3, :] = extrs
    extrs_inv = torch.inverse(extrs_square.float()).type(extrs.dtype)
    assert intrs_inv.shape == (V, T, 3, 3)
    assert extrs_inv.shape == (V, T, 4, 4)

    for v in range(V):  # Iterate over views
        for t in range(T):  # Iterate over frames
            rr.set_time_seconds("frame", t / fps)

            # Log RGB image
            rgb_image = rgbs[v, t].permute(1, 2, 0).cpu().numpy()
            if viz_camera:
                rr.log(f"{entity_prefix}image/view-{v}/rgb", rr.Image(rgb_image))

            # Log Depth map
            depth_map = depths[v, t, 0].cpu().numpy()
            if viz_camera:
                rr.log(f"{entity_prefix}image/view-{v}/depth", rr.DepthImage(depth_map, point_fill_ratio=0.2))

            # Log Camera
            K = intrs[v, t].cpu().numpy()
            world_T_cam = np.eye(4)
            world_T_cam[:3, :3] = extrs_inv[v, t, :3, :3].cpu().numpy()
            world_T_cam[:3, 3] = extrs_inv[v, t, :3, 3].cpu().numpy()
            if viz_camera:
                rr.log(f"{entity_prefix}image/view-{v}", rr.Pinhole(image_from_camera=K, width=W, height=H))
                rr.log(f"{entity_prefix}image/view-{v}",
                       rr.Transform3D(translation=world_T_cam[:3, 3], mat3x3=world_T_cam[:3, :3]))

            # Generate and log point cloud colored by RGB values
            # Compute 3D points from depth map
            y, x = np.indices((H, W))
            homo_pixel_coords = np.stack([x.ravel(), y.ravel(), np.ones_like(x).ravel()], axis=1).T
            depth_values = depth_map.ravel()
            cam_coords = (intrs_inv[v, t].cpu().numpy() @ homo_pixel_coords) * depth_values
            cam_coords = np.vstack((cam_coords, np.ones((1, cam_coords.shape[1]))))
            world_coords = (world_T_cam @ cam_coords)[:3].T

            # Filter out points with zero depth
            valid_mask = depth_values > 0
            world_coords = world_coords[valid_mask]
            rgb_colors = rgb_image.reshape(-1, 3)[valid_mask].astype(np.uint8)

            # Log the point cloud
            if viz_point_cloud:
                rr.log(f"{entity_prefix}point_cloud/view-{v}",
                       rr.Points3D(world_coords, colors=rgb_colors, radii=0.02 * radii_scale))

    # Log 3D tracks
    x = tracks[0, :, 0]
    c = (x - x.min()) / (x.max() - x.min() + 1e-8)
    colors = (matplotlib.colormaps["gist_rainbow"](c)[:, :3] * 255).astype(np.uint8)
    for t in range(T):
        rr.set_time_seconds("frame", t / fps)
        rr.log(
            f"{entity_prefix}tracks/points",
            rr.Points3D(positions=tracks[t], colors=colors, radii=0.01 * radii_scale),
        )
        if t > 0:
            strips = np.concatenate(
                [np.stack([tracks[:t, n], tracks[1:t + 1, n]], axis=-2) for n in range(N)],
                axis=0,
            )
            strip_colors = np.concatenate(
                [np.repeat(colors[n][None], t, axis=0) for n in range(N)],
                axis=0,
            )
            rr.log(
                f"{entity_prefix}tracks/lines",
                rr.LineStrips3D(strips=strips, colors=strip_colors, radii=0.005 * radii_scale),
            )
