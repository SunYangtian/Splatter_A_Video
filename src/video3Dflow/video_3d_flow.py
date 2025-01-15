"""
Given depth / tracking results, 
merge them into a temporal point cloud sequence [N,T,3].
"""

import os, glob
from functools import partial
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import imageio
import cv2
import torch

from video3Dflow.utils import median_filter_2d, get_tracks_3d_for_query_frame

@dataclass
class Video3DFlow:
    depth_dir : str
    tracks_dir : str
    img_dir : str
    mask_dir : str
    start: int = 0
    end: int = -1
    mask_erosion_radius: int = 3
    depth_range_min: float = 0.5
    depth_range_max: float = 2.0

    def setup(self):
        self.depth_files = sorted(glob.glob(os.path.join(self.depth_dir, "*.npy")))
        self.tracking_files = sorted(glob.glob(os.path.join(self.tracks_dir, "*.npy")))
        frame_names = [os.path.splitext(p)[0] for p in sorted(os.listdir(self.img_dir))]
        self.img_ext = os.path.splitext(os.listdir(self.img_dir)[0])[1]

        if self.end == -1:
            self.end = len(frame_names)
        self.frame_names = frame_names[self.start:self.end]

        self.imgs: list[torch.Tensor | None] = [None for _ in self.frame_names]
        self.depths: list[torch.Tensor | None] = [None for _ in self.frame_names]
        self.masks: list[torch.Tensor | None] = [None for _ in self.frame_names]

        self.scale = 1
        self.extract_fg = True


    def get_tracks_3d(
        self, num_samples: int, start: int = 0, end: int = -1, step: int = 1, **kwargs
    ):
        num_frames = len(self.imgs)
        if end < 0:
            end = num_frames + 1 + end
        query_idcs = list(range(start, end, step))
        target_idcs = list(range(start, end, step))
        masks = torch.stack([self.get_mask(i) for i in target_idcs], dim=0)
        mask_val = 1 if self.extract_fg else -1
        fg_masks = (masks == mask_val).float()

        depths = torch.stack([self.get_depth(i) for i in target_idcs], dim=0)
        ### scale to [0.5 ~ 2.0]
        range_min, range_max = self.depth_range_min, self.depth_range_max
        self.depths_min, self.depths_max = depths.min(), depths.max()
        depths = (depths - self.depths_min) / (self.depths_max - self.depths_min) * (range_max - range_min) + range_min

        num_per_query_frame = int(np.ceil(num_samples / len(query_idcs)))
        cur_num = 0
        tracks_all_queries = []
        # from each query to all targets
        for q_idx in query_idcs:
            # (N, T, 4)
            tracks_2d = self.load_target_tracks(q_idx, target_idcs)
            num_sel = int(
                min(num_per_query_frame, num_samples - cur_num, len(tracks_2d))
            )
            if num_sel < len(tracks_2d):
                sel_idcs = np.random.choice(len(tracks_2d), num_sel, replace=False)
                tracks_2d = tracks_2d[sel_idcs]

            cur_num += tracks_2d.shape[0]
            img = self.get_image(q_idx)
            tidx = target_idcs.index(q_idx)  # index in target_idcs

            # consider tracks_3d, colors, visibles, invisibles, confidences
            tracks_tuple = get_tracks_3d_for_query_frame(
                tidx, img, tracks_2d, depths, fg_masks, extract_fg=self.extract_fg
            )
            tracks_all_queries.append(tracks_tuple)

        tracks_3d, colors, visibles, invisibles, confidences = map(
            partial(torch.cat, dim=0), zip(*tracks_all_queries)
        )

        return tracks_3d, visibles, invisibles, confidences, colors
    

    def load_target_tracks(
        self, query_index: int, target_indices: list[int], dim: int = 1
    ):
        """
        tracks are 2d, occs and uncertainties
        :param dim (int), default 1: dimension to stack the time axis
        return (N, T, 4) if dim=1, (T, N, 4) if dim=0
        """
        q_name = self.frame_names[query_index]
        all_tracks = []
        for ti in target_indices:
            t_name = self.frame_names[ti]
            path = f"{self.tracks_dir}/{q_name}_{t_name}.npy"
            tracks = np.load(path).astype(np.float32)
            all_tracks.append(tracks)
        return torch.from_numpy(np.stack(all_tracks, axis=dim))
    

    def get_image(self, index) -> torch.Tensor:
        if self.imgs[index] is None:
            self.imgs[index] = self.load_image(index)
        img = cast(torch.Tensor, self.imgs[index])
        return img
    
    def load_image(self, index) -> torch.Tensor:
        path = f"{self.img_dir}/{self.frame_names[index]}{self.img_ext}"
        return torch.from_numpy(imageio.imread(path)).float() / 255.0


    def get_depth(self, index) -> torch.Tensor:
        if self.depths[index] is None:
            self.depths[index] = self.load_depth(index)
        return self.depths[index] / self.scale
    
    def load_depth(self, index) -> torch.Tensor:
        path = f"{self.depth_dir}/{self.frame_names[index]}.npy"
        disp = np.load(path)
        depth = 1.0 / np.clip(disp, a_min=1e-6, a_max=1e6)
        depth = torch.from_numpy(depth).float()
        depth = median_filter_2d(depth[None, None], 11, 1)[0, 0]
        return depth
    
    def get_mask(self, index) -> torch.Tensor:
        if self.masks[index] is None:
            self.masks[index] = self.load_mask(index)
        mask = cast(torch.Tensor, self.masks[index])
        return mask
    
    def load_mask(self, index) -> torch.Tensor:
        path = f"{self.mask_dir}/{self.frame_names[index]}.png"
        r = self.mask_erosion_radius
        mask = imageio.imread(path)
        fg_mask = mask.reshape((*mask.shape[:2], -1)).max(axis=-1) > 0
        bg_mask = ~fg_mask
        fg_mask_erode = cv2.erode(
            fg_mask.astype(np.uint8), np.ones((r, r), np.uint8), iterations=1
        )
        bg_mask_erode = cv2.erode(
            bg_mask.astype(np.uint8), np.ones((r, r), np.uint8), iterations=1
        )
        out_mask = np.zeros_like(fg_mask, dtype=np.float32)
        out_mask[bg_mask_erode > 0] = -1
        out_mask[fg_mask_erode > 0] = 1
        return torch.from_numpy(out_mask).float()

    
    # def extend_track3d(self, track3d, grid_size=16, margin=0.5):
    def extend_track3d(self, track3d, grid_size=64, margin=0.25):
        # track3d: (N, T, 3)
        # extend with the first frame and last frame
        H, W = self.depths[0].shape
        image_size_wh = torch.tensor([W, H], device=track3d.device)[None]
        points_3d_seq, points_colors = [], []

        #################### add left frame
        pixel_coordinates = torch.meshgrid(
            torch.linspace(0, int((W - 1)*margin), W//grid_size), 
            torch.linspace(0, H - 1, H//int(grid_size*margin))
        )
        pixel_coordinates = torch.stack(pixel_coordinates, dim=-1).reshape(-1, 2)
        points_2d = (pixel_coordinates - image_size_wh / 2) / (image_size_wh / 2)

        # (T, 1, H, W), (T, 1, N, 2) -> (T, 1, 1, N)
        point_depths = torch.nn.functional.grid_sample(
            self.get_depth(0)[None,None],
            points_2d[None,None],
            align_corners=True,
            padding_mode="border",
        )[0, 0, 0]
        point_depths = (point_depths - self.depths_min) / (self.depths_max - self.depths_min) * (self.depth_range_max - self.depth_range_min) + self.depth_range_min

        cur_points_colors = torch.nn.functional.grid_sample(
            self.get_image(0)[None].permute(0, 3, 1, 2),
            points_2d[None,None],
            align_corners=True,
            padding_mode="border",
        )[0, :, 0].permute(1,0)

        is_in_masks = (torch.nn.functional.grid_sample(
                self.get_mask(0)[None, None],
                points_2d[None,None],
                align_corners=True,
            )[0, 0, 0]
            == 1
        )
        valid = torch.logical_not(is_in_masks)

        points_3d = torch.cat([points_2d[valid], point_depths[valid][..., None]], dim=-1)
        delta_pos = track3d - track3d[:, 0:1]
        points_3d_seq += [points_3d[:,None] + delta_pos.mean(dim=0, keepdim=True)]
        points_colors += [cur_points_colors[valid]]

        #################### add right frame
        pixel_coordinates = torch.meshgrid(
            torch.linspace(int((W - 1)*(1-margin)), W - 1, W//grid_size), 
            torch.linspace(0, H - 1, H//int(grid_size*margin))
        )
        pixel_coordinates = torch.stack(pixel_coordinates, dim=-1).reshape(-1, 2)
        points_2d = (pixel_coordinates - image_size_wh / 2) / (image_size_wh / 2)

        num_frames = track3d.shape[1]
        point_depths = torch.nn.functional.grid_sample(
            self.get_depth(num_frames-1)[None,None],
            points_2d[None,None],
            align_corners=True,
            padding_mode="border",
        )[0, 0, 0]
        point_depths = (point_depths - self.depths_min) / (self.depths_max - self.depths_min) * (self.depth_range_max - self.depth_range_min) + self.depth_range_min

        cur_points_colors = torch.nn.functional.grid_sample(
            self.get_image(num_frames-1)[None].permute(0, 3, 1, 2),
            points_2d[None,None],
            align_corners=True,
            padding_mode="border",
        )[0, :, 0].permute(1,0)

        is_in_masks = (torch.nn.functional.grid_sample(
                self.get_mask(num_frames-1)[None, None],
                points_2d[None,None],
                align_corners=True,
            )[0, 0, 0]
            == 1
        )
        valid = torch.logical_not(is_in_masks)

        points_3d = torch.cat([points_2d[valid], point_depths[valid][..., None]], dim=-1)
        delta_pos = track3d - track3d[:, -1:]
        points_3d_seq += [points_3d[:,None] + delta_pos.mean(dim=0, keepdim=True)]
        points_colors += [cur_points_colors[valid]]


        return torch.cat(points_3d_seq, dim=0), torch.cat(points_colors, dim=0)