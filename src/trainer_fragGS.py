import glob
import os
import pdb
import time
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import util
from criterion import masked_mse_loss, masked_l1_loss, compute_depth_range_loss, lossfun_distortion
from pointrix.model.loss import l1_loss, ssim
from kornia import morphology as morph
# from gs import GSTrainer
# from frag_gs import GSTrainer
from pointrix.utils.config import load_config, parse_structured
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Union, List
from pointrix.camera.cam_utils import construct_canonical_camera, construct_canonical_camera_from_focal
from pointrix.optimizer import parse_optimizer, parse_scheduler
from pointrix.renderer import parse_renderer
from frag_model import FragModel
import dptr.gs as gs
from tqdm import tqdm
from pytorch3d.renderer import look_at_rotation
import matplotlib.pyplot as plt
from video3Dflow.utils import parse_tapir_track_info


torch.manual_seed(1234)


def init_weights(m):
    # Initializes weights according to the DCGAN paper
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


def de_parallel(model):
    return model.module if hasattr(model, 'module') else model

def entropy_loss(opacity, gs_idx):
    """
    Entropy loss for the features.

    Parameters
    ----------
    opacity : torch.Tensor [N,1]
        The opacity values.
    gs_idx : torch.Tensor [B,H,W,K]
        The first K gaussian index for each pixel.
    
    Returns
    -------
    torch.Tensor
        The entropy loss.
    """
    # mask = gs_idx > -1  # [H,W,K]
    bs, H, W, K = gs_idx.shape
    ### use 1 opacity to omit gs_idx=-1
    opacity = torch.cat([opacity, torch.ones_like(opacity[-1:])], dim=0).squeeze(1)  # [N+1,]
    gs_idx[gs_idx == -1] = opacity.shape[0] - 1
    opacity = opacity.unsqueeze(0).repeat(bs, 1)  # [B,N+1,]
    flatten_gs_idx = gs_idx.reshape(bs,-1).long()  # [bs, H*W*K]
    pixel_opacity = torch.gather(opacity, dim=1, index=flatten_gs_idx)  # [bs, H*W*K]
    pixel_opacity = pixel_opacity.reshape(bs, H, W, K)  # [bs, H, W, K]
    # import matplotlib.pyplot as plt
    # plt.hist(pixel_opacity.detach().cpu().numpy().reshape(-1,1))
    pixel_opacity = pixel_opacity / (pixel_opacity.sum(dim=-1, keepdim=True) + 1e-8)
    pixel_entropy = -torch.sum(pixel_opacity * torch.log(pixel_opacity), dim=-1)  # [bs,]
    return pixel_entropy.mean()

    pixel_alpha = torch.cumprod(1 - pixel_opacity, dim=-1)
    pixel_transparency = torch.cat([torch.ones_like(pixel_alpha[..., :1]), pixel_alpha[..., :-1]], dim=-1)
    pixel_weight = pixel_transparency * pixel_opacity + 1e-5
    pixel_entropy = -torch.sum(pixel_weight * torch.log(pixel_weight), dim=-1)  # [bs,]
    return pixel_weight, pixel_entropy.mean()



def alpha_blending_firstK(attribute, gs_idx, pixel_weight, bg=1):
    """
    A function approximates alpha-blending.
    Attribute: [N,D]
    """
    ### query attribute for each pixel
    bs, H, W, K = gs_idx.shape
    gs_idx[gs_idx == -1] = attribute.shape[0] - 1
    flatten_gs_idx = gs_idx.reshape(bs,-1).long()  # [bs, H*W*K]
    attribute = torch.cat([attribute, bg*torch.ones_like(attribute[-1:])], dim=0)  # [N+1,D]
    attribute = attribute.unsqueeze(0).repeat(bs, 1, 1)  # [bs, N+1, D]
    flatten_gs_idx = flatten_gs_idx.unsqueeze(-1).repeat(1,1,attribute.shape[-1])  # [bs, H*W*K, D]
    pixel_attribute = torch.gather(attribute, dim=1, index=flatten_gs_idx)  # [bs, H*W*K, D]
    pixel_attribute = pixel_attribute.reshape(bs, H, W, K, -1)  # [bs, H, W, K, D]
    ### alpha-blending
    pixel_render = torch.sum(pixel_attribute * pixel_weight.unsqueeze(-1), dim=-2)  # [bs, H, W, D]
    return pixel_render


class FragTrainer:
    @dataclass
    class Config:
        # Modules
        model: dict = field(default_factory=dict)
        optimizer: dict = field(default_factory=dict)
        renderer: dict = field(default_factory=dict)
        scheduler: Optional[dict] = field(default_factory=dict)
        writer: dict = field(default_factory=dict)
        hooks: dict = field(default_factory=dict)
        render_attributes: dict = field(default_factory=dict)
        # Dataset
        dataset_name: str = "NeRFDataset"
        dataset: dict = field(default_factory=dict)

        # Training config
        batch_size: int = 1
        num_workers: int = 0
        max_steps: int = 30000
        val_interval: int = 2000
        spatial_lr_scale: bool = True

        # Progress bar
        bar_upd_interval: int = 10
        # Output path
        output_path: str = "output"

    cfg: Config

    def __init__(self, args, device='cuda'):
        self.args = args
        self.device = device

        # self.read_data()
        self.read_data_simple()
        
        ########### add canonical space GS. This contains the GS optimizer

        cfg = load_config(args.gs_config_file)
        cfg.trainer.dataset.data_path = self.img_files[0]
        self.cfg = parse_structured(self.Config, cfg.trainer)

        # TODO define the gs atlases here !
        ### define the gs atlas config
        @dataclass
        class GSAtlasCFG:
            name: str
            num_images: int
            start_frame_path: str
            end_frame_path: str
            start_frame_mask_path: str
            start_depth_npy: str
            end_frame_mask_path: str
            start_frame_id: int
            end_frame_id: int
            reverse_mask: bool = False
            render_attributes: dict = field(default_factory=dict)
            is_fg: bool = True

        gs_atlas_cfg1 = GSAtlasCFG(
            name='gs_base',
            num_images=self.num_imgs,
            start_frame_path='test_data/bear/color/00000.jpg',
            end_frame_path=None,
            start_frame_mask_path='test_data/bear/masks/00000.png',
            start_depth_npy='test_data/bear/marigold/depth_npy/00000_pred.npy',
            end_frame_mask_path=None,
            start_frame_id=0,
            end_frame_id=50,
            render_attributes=self.cfg.render_attributes
        )

        gs_atlas_cfg2 = GSAtlasCFG(
            name='gs_fg',
            num_images=self.num_imgs,
            start_frame_path='test_data/bear/color/00000.jpg',
            end_frame_path=None,
            start_frame_mask_path='test_data/bear/masks/00000.png',
            start_depth_npy='test_data/bear/marigold/depth_npy/00000_pred.npy',
            end_frame_mask_path=None,
            start_frame_id=0,
            end_frame_id=50,
            render_attributes=self.cfg.render_attributes,
            is_fg = True
        )

        gs_atlas_cfg3 = GSAtlasCFG(
            name='gs_bg',
            num_images=self.num_imgs,
            start_frame_path='test_data/bear/color/00000.jpg',
            end_frame_path=None,
            start_frame_mask_path='test_data/bear/masks/00000.png',
            start_depth_npy='test_data/bear/marigold/depth_npy/00000_pred.npy',
            end_frame_mask_path=None,
            start_frame_id=0,
            end_frame_id=50,
            render_attributes=self.cfg.render_attributes,
            reverse_mask=True,
            is_fg=False
        )

        # self.gs_atlas_cfg_list = [gs_atlas_cfg1, gs_atlas_cfg2, gs_atlas_cfg3]
        # self.gs_atlas_cfg_list = [gs_atlas_cfg2, gs_atlas_cfg3]
        self.gs_atlas_cfg_list = [gs_atlas_cfg1]
        base_point_seq_list = [self.tracks_fg_info['tracks_3d'].permute(1,0,2).to(self.device),
                               self.tracks_bg_info['tracks_3d'].permute(1,0,2).to(self.device)]
        # TODO define GSTrainer with __init__(self, cfg, *gs_atlas_cfg_list)
        # Rewrite. This should be a model, instead of a trainer !!!
        ### 1. initialize gs model / optimizer for each atlas
        self.cfg.model.pop('name')
        self.white_bg = self.cfg.dataset.white_bg
        self.gs_atlases_model = FragModel(self.cfg.model, self.gs_atlas_cfg_list, base_point_seq_list, white_bg=self.white_bg)
        ### 2. set fixed camera parameter
        self.construct_render_dict(self.h, self.w, self.gs_atlases_model.focal_y_ratio)
        ### 3. renderer
        self.enable_ortho_projection = True if 'Ortho' in self.cfg.renderer.name else False
        self.renderer = parse_renderer(
            self.cfg.renderer, white_bg=self.white_bg, device=device)
        ### 4. optimizer and scheduler
        cameras_extent = 5  # the extent of the camera
        for gs_atlas_cfg in self.gs_atlas_cfg_list:
            name = gs_atlas_cfg.name
            optimizer_config = self.cfg.optimizer.copy()
            if name == 'gs_base':
                # optimizer_config["optimizer_1"]["extra_cfg"]["densify_start_iter"] = 100000
                # optimizer_config["optimizer_1"]["extra_cfg"]["opacity_reset_interval"] = 100000
                pass
            setattr(self, name+"_optimizer", 
                    parse_optimizer(optimizer_config, 
                                    model=self.gs_atlases_model.get_atlas(name),
                                    cameras_extent=cameras_extent)
                    )
            setattr(self, name+"_scheduler",
                    parse_scheduler(self.cfg.scheduler, 
                                    cameras_extent if self.cfg.spatial_lr_scale else 1.)
                    )
            
        # seq_name = os.path.basename(args.data_dir.rstrip('/'))
        self.out_dir = os.path.join(args.save_dir, '{}_{}'.format(args.expname, self.seq_name))
        self.step = self.load_from_ckpt(self.out_dir,
                                        load_opt=self.args.load_opt,
                                        load_scheduler=self.args.load_scheduler)
        self.time_steps = torch.linspace(1, self.num_imgs, self.num_imgs, device=self.device)[:, None] / self.num_imgs

        #### for detph loss
        from loss import ScaleAndShiftInvariantLoss
        self.depth_loss = ScaleAndShiftInvariantLoss(alpha=0.5, scales=1)

    
    def read_data_simple(self):
        # self.seq_dir = 
        self.img_dir = "/mnt/sda/syt/dataset/laptop_10211/processed/images"
        self.seq_name = self.args.seq_name
        img_files = sorted(glob.glob(os.path.join(self.img_dir, '*')))
        if self.args.num_imgs < 0:
            self.num_imgs = len(img_files)
        else:
            self.num_imgs = min(self.args.num_imgs, len(img_files))

        self.base_idx = self.args.base_idx
        self.img_files = img_files[self.base_idx:self.num_imgs+self.base_idx]

        t1 = time.time()
        images = np.array([imageio.imread(img_file) / 255. for img_file in self.img_files])
        self.images = torch.from_numpy(images).float()  # [n_imgs, h, w, 3]
        # image = np.array(imageio.imread(self.img_files[0])) / 255.
        # self.images = torch.from_numpy(image).float()[None].repeat(self.num_imgs,1,1,1)  # [n_imgs, h, w, 3]
        self.h, self.w = self.images.shape[1:3]
        print(f'Read images time: {time.time() - t1} s')


        ### load depth
        if True:
        #     self.depth_dir = "/mnt/sda/syt/dataset/laptop_10211/processed/aligned_depth_anything_v2"
        #     depth_files = sorted(glob.glob(os.path.join(self.depth_dir, '*.npy')))[self.base_idx:self.num_imgs+self.base_idx]
        #     depth = [np.load(depth_file) for depth_file in depth_files]
        #     depth = [1.0 / np.clip(x, a_min=1e-6, a_max=1e6) for x in depth]
        #     self.gt_depths = torch.from_numpy(np.array(depth)).float().to(self.device)  # [n_imgs, h, w]
            self.depth_dir = "/mnt/sda/syt/dataset/laptop_10211/processed/marigold/depth_npy"
            depth_files = sorted(glob.glob(os.path.join(self.depth_dir, '*.npy')))[self.base_idx:self.num_imgs+self.base_idx]
            depth = np.array([np.load(depth_file) for depth_file in depth_files])
            self.gt_depths = torch.from_numpy(depth).float().to(self.device)  # [n_imgs, h, w]

        ### load mask
        if True:
            self.mask_dir = "/mnt/sda/syt/dataset/laptop_10211/processed/masks"
            mask_files = sorted(glob.glob(os.path.join(self.mask_dir, '*.png')))[self.base_idx:self.num_imgs+self.base_idx]
            masks = np.array([imageio.imread(mask_file)/255. for mask_file in mask_files])
            if masks.ndim == 4:
                masks = masks.sum(axis=-1) > 0
            self.masks = torch.from_numpy(masks).float().to(self.device)

        self.grid = util.gen_grid(self.h, self.w, device=self.device, normalize=False, homogeneous=True).float()

        ### load 3D flow
        depth_folder = "/mnt/sda/syt/dataset/laptop_10211/processed/aligned_depth_anything_v2"
        tracking_folder = "/mnt/sda/syt/dataset/laptop_10211/processed/bootstapir"
        frames_folder = "/mnt/sda/syt/dataset/laptop_10211/processed/images"
        mask_folder = "/mnt/sda/syt/dataset/laptop_10211/processed/masks"
        from video3Dflow.video_3d_flow import Video3DFlow
        self.video_3d_flow = Video3DFlow(depth_folder, tracking_folder, frames_folder, mask_folder)
        self.video_3d_flow.setup()
        tracks_3d, visibles, invisibles, confidences, colors = self.video_3d_flow.get_tracks_3d(
            num_samples=10000, 
            start=self.base_idx, 
            end=self.num_imgs+self.base_idx, 
            step=1, )
        self.tracks_fg_info = {
            "tracks_3d": tracks_3d,  # [N, T, 3]
            "visibles": visibles,  # [N, T]
            "invisibles": invisibles,
            "confidences": confidences,
            "colors": colors
        }
        ##### extract bg tracks
        self.video_3d_flow.extract_fg = False
        tracks_3d, visibles, invisibles, confidences, colors = self.video_3d_flow.get_tracks_3d(
            num_samples=10000, 
            start=self.base_idx, 
            end=self.num_imgs+self.base_idx, 
            step=1, )
        grid_size = int(64 / (self.args.video_flow_margin / 0.25))
        extended_tracks_3d, extended_colors = self.video_3d_flow.extend_track3d(tracks_3d, margin=self.args.video_flow_margin, grid_size=grid_size)
        # extended_tracks_3d, extended_colors = self.video_3d_flow.extend_track3d(tracks_3d, margin=0.75)
        tracks_3d = torch.cat([extended_tracks_3d, tracks_3d], dim=0)
        colors = torch.cat([extended_colors, colors], dim=0)
        self.tracks_bg_info = {
            "tracks_3d": tracks_3d,  # [N, T, 3]
            "visibles": visibles,  # [N, T]
            "invisibles": invisibles,
            "confidences": confidences,
            "colors": colors
        }


    def read_data(self):
        # self.seq_dir = self.args.data_dir
        # self.seq_name = os.path.basename(self.seq_dir.rstrip('/'))
        # self.img_dir = os.path.join(self.seq_dir, 'color')
        self.seq_dir = self.args.data_dir
        self.seq_name = self.args.seq_name
        self.img_dir = os.path.join(self.seq_dir, 'JPEGImages/480p', self.seq_name)

        img_files = sorted(glob.glob(os.path.join(self.img_dir, '*')))
        if self.args.num_imgs < 0:
            self.num_imgs = len(img_files)
        else:
            self.num_imgs = min(self.args.num_imgs, len(img_files))
        self.base_idx = self.args.base_idx
        self.img_files = img_files[self.base_idx:self.num_imgs+self.base_idx]

        t1 = time.time()
        images = np.array([imageio.imread(img_file) / 255. for img_file in self.img_files])
        self.images = torch.from_numpy(images).float()  # [n_imgs, h, w, 3]
        # image = np.array(imageio.imread(self.img_files[0])) / 255.
        # self.images = torch.from_numpy(image).float()[None].repeat(self.num_imgs,1,1,1)  # [n_imgs, h, w, 3]
        self.h, self.w = self.images.shape[1:3]
        print(f'Read images time: {time.time() - t1} s')

        ### load depth
        if True:
            self.depth_dir = os.path.join(self.seq_dir, 'marigold/480p', self.seq_name, 'depth_npy')
            depth_files = sorted(glob.glob(os.path.join(self.depth_dir, '*.npy')))[self.base_idx:self.num_imgs+self.base_idx]
            depth = np.array([np.load(depth_file) for depth_file in depth_files])
            self.gt_depths = torch.from_numpy(depth).float().to(self.device)  # [n_imgs, h, w]

        ### load mask
        if True:
            self.mask_dir = os.path.join(self.seq_dir, 'Annotations/480p', self.seq_name)
            mask_files = sorted(glob.glob(os.path.join(self.mask_dir, '*.png')))[self.base_idx:self.num_imgs+self.base_idx]
            masks = np.array([imageio.imread(mask_file)/255. for mask_file in mask_files])
            if masks.ndim == 4:
                masks = masks.sum(axis=-1) > 0
            self.masks = torch.from_numpy(masks).float().to(self.device)

        ### load dino feature
        if False:
            self.dino_dir = os.path.join(self.seq_dir, 'dinov2')
            # self.dino_dir = os.path.join(self.seq_dir, 'sam_feature')
            dino_files = sorted(glob.glob(os.path.join(self.dino_dir, '*.png')))[self.base_idx:self.num_imgs+self.base_idx]
            dino_files += sorted(glob.glob(os.path.join(self.dino_dir, '*.jpg')))[self.base_idx:self.num_imgs+self.base_idx]
            dinos = np.array([imageio.imread(dino_file)/255. for dino_file in dino_files])
            self.dinos = torch.from_numpy(dinos).float().to(self.device)

        self.grid = util.gen_grid(self.h, self.w, device=self.device, normalize=False, homogeneous=True).float()

        ### load 3D flow
        depth_folder = f"/mnt/sda/syt/dataset/DAVIS_processed/aligned_depth_anything_v2/480p/{self.seq_name}"
        tracking_folder = f"/mnt/sda/syt/dataset/DAVIS_processed/bootstapir_v2/480p/{self.seq_name}"
        frames_folder = f"/mnt/sda/syt/dataset/DAVIS/JPEGImages/480p/{self.seq_name}"
        mask_folder = f"/mnt/sda/syt/dataset/DAVIS_processed/Annotations/480p/{self.seq_name}"
        from video3Dflow.video_3d_flow import Video3DFlow
        self.video_3d_flow = Video3DFlow(depth_folder, tracking_folder, frames_folder, mask_folder)
        self.video_3d_flow.setup()
        tracks_3d, visibles, invisibles, confidences, colors = self.video_3d_flow.get_tracks_3d(
            num_samples=10000, 
            start=self.base_idx, 
            end=self.num_imgs+self.base_idx, 
            step=1, )
        self.tracks_fg_info = {
            "tracks_3d": tracks_3d,  # [N, T, 3]
            "visibles": visibles,  # [N, T]
            "invisibles": invisibles,
            "confidences": confidences,
            "colors": colors
        }
        ##### extract bg tracks
        self.video_3d_flow.extract_fg = False
        tracks_3d, visibles, invisibles, confidences, colors = self.video_3d_flow.get_tracks_3d(
            num_samples=10000, 
            start=self.base_idx, 
            end=self.num_imgs+self.base_idx, 
            step=1, )
        grid_size = int(64 / (self.args.video_flow_margin / 0.25))
        extended_tracks_3d, extended_colors = self.video_3d_flow.extend_track3d(tracks_3d, margin=self.args.video_flow_margin, grid_size=grid_size)
        # extended_tracks_3d, extended_colors = self.video_3d_flow.extend_track3d(tracks_3d, margin=0.75)
        tracks_3d = torch.cat([extended_tracks_3d, tracks_3d], dim=0)
        colors = torch.cat([extended_colors, colors], dim=0)
        self.tracks_bg_info = {
            "tracks_3d": tracks_3d,  # [N, T, 3]
            "visibles": visibles,  # [N, T]
            "invisibles": invisibles,
            "confidences": confidences,
            "colors": colors
        }


    def construct_render_dict(self, h, w, focal_y_ratio=None):
        """
        define the fixed camera

        Parameters
        ----------
        h : int
            The height of the image.
        w : int
            The width of the image.
        """
        if focal_y_ratio is not None:
            focal = focal_y_ratio * h
            camera = construct_canonical_camera_from_focal(width=w, height=h, focal=focal)
        else:
            camera = construct_canonical_camera(width=w, height=h)

        self.batch_dict = {
            "camera": camera,
            "FovX": camera.fovX,
            "FovY": camera.fovY,
            "height": int(camera.image_height),
            "width": int(camera.image_width),
            "world_view_transform": camera.world_view_transform,
            "full_proj_transform": camera.full_proj_transform,
            "extrinsic_matrix": camera.extrinsic_matrix,
            "intrinsic_matrix": camera.intrinsic_matrix,
            "camera_center": camera.camera_center,
        }



    def compute_all_losses(self, 
                           batch,
                           w_rgb=1,
                           w_depth_range=10,
                           w_distortion=1.,
                           w_scene_flow_smooth=10.,
                           w_canonical_unit_sphere=0.,
                           w_flow_grad=0.01,
                           write_logs=True,
                           return_data=False,
                           log_prefix='loss',
                           ):
        ids1 = batch['ids1'].to(self.device)
        ids2 = batch['ids2'].to(self.device)
        # px1s = batch['pts1'].to(self.device)
        # px2s = batch['pts2'].to(self.device)
        gt_rgb1 = batch['gt_rgb1'].to(self.device)
        # weights = batch['weights'].to(self.device)

        # Step 1: render frames given two time index (for flow and rgb)
        render_dict = self.gs_atlases_model.forward(ids1)
        render_dict2 = self.gs_atlases_model.forward(ids2)

        # ### TODO unify the parameter to merge two cases
        # if self.enable_ortho_projection:
        #     (uv1, depth1) = self.renderer.project_point(
        #         render_dict["detached_position"],
        #         self.batch_dict["extrinsic_matrix"].cuda(),
        #         self.batch_dict["width"], 
        #         self.batch_dict["height"],
        #         nearest=0.01)
            
        #     (uv2, depth2) = self.renderer.project_point(
        #         render_dict2["detached_position"],
        #         self.batch_dict["extrinsic_matrix"].cuda(),
        #         self.batch_dict["width"], 
        #         self.batch_dict["height"],
        #         nearest=0.01)
        # else:
        #     raise NotImplementedError("Not implemented for perspective projection")
        track_gs = render_dict2["position"]

        render_dict.update({"track_gs": track_gs})  # TODO add flow
        ##### use copy of batch_dict to avoid changing the original batch_dict of "pixel flow"
        batch_dict_copy = self.batch_dict.copy()
        batch_dict_copy.update({"render_attributes_list" : ["track_gs"]+list(self.cfg.render_attributes.keys())})
        batch_dict_copy.update({"num_idx": 20})

        ### fix attributes
        if False:
            optim = getattr(self, self.gs_atlas_cfg_list[0].name+"_optimizer")
            if hasattr(optim.optimizer_dict['optimizer_1'], 'last_reset_opacity_step') and abs(optim.optimizer_dict['optimizer_1'].last_reset_opacity_step - self.step) < 200:
                render_dict['position'] = render_dict['position'].detach()            
                render_dict['scaling'] = render_dict['scaling'].detach()
                render_dict['rotation'] = render_dict['rotation'].detach()
                render_dict['shs'] = render_dict['shs'].detach()
                render_dict['pixel_flow'] = render_dict['pixel_flow'].detach()

        render_results = self.renderer.render_batch(render_dict, [batch_dict_copy])
        # pred_flow = render_results['pixel_flow'][0].permute(1,2,0).reshape(1,-1,2)
        # optical_flow_loss = masked_l1_loss(pred_flow, px2s - px1s, weights, normalize=False)

        ##### a new optical flow loss version
        # frame intervals for weigths
        frame_intervals = torch.abs(ids2 - ids1).float()
        w_interval = torch.exp(-2 * frame_intervals / self.num_imgs)
        # calculate predict track
        predicted_track_gs = render_results['track_gs'].permute(0,2,3,1)  # [1, h, w, 3]
        predicted_track_2d = util.denormalize_coords(predicted_track_gs[...,:2], self.h, self.w)  # [1, h, w, 2]
        # resize_h, resize_w = math.ceil(self.h / 4), math.ceil(self.w / 4)
        # load gt track_2d
        query_tracks_2d = self.video_3d_flow.load_target_tracks(ids1.item(), [ids1.item()])[:, 0, :2]  # [NumPoints, 2]
        target_tracks = self.video_3d_flow.load_target_tracks(ids1.item(), [ids2.item()], dim=0)  # [NumT, NumPoints, 4]
        query_tracks_2d = query_tracks_2d.to(self.device)
        target_tracks = target_tracks.to(self.device)

        gt_tracks_2d = target_tracks[...,:2].reshape(-1,2)  # (P_all, 2)
        target_visibles, target_invisibles, target_confidences = \
            parse_tapir_track_info(target_tracks[..., 2], target_tracks[..., 3])
        gt_visibles = target_visibles.reshape(-1)
        gt_confidences = target_confidences.reshape(-1)

        # optical_flow_loss
        track_weights = gt_confidences[..., None] * w_interval  # [NumPoints, ]
        # (B, H, W).
        masks_flatten = torch.zeros_like(predicted_track_2d[...,0])
            # This takes advantage of the fact that the query 2D tracks are
            # always on the grid.
        query_pixels = query_tracks_2d.to(torch.int64)
        masks_flatten[0, query_pixels[:, 1], query_pixels[:, 0]] = 1.0
        # (B * N, H * W).
        masks_flatten = (
            masks_flatten.reshape(-1, self.h * self.w) > 0.5
        )
        predicted_track_2d = predicted_track_2d.reshape(-1, self.h * self.w, 2)
        if len(predicted_track_2d[masks_flatten][gt_visibles]) > 0:
            optical_flow_loss = masked_l1_loss(
                predicted_track_2d[masks_flatten][gt_visibles],
                gt_tracks_2d[gt_visibles],
                mask=track_weights[gt_visibles],
                quantile=0.98,
            ) / max(self.h, self.w)
        else:
            optical_flow_loss = torch.tensor(0.0).to(self.device)


        pred_rgb1 = render_results['rgb'][0].permute(1,2,0).reshape(1,-1,3)
        # imageio.imwrite("./debug.png", (pred_rgb1.reshape(self.h,self.w,3).detach().cpu().numpy()*255).astype(np.uint8))
        # loss_rgb = F.mse_loss(pred_rgb1, gt_rgb1)
        L1_loss_rgb = l1_loss(pred_rgb1.reshape(-1,self.h,self.w,3), gt_rgb1.reshape(-1,self.h,self.w,3))
        ssim_loss = 1 - ssim(pred_rgb1.reshape(-1,self.h,self.w,3), gt_rgb1.reshape(-1,self.h,self.w,3))
        lambda_dssim = 0.2
        loss_rgb = (1.0 - lambda_dssim) * L1_loss_rgb + lambda_dssim * ssim_loss

        # Step 2: prepare for GS optimization
        # loss = w_rgb * loss_rgb + optical_flow_loss * 0.1
        # loss = 50 * loss_rgb + optical_flow_loss * 0.1
        loss = self.args.loss_rgb_weight * loss_rgb + self.args.loss_flow_weight * optical_flow_loss
        # loss = loss_rgb + optical_flow_loss
        # loss = loss_rgb
        # loss = w_rgb * loss_rgb

        # Step 4: add depth loss
        depth = render_results['depth'][0].permute(1,2,0)  # [h,w,1]
        gt_depth = self.gt_depths[ids1][0,...,None]
        ### input should be [bs,h,w,c]
        # loss += self.depth_loss(depth[None], gt_depth[None], torch.ones_like(gt_depth)[None])
        # loss += torch.nn.functional.l1_loss(depth, gt_depth) * 0.01

        from loss import depth_correlation_loss
        # loss += depth_correlation_loss(gt_depth, depth, patch_size=32, num_patches=64)

        if True:
            from loss import depth_loss_dpt
            loss_depth = depth_loss_dpt(depth, gt_depth)
            loss += loss_depth

        # ### add metric depth loss
        # metric_depth = self.video_3d_flow.depths[ids1][...,None].to(self.device)
        # loss_metric_depth = F.mse_loss(depth, metric_depth)
        # loss += loss_metric_depth

        # Step 5: constrain opacity of each gaussian
        # loss += torch.nn.functional.l1_loss(render_dict['opacity'], torch.zeros_like(render_dict['opacity'])) * 0.001
        ### adopt entropy loss
        # loss_entropy = entropy_loss(render_dict['opacity'], render_results['gs_idx'])
        # loss_entropy = torch.nn.functional.binary_cross_entropy(render_dict['opacity'], render_dict['opacity'])
        # loss += loss_entropy * 0.1

        # assume a bell function for the opacity

        ############# This is used to test topK rendering
        if False:
            from pointrix.utils.sh_utils import eval_sh
            # attribute = uv2-uv1
            pixel_weight, loss_entropy = entropy_loss(render_dict['opacity'], render_results['gs_idx'])
            loss += loss_entropy * 0.01
            attribute = render_dict['mask_attribute']
            test_render_result = alpha_blending_firstK(attribute,  
                                                    render_results['gs_idx'],
                                                    pixel_weight,
                                                    bg=0)

        # Step 6: 
        #### loss for the attributes
        if False:
            rendered_mask = render_results['mask_attribute'][0].permute(1,2,0)  # [h,w,1]
            gt_mask = self.masks[ids1][0,...,None]
            loss_mask_attribute = F.mse_loss(rendered_mask, gt_mask)
            # loss_mask_attribute = F.binary_cross_entropy(rendered_mask, gt_mask)
            loss += loss_mask_attribute * 20

        if False:
            rendered_dino = render_results['dino_attribute'][0].permute(1,2,0)  # [h,w,3]
            gt_dino = self.dinos[ids1][0]
            loss_dino_attribute = F.mse_loss(rendered_dino, gt_dino)
            loss += loss_dino_attribute * 20

        ######## extract GS layer based on attributes
        fg_gs_mask = (render_dict['mask_attribute'].squeeze() > 0.5).detach()
        # render_dict_fg_layer1 = {k: v[fg_gs_mask] for k, v in render_dict.items()}

        # candidate_list = list(np.arange(self.num_imgs))
        # candidate_list.remove(ids1.item())
        # ids2 = torch.tensor([np.random.choice(candidate_list)]).to(self.device)
        # render_dict2 = self.gs_atlases_model.forward(ids2)
        render_dict_fg_layer2 = {k: v[fg_gs_mask] for k, v in render_dict2.items()}

        ### avoid reduandant mask
        if self.step > 100 and False:
            batch_dict_copy = self.batch_dict.copy()
            batch_dict_copy.update({"render_attributes_list" : ['mask_attribute'],
                                    "bg_color" : 0.0})
            render_results_layer1 = self.renderer.render_batch(render_dict_fg_layer1, [batch_dict_copy])
            # imageio.imwrite("./debug.png", (render_results_layer1['mask_attribute'][0,0].clamp(0,1).detach().cpu().numpy()*255).astype(np.uint8))

            pred_rgb1_fg = render_results_layer1['rgb'][0].permute(1,2,0).reshape(1,-1,3)
            # imageio.imwrite("./debug.png", (pred_rgb1.reshape(self.h,self.w,3).detach().cpu().numpy()*255).astype(np.uint8))
            loss_rgb_fg = F.mse_loss(pred_rgb1_fg, gt_rgb1*(gt_mask.reshape(1,-1,1)))
            loss += loss_rgb_fg * 20
            
            loss_mask_attribute_fg = F.mse_loss(render_results_layer1['mask_attribute'][0].permute(1,2,0), gt_mask)
            loss += loss_mask_attribute_fg * 20
        
        # ### add rigid constraint
        from geometry_utils import cal_connectivity_from_points, cal_arap_error, cal_smooth_error
        ii, jj, nn, weight = cal_connectivity_from_points(points=render_dict["position"], K=5)
        pos = torch.stack([render_dict["position"], render_dict2["position"]], dim=0)
        rigid_error = cal_arap_error(pos, ii, jj, nn) / 1000.  # this loss is too large
        loss += rigid_error

        # render_dict_bg_layer1 = {k: v[~fg_gs_mask] for k, v in render_dict.items()}
        # render_dict_bg_layer2 = {k: v[~fg_gs_mask] for k, v in render_dict2.items()}
        # ### estimate a global motion for the background
        # if True:
        #     ### add later
        #     pos1 = render_dict_bg_layer1["detached_position"]  # [N,3]
        #     pos2 = render_dict_bg_layer2["detached_position"]
        #     translation = pos2.mean(dim=0, keepdim=True) - pos1.mean(dim=0, keepdim=True)
        #     rotation = torch.eye(3).to(pos1.device)  # replated with SVD estimation from sampled points
        #     transformed_pos = torch.matmul(pos1, rotation.detach()) + translation.detach()
        #     smooth_error = torch.nn.functional.l1_loss(transformed_pos, pos2)
        # else:
        #     # ii, jj, nn, weight = cal_connectivity_from_points(points=render_dict_bg_layer1["position"], K=10)
        #     dynamic_features = torch.cat([render_dict_bg_layer1["pos_poly_feat"], 
        #                                 render_dict_bg_layer1["pos_fourier_feat"],
        #                                 render_dict_bg_layer1["rot_poly_feat"],
        #                                 render_dict_bg_layer1["rot_fourier_feat"]
        #                                 ], dim=-1)
        #     dynamic_features_mean = dynamic_features.mean(dim=0, keepdim=True)
        #     smooth_error = torch.nn.functional.l1_loss(dynamic_features-dynamic_features_mean, torch.zeros_like(dynamic_features))
        #     # smooth_error = cal_smooth_error(dynamic_features, ii, jj, nn)
        # loss += smooth_error * 50



        if False:
            ### split GS layer and constrain their motion seperately
            num_gs = render_dict['opacity'].shape[0]
            pixel_gs_idx = pixel2gs(render_dict['opacity'], render_results['gs_idx'])
            selected_gs_idx = pixel_gs_idx[self.masks[ids1] > 0]  # gs correspondent to the mask region
            selected_gs_idx = selected_gs_idx[selected_gs_idx != num_gs]  # empty pixel is filled with invalid index
            selected_gs_idx = torch.unique(selected_gs_idx)
            if selected_gs_idx.max() >= render_dict['opacity'].shape[0]:
                import ipdb; ipdb.set_trace()
            render_dict_layer1 = {k: v[selected_gs_idx] for k, v in render_dict.items()}

            complementary_gs_idx = torch.tensor(list(set(range(num_gs)) - set(selected_gs_idx))).to(selected_gs_idx.device)
            render_dict_layer2 = {k: v[complementary_gs_idx] for k, v in render_dict.items()}


        data = {'ids1': ids1,
                'ids2': ids2,
                }
        data.update(render_results)
        if return_data:
            return loss, data
        else:
            return loss


    def weight_scheduler(self, step, start_step, w, min_weight, max_weight):
        if step <= start_step:
            weight = 0.0
        else:
            weight = w * (step - start_step)
        weight = np.clip(weight, a_min=min_weight, a_max=max_weight)
        return weight
    
    
    def train_one_step(self, step, batch):
        self.step = step
        start = time.time()
        self.scalars_to_log = {}

        w_rgb = self.weight_scheduler(step, 0, 1./5000, 0, 10)
        w_flow_grad = self.weight_scheduler(step, 0, 1./500000, 0, 0.1)
        w_distortion = self.weight_scheduler(step, 40000, 1./2000, 0, 10)
        w_scene_flow_smooth = 20.

        # self.renderer.update_sh_degree(iteration)  # TODO add later
        loss, render_results = self.compute_all_losses(batch,
                                                  w_rgb=w_rgb,
                                                  w_scene_flow_smooth=w_scene_flow_smooth,
                                                  w_distortion=w_distortion,
                                                  w_flow_grad=w_flow_grad,
                                                  return_data=True)
        
        if torch.isnan(loss):
            pdb.set_trace()

        loss.backward()

        ### TODO GS optimizer
        ##### TODO parse viewspace_points, visibility, radii to each GS atlas
        self.optimizer_dict = self.gs_atlases_model.prepare_optimizer_dict(loss, render_results)
        for gs_atlas_cfg in self.gs_atlas_cfg_list:
            name = gs_atlas_cfg.name
            getattr(self, name+"_optimizer").update_model(**self.optimizer_dict[name])
            getattr(self, name+"_scheduler").step(self.step, getattr(self, name+"_optimizer"))

            for x in getattr(self, name+"_optimizer").param_groups:
                self.scalars_to_log['lr_'+x['name']] = x['lr']
        self.scalars_to_log['loss'] = loss.item()


        self.scalars_to_log['time'] = time.time() - start
        self.ids1 = render_results['ids1']
        self.ids2 = render_results['ids2']

    
    def get_pred_flows_gs(self, ids1, ids2, return_original=False):
        render_dict = self.gs_atlases_model.forward(ids1)
        render_dict2 = self.gs_atlases_model.forward(ids2)
        if self.enable_ortho_projection:
            (uv1, depth1) = self.renderer.project_point(
                render_dict["position"],
                self.batch_dict["extrinsic_matrix"].cuda(),
                self.batch_dict["width"], 
                self.batch_dict["height"],
                nearest=0.01)
            
            (uv2, depth2) = self.renderer.project_point(
                render_dict2["position"],
                self.batch_dict["extrinsic_matrix"].cuda(),
                self.batch_dict["width"], 
                self.batch_dict["height"],
                nearest=0.01)
        else:
            (uv1, depth1) = gs.project_point(
                render_dict["position"],
                self.batch_dict["intrinsic_matrix"].cuda(),
                self.batch_dict["extrinsic_matrix"].cuda(),
                self.batch_dict["width"], 
                self.batch_dict["height"],
                nearest=0.01
                )

            (uv2, depth2) = gs.project_point(
                render_dict2["position"],
                self.batch_dict["intrinsic_matrix"].cuda(),
                self.batch_dict["extrinsic_matrix"].cuda(),
                self.batch_dict["width"], 
                self.batch_dict["height"],
                nearest=0.01
                )
        render_dict.update({"pixel_flow": uv2-uv1})  # TODO add flow
        batch_dict_copy = self.batch_dict.copy()
        batch_dict_copy.update({"render_attributes_list" : ["pixel_flow"]+list(self.cfg.render_attributes.keys())})
        render_results = self.renderer.render_batch(render_dict, [batch_dict_copy])
        pred_flow = render_results['pixel_flow'][0].permute(1,2,0).cpu().numpy()
        flow_imgs = util.flow_to_image(pred_flow)
        if return_original:
            return flow_imgs, pred_flow
        else:
            return flow_imgs  # [n, h, w, 3], numpy arra


    
    def get_pred_color_and_depth_maps_gs(self, ids):
        pred_rgbs, pred_depths = [], [] 
        for id in ids:
            render_dict = self.gs_atlases_model.forward(id)
            render_results = self.renderer.render_batch(render_dict, [self.batch_dict])

            pred_rgbs.append(render_results['rgb'].permute(0,2,3,1).clamp(0,1))
            pred_depths.append(render_results['depth'].permute(0,2,3,1))
        
        return torch.cat(pred_rgbs, dim=0), torch.cat(pred_depths, dim=0)  # [n, h, w, 3/1]
    
    
    def log(self, writer, step):
        if self.args.local_rank == 0:
            if step % self.args.i_print == 0:
                logstr = '{}_{} | step: {} |'.format(self.args.expname, self.seq_name, step)
                for k in self.scalars_to_log.keys():
                    logstr += ' {}: {:.6f}'.format(k, self.scalars_to_log[k])
                    if k != 'time':
                        writer.add_scalar(k, self.scalars_to_log[k], step)
                print(logstr)

            if step % self.args.i_img == 0 and False:
                ids = torch.cat([self.ids1[0:1], self.ids2[0:1]])
                with torch.no_grad():
                    pred_imgs, pred_depths = self.get_pred_color_and_depth_maps_gs(ids)
                    pred_imgs = pred_imgs.cpu()
                    pred_depths = pred_depths[...,0].cpu()

                # write depth maps
                pred_depths_cat = pred_depths.permute(1, 0, 2).reshape(self.h, -1)
                min_depth = pred_depths_cat.min().item()
                max_depth = pred_depths_cat.max().item()

                pred_depths_vis = util.colorize(pred_depths_cat, range=(min_depth, max_depth), append_cbar=True)
                pred_depths_vis = F.interpolate(pred_depths_vis.permute(2, 0, 1)[None], scale_factor=0.5, mode='area')
                writer.add_image('depth', pred_depths_vis, step, dataformats='NCHW')

                # write gt and predicted rgbs
                gt_imgs = self.images[ids.cpu()]
                imgs_vis = torch.cat([gt_imgs, pred_imgs], dim=1)
                imgs_vis = F.interpolate(imgs_vis.permute(0, 3, 1, 2), scale_factor=0.5, mode='area')
                writer.add_images('images', imgs_vis, step, dataformats='NCHW')

                # write flow
                with torch.no_grad():
                    flows = self.get_pred_flows_gs(self.ids1[0:1], self.ids2[0:1])
                    id1, id2 = self.ids1[0], self.ids2[0]
                    gt_flow = np.load(os.path.join(self.seq_dir, 'raft_exhaustive',
                                                   '{}_{}.npy'.format(os.path.basename(self.img_files[id1]),
                                                                      os.path.basename(self.img_files[id2]))
                                                   ))
                    gt_flow_img = util.flow_to_image(gt_flow)
                    
                writer.add_image('flow', np.concatenate([flows, gt_flow_img], axis=1), step, dataformats='HWC')

            if step % self.args.i_weight == 0 and step > 0:
                vis_dir = os.path.join(self.out_dir, 'vis')
                os.makedirs(vis_dir, exist_ok=True)
                print('saving visualizations to {}...'.format(vis_dir))
                if False:
                    images, depths = [], []
                    for id in range(self.num_imgs):
                        with torch.no_grad():
                            render_dict = self.gs_atlases_model.forward(id)
                            render_results = self.renderer.render_batch(render_dict, [self.batch_dict])
                            pred_rgb = render_results['rgb'][0].permute(1,2,0).clamp(0,1).cpu().numpy()
                            pred_depth = render_results['depth'][0].permute(1,2,0).cpu().numpy()
                        images.append((pred_rgb*255).astype(np.uint8))
                        depths.append(pred_depth)
                        # depths is processed later

                    imageio.mimwrite(os.path.join(vis_dir, '{}_rgb{:06d}.mp4'.format(self.seq_name, step)),
                                        images,
                                        quality=8, fps=4)
                    depths_np = np.stack(depths, axis=0)
                    depths_np = (depths_np - depths_np.min()) / (depths_np.max() - depths_np.min())
                    depths_np = (depths_np * 255).astype(np.uint8)
                    imageio.mimwrite(os.path.join(vis_dir, '{}_depth{:06d}.mp4'.format(self.seq_name, step)),
                                        depths_np,
                                        quality=8, fps=4)
                ### save tracking result
                save_dir = os.path.join(self.out_dir, 'tracking')
                os.makedirs(save_dir, exist_ok=True)
                # for i, img in enumerate(track_imgs):
                #     imageio.imwrite(os.path.join(save_dir, f'{step}_{i}.png'), img)
                track_imgs = self.draw_pixel_trajectory(use_mask=True)
                track_imgs = [x[:,self.w:] for x in track_imgs]
                imageio.mimwrite(os.path.join(save_dir, f'{step}.mp4'), track_imgs, fps=10)
                # track_imgs = self.draw_pixel_trajectory(use_mask=False)
                # track_imgs = [x[:,self.w:] for x in track_imgs]
                # imageio.mimwrite(os.path.join(save_dir, f'{step}_no_mask.mp4'), track_imgs, fps=10)

                ### save video render result
                self.render_video(step, save_frames=True)

                # self.render_part(fg=True, threshold=0.5)
                # self.render_part(fg=False, threshold=0.5)
                
                fpath = os.path.join(self.out_dir, 'model_{:06d}.pth'.format(step))
                self.save_model(fpath)
                

    def save_model(self, path: Path = None) -> None:
        data_list = {
            "gs_atlases_model": self.gs_atlases_model.get_state_dict(),
            "renderer": self.renderer.state_dict()
        }
        data_list.update(
            {k.name+"_optimizer": getattr(self,k.name+"_optimizer").state_dict() for k in self.gs_atlas_cfg_list}
        )
        # data_list.update(
        #     {k.name+"_scheduler": getattr(self,k.name+"_scheduler").state_dict() for k in self.gs_atlas_cfg_list}
        # )
        torch.save(data_list, path)


    def load_model(self, path: Path = None) -> None:
        data_list = torch.load(path)
        for k, v in data_list.items():
            print(f"Loaded {k} from checkpoint")
            # get arrtibute from model
            arrt = getattr(self, k)
            if hasattr(arrt, 'load_state_dict'):
                arrt.load_state_dict(v)
            else:
                setattr(self, k, v)
        ### TODO fix this
        # re-initialize optimizer
        cameras_extent = 5  # the extent of the camera
        for gs_atlas_cfg in self.gs_atlas_cfg_list:
            name = gs_atlas_cfg.name
            setattr(self, name+"_optimizer", 
                    parse_optimizer(self.cfg.optimizer, 
                                    model=self.gs_atlases_model.get_atlas(name),
                                    cameras_extent=cameras_extent)
                    )
            setattr(self, name+"_scheduler",
                    parse_scheduler(self.cfg.scheduler, 
                                    cameras_extent if self.cfg.spatial_lr_scale else 1.)
                    )


    def load_from_ckpt(self, out_folder,
                       load_opt=True,
                       load_scheduler=True,
                       force_latest_ckpt=False):
        '''
        load model from existing checkpoints and return the current step
        :param out_folder: the directory that stores ckpts
        :return: the current starting step
        '''

        # all existing ckpts
        ckpts = []
        if os.path.exists(out_folder):
            ckpts = [os.path.join(out_folder, f)
                     for f in sorted(os.listdir(out_folder)) if f.endswith('.pth')]

        if self.args.ckpt_path is not None and not force_latest_ckpt:
            if os.path.isfile(self.args.ckpt_path):  # load the specified ckpt
                ckpts = [self.args.ckpt_path]

        if len(ckpts) > 0 and not self.args.no_reload:
            fpath = ckpts[-1]
            self.load_model(fpath)
            step = int(fpath[-10:-4])
            print('Reloading from {}, starting at step={}'.format(fpath, step))
        else:
            print('No ckpts found, from scratch...')
            step = 0

        return step
    
    def optimize_appearance_from_mask(self, mask_path, img_path):
        mask = imageio.imread(mask_path)
        if mask.ndim == 3:
            mask = mask[...,0]
        # mask = cv2.dilate(mask, np.ones((5,5), np.uint8), iterations=1)
        mask = mask / 255.
        mask = torch.tensor(mask).float().to(self.device)  # [H,W]

        gt_image = imageio.imread(img_path)[...,:3] / 255.
        gt_image = torch.tensor(gt_image).float().to(self.device)

        render_dict = self.gs_atlases_model.forward(0)
        batch_dict_copy = self.batch_dict.copy()
        batch_dict_copy.update({"num_idx" : 10})
        render_results = self.renderer.render_batch(render_dict, [batch_dict_copy])
        selected_gs_idx = torch.unique(render_results['gs_idx'][0][mask > 0])
        selected_gs_idx = selected_gs_idx[selected_gs_idx != -1]

        ### construct an optimizer
        from pointrix.utils.gaussian_points.gaussian_utils import inverse_sigmoid
        optimized_shs = nn.Parameter(render_dict['shs'][selected_gs_idx].detach(), requires_grad=True)
        optimized_opacity = nn.Parameter(inverse_sigmoid(render_dict['opacity'][selected_gs_idx]).detach(), requires_grad=True)
        parameters = [{'params': optimized_shs, 'lr': 0.0025}, 
                    #   {'params': optimized_opacity, 'lr': 0.05}
                      ]
        optim = torch.optim.Adam(parameters)
        # optim = torch.optim.SGD(
        #     [optimized_shs, optimized_opacity], 
        #     lr=0.05)
        # scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=500, gamma=0.99999)
        
        for idx in tqdm(range(1000), mininterval=100):
            with torch.no_grad():
                render_dict = self.gs_atlases_model.forward(0)
            render_dict['shs'][selected_gs_idx] = optimized_shs
            render_dict['opacity'][selected_gs_idx] = torch.sigmoid(optimized_opacity)
            render_results = self.renderer.render_batch(render_dict, [batch_dict_copy])
            loss = F.mse_loss(render_results['rgb'][0].permute(1,2,0), gt_image)
            optim.zero_grad()
            loss.backward()
            optim.step()
            # scheduler.step()
            if idx % 10 == 0:
                print(f'loss: {loss.item()}')
            if loss.item() < 0.0001:
                break

        images, depths = [], []
        for id in range(self.num_imgs):
            with torch.no_grad():
                render_dict = self.gs_atlases_model.forward(id)
                render_dict['shs'][selected_gs_idx] = optimized_shs
                render_dict['opacity'][selected_gs_idx] = torch.sigmoid(optimized_opacity)
                render_results = self.renderer.render_batch(render_dict, [batch_dict_copy])
                pred_rgb = render_results['rgb'][0].permute(1,2,0).clamp(0,1).cpu().numpy()
                pred_depth = render_results['depth'][0].permute(1,2,0).clamp(0,1).cpu().numpy()
            images.append((pred_rgb*255).astype(np.uint8))
            depths.append((pred_depth*255).astype(np.uint8))

        imageio.mimwrite(os.path.join('{}_editing.mp4'.format(self.seq_name)),
                            images,
                            quality=8, fps=4)
        print()


    

    def optimize_appearance_from_img(self, img_path):
        # The optimization process should be constrained in a mask region.
        gt_image = imageio.imread(img_path)[...,:3] / 255.
        gt_image = torch.tensor(gt_image).float().to(self.device)

        cameras_extent = 5  # the extent of the camera
        for gs_atlas_cfg in self.gs_atlas_cfg_list:
            name = gs_atlas_cfg.name
            setattr(self, name+"_optimizer", 
                    parse_optimizer(self.cfg.optimizer, 
                                    model=self.gs_atlases_model.get_atlas(name),
                                    cameras_extent=cameras_extent)
                    )

        ### only optimize appearance
        for gs_atlas_cfg in self.gs_atlas_cfg_list:
            name = gs_atlas_cfg.name
            self.gs_atlases_model.get_atlas(name).point_cloud.position.requires_grad = False
            self.gs_atlases_model.get_atlas(name).point_cloud.scaling.requires_grad = False
            self.gs_atlases_model.get_atlas(name).point_cloud.rotation.requires_grad = False
            self.gs_atlases_model.get_atlas(name).point_cloud.opacity.requires_grad = False
            self.gs_atlases_model.get_atlas(name).point_cloud.pos_del_feat.requires_grad = False
            # self.gs_atlases_model.get_atlas(name).point_cloud.features.requires_grad = False
            self.gs_atlases_model.get_atlas(name).point_cloud.features_rest.requires_grad = False
            self.gs_fg_optimizer.optimizer_dict['optimizer_1'].cfg.densify_start_iter = 2000

        for idx in tqdm(range(1000)):
            render_dict = self.gs_atlases_model.forward(0)
            render_results = self.renderer.render_batch(render_dict, [self.batch_dict])
            pred_rgb = render_results['rgb'][0].permute(1,2,0)
            loss = F.mse_loss(pred_rgb, gt_image)
            loss.backward()
            self.optimizer_dict = self.gs_atlases_model.prepare_optimizer_dict(loss, render_results)
            for gs_atlas_cfg in self.gs_atlas_cfg_list:
                name = gs_atlas_cfg.name
                getattr(self, name+"_optimizer").update_model(**self.optimizer_dict[name])
            if idx % 10 == 0:
                print(f'loss: {loss.item()}')
            if loss.item() < 0.0001:
                break

        images, depths = [], []
        for id in range(self.num_imgs):
            with torch.no_grad():
                render_dict = self.gs_atlases_model.forward(id)
                render_results = self.renderer.render_batch(render_dict, [self.batch_dict])
                pred_rgb = render_results['rgb'][0].permute(1,2,0).clamp(0,1).cpu().numpy()
                pred_depth = render_results['depth'][0].permute(1,2,0).clamp(0,1).cpu().numpy()
            images.append((pred_rgb*255).astype(np.uint8))
            depths.append((pred_depth*255).astype(np.uint8))

        imageio.mimwrite(os.path.join('{}_editing.mp4'.format(self.seq_name)),
                            images,
                            quality=8, fps=4)
        print()

        
    def get_nvs_rendered_imgs(self,):            
        ##### project to frame space
        radius = 0.05
        z_center = 1.
        color_frame_list = []
        for idx, phi in enumerate(torch.linspace(0, 4 * np.pi, self.num_imgs)):
            render_dict = self.gs_atlases_model.forward(idx)
            camera_position = torch.tensor([[radius*torch.cos(phi), radius*torch.sin(phi), 0.0]], device=self.device)
            camra_rotation = look_at_rotation(camera_position, at=((0,0,z_center),), device=self.device)
            c2w = torch.cat([camra_rotation[0], camera_position.T], dim=1).cpu().numpy()
            c2w = np.concatenate([c2w, np.array([[0,0,0,1]])], axis=0)
            camera = construct_canonical_camera(width=self.w, height=self.h, c2w=c2w)

            batch_dict = {
                "camera": camera,
                "FovX": camera.fovX,
                "FovY": camera.fovY,
                "height": int(camera.image_height),
                "width": int(camera.image_width),
                "world_view_transform": camera.world_view_transform,
                "full_proj_transform": camera.full_proj_transform,
                "extrinsic_matrix": camera.extrinsic_matrix,
                "intrinsic_matrix": camera.intrinsic_matrix,
                "camera_center": camera.camera_center,
            }

            with torch.no_grad():
                render_results = self.renderer.render_batch(render_dict, [batch_dict])
            color_frame_list.append(render_results['rgb'][0].clamp(0,1).permute(1,2,0).cpu().numpy())
        color_frame_list = [(x*255).astype(np.uint8) for x in color_frame_list]
        save_path = os.path.join(self.out_dir, 'nvs.mp4')
        imageio.mimwrite(save_path, color_frame_list, fps=4)
        print()


    def get_stereo_rendered_imgs(self,):
        color_frame_list = []
        radius = 0.05
        phi = 0
        import math
        camera_position = torch.tensor([[radius*math.cos(phi), radius*math.sin(phi), 0.0]], device=self.device)
        camra_rotation = look_at_rotation(camera_position, at=((0,0,2.5),), device=self.device)
        c2w = torch.cat([camra_rotation[0], camera_position.T], dim=1).cpu().numpy()
        c2w = np.concatenate([c2w, np.array([[0,0,0,1]])], axis=0)
        camera1 = construct_canonical_camera(width=self.w, height=self.h, c2w=c2w)

        phi = math.pi
        camera_position = torch.tensor([[radius*math.cos(phi), radius*math.sin(phi), 0.0]], device=self.device)
        camra_rotation = look_at_rotation(camera_position, at=((0,0,2.5),), device=self.device)
        c2w = torch.cat([camra_rotation[0], camera_position.T], dim=1).cpu().numpy()
        c2w = np.concatenate([c2w, np.array([[0,0,0,1]])], axis=0)
        camera2 = construct_canonical_camera(width=self.w, height=self.h, c2w=c2w)

        batch_dict1 = {
            "camera": camera1,
            "FovX": camera1.fovX,
            "FovY": camera1.fovY,
            "height": int(camera1.image_height),
            "width": int(camera1.image_width),
            "world_view_transform": camera1.world_view_transform,
            "full_proj_transform": camera1.full_proj_transform,
            "extrinsic_matrix": camera1.extrinsic_matrix,
            "intrinsic_matrix": camera1.intrinsic_matrix,
            "camera_center": camera1.camera_center,
        }

        batch_dict2 = {
            "camera": camera2,
            "FovX": camera2.fovX,
            "FovY": camera2.fovY,
            "height": int(camera2.image_height),
            "width": int(camera2.image_width),
            "world_view_transform": camera2.world_view_transform,
            "full_proj_transform": camera2.full_proj_transform,
            "extrinsic_matrix": camera2.extrinsic_matrix,
            "intrinsic_matrix": camera2.intrinsic_matrix,
            "camera_center": camera2.camera_center,
        }

        matrices = {
            'true': [ [ 0.299, 0.587, 0.114, 0, 0, 0, 0, 0, 0 ], [ 0, 0, 0, 0, 0, 0, 0.299, 0.587, 0.114 ] ],
            'mono': [ [ 0.299, 0.587, 0.114, 0, 0, 0, 0, 0, 0 ], [ 0, 0, 0, 0.299, 0.587, 0.114, 0.299, 0.587, 0.114 ] ],
            'color': [ [ 1, 0, 0, 0, 0, 0, 0, 0, 0 ], [ 0, 0, 0, 0, 1, 0, 0, 0, 1 ] ],
            'halfcolor': [ [ 0.299, 0.587, 0.114, 0, 0, 0, 0, 0, 0 ], [ 0, 0, 0, 0, 1, 0, 0, 0, 1 ] ],
            'optimized': [ [ 0, 0.7, 0.3, 0, 0, 0, 0, 0, 0 ], [ 0, 0, 0, 0, 1, 0, 0, 0, 1 ] ],
        }

        for idx in range(self.num_imgs):
            render_dict = self.gs_atlases_model.forward(idx)
            batch_dict1_copy = batch_dict1.copy()
            batch_dict2_copy = batch_dict2.copy()
            batch_dict1_copy.update({"render_attributes_list" : ['dino_attribute']})
            batch_dict2_copy.update({"render_attributes_list" : ['dino_attribute']})

            with torch.no_grad():
                render_results1 = self.renderer.render_batch(render_dict, [batch_dict1_copy])
                render_results2 = self.renderer.render_batch(render_dict, [batch_dict2_copy])
            img1 = render_results1['rgb'][0].clamp(0,1).permute(1,2,0).cpu().numpy()
            img2 =  render_results2['rgb'][0].clamp(0,1).permute(1,2,0).cpu().numpy()

            if False:
                # color_frame_list.append(img1*0.5 + img2*0.5)
                hsv = cv2.cvtColor((img2*255).astype(np.uint8), cv2.COLOR_RGB2HSV)
                hsv[..., 0] = (hsv[..., 0] + 30) % 180
                img2 = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB) / 255.
                img = img1 * 0.5 + img2 * 0.5
                img = (img*255).astype(np.uint8)
            else:
                color = 'optimized'
                # from PIL import Image
                # left = Image.fromarray((img1*255).astype(np.uint8))
                # right = Image.fromarray((img2*255).astype(np.uint8))
                # width, height = left.size
                # leftMap = left.load()
                # rightMap = right.load()
                # m = matrices[color]

                # for y in range(0, height):
                #     for x in range(0, width):
                #         r1, g1, b1 = leftMap[x, y]
                #         r2, g2, b2 = rightMap[x, y]
                #         leftMap[x, y] = (
                #             int(r1*m[0][0] + g1*m[0][1] + b1*m[0][2] + r2*m[1][0] + g2*m[1][1] + b2*m[1][2]),
                #             int(r1*m[0][3] + g1*m[0][4] + b1*m[0][5] + r2*m[1][3] + g2*m[1][4] + b2*m[1][5]),
                #             int(r1*m[0][6] + g1*m[0][7] + b1*m[0][8] + r2*m[1][6] + g2*m[1][7] + b2*m[1][8])
                #         )
                # img = np.array(left)
                m = np.array(matrices[color]).reshape(2,3,3).transpose(1,0,2).reshape(3,6)
                img_cat = np.concatenate([img1, img2], axis=2)  # [H, W, 6]
                img = np.einsum('ijk,lk->ijl', img_cat, m)
                img = (img*255).astype(np.uint8)

            color_frame_list.append(img)

        # color_frame_list = [(x*255).astype(np.uint8) for x in color_frame_list]
        color_frame_list = [x for x in color_frame_list]
        save_path = os.path.join(self.out_dir, 'stereo.mp4')
        imageio.mimwrite(save_path, color_frame_list, fps=4)
        print()


    def render_video(self, step=0, save_frames=False):
        ### render image / depth / dinov2
        images, depths = [], []
        dinos = []
        for id in range(self.num_imgs):
            with torch.no_grad():
                render_dict = self.gs_atlases_model.forward(id)
                batch_dict_copy = self.batch_dict.copy()
                batch_dict_copy.update({"render_attributes_list" : ['dino_attribute', 'mask_attribute']})
                render_results = self.renderer.render_batch(render_dict, [batch_dict_copy])
                pred_rgb = render_results['rgb'][0].clamp(0,1).permute(1,2,0).cpu().numpy()
                pred_depth = render_results['depth'][0].permute(1,2,0).cpu().numpy()
                pred_dino = render_results['dino_attribute'][0].permute(1,2,0).cpu().numpy()
            images.append((pred_rgb*255).astype(np.uint8))
            depths.append(pred_depth)
            dinos.append((pred_dino*255).astype(np.uint8))
        
        from util import colorize_np
        depths = np.stack(depths, axis=0).squeeze()
        depth_max, depth_min = depths.max(), depths.min()
        normed_depth = [colorize_np(x, cmap_name="jet", 
                                    mask=None, range=(depth_min,depth_max), 
                                    append_cbar=False, 
                                    cbar_in_image=False) for x in depths]
        normed_depth = [(x*255).astype(np.uint8) for x in normed_depth]

        if save_frames:
            save_dir = os.path.join(self.out_dir, 'vis', 'frames_{:06d}'.format(step))
            os.makedirs(save_dir, exist_ok=True)
            for i, img in enumerate(images):
                imageio.imwrite(os.path.join(save_dir, f'{i:05d}.png'), img)
        
        imageio.mimwrite(os.path.join(self.out_dir, 'vis', 'render_{:06d}.mp4'.format(step)),
                            images,
                            quality=8, fps=10)
        
        # imageio.mimwrite(os.path.join(self.out_dir, 'vis', 'depth_{:06d}.mp4'.format(step)),
        #                     normed_depth,
        #                     quality=8, fps=10)
        # imageio.mimwrite(os.path.join(self.out_dir, 'vis', 'dino_{:06d}.mp4'.format(step)),
        #                     dinos,
        #                     quality=8, fps=10)
        print()


    
    def render_part(self, fg=True, threshold=0.5):
        """
        Remove the foreground object from the scene
        """
        images, depths = [], []
        for id in range(self.num_imgs):
            with torch.no_grad():
                render_dict = self.gs_atlases_model.forward(id)

                if fg:
                    mask = (render_dict['mask_attribute'].squeeze() > threshold).detach()
                else:
                    mask = (render_dict['mask_attribute'].squeeze() <= threshold).detach()
                render_dict = {k: v[mask] for k, v in render_dict.items()}

                batch_dict_copy = self.batch_dict.copy()
                batch_dict_copy.update({"render_attributes_list" : ['mask_attribute'], 
                                        "bg_color" : 1.0})
                render_results = self.renderer.render_batch(render_dict, [batch_dict_copy])
                pred_rgb = render_results['rgb'][0].clamp(0,1).permute(1,2,0).cpu().numpy()
                pred_depth = render_results['depth'][0].clamp(0,1).permute(1,2,0).cpu().numpy()
                pred_attribute = render_results['mask_attribute'][0].clamp(0,1).permute(1,2,0).cpu().numpy()

                # pred_rgb = (1 - self.masks[id][...,None]).cpu().numpy() * pred_rgb

            images.append((pred_rgb*255).astype(np.uint8))

        save_path = os.path.join(self.out_dir, '%s_part2.mp4' % ('fg' if fg else 'bg'))
        imageio.mimwrite(save_path,
                            images,
                            quality=8, fps=4)
        print()


    def add_fg(self, delta_pos, scale, threshold=0.5):
        render_dict = self.gs_atlases_model.forward(0)
        fg_mask = (render_dict['mask_attribute'].squeeze() > 0.5).detach()

        images, depths = [], []
        for id in range(self.num_imgs):
            render_dict = self.gs_atlases_model.forward(id)
            ### for cow
            # new_idx = max(0, id-2)
            new_idx = int(id / 1.)
            new_delta_pos = delta_pos.clone() + \
                torch.tensor([[-0.0, 0, 0]], device='cuda') * new_idx
            render_dict_tmp = self.gs_atlases_model.forward(new_idx)
            for k, v in render_dict.items():
                if k == 'position':
                    fg_pos = render_dict_tmp['position'][fg_mask]
                    fg_pos_mean = fg_pos.mean(dim=0, keepdim=True)
                    fg_pos = (fg_pos - fg_pos_mean) * scale + fg_pos_mean
                    fg_pos = fg_pos + new_delta_pos

                    # fg_pos[...,0] *= -1
                    fg_rot = render_dict_tmp['rotation'][fg_mask]
                    import pytorch3d
                    from pytorch3d import transforms
                    fg_rot_mat = pytorch3d.transforms.quaternion_to_matrix(fg_rot)
                    # fg_rot_mat[:,0,:3] *= -1
                    fg_rot = pytorch3d.transforms.matrix_to_quaternion(fg_rot_mat)
                    fg_rot = fg_rot / fg_rot.norm(dim=-1, keepdim=True)

                    render_dict['position'] = torch.cat([
                                        render_dict['position'], 
                                        fg_pos
                                    ], dim=0)
                    render_dict['rotation'] = torch.cat([
                                        render_dict['rotation'], 
                                        fg_rot
                                    ], dim=0)
                elif k == "rotation":
                    pass
                else:
                    render_dict[k] = torch.cat([
                                        render_dict[k], 
                                        render_dict_tmp[k][fg_mask]
                                    ], dim=0)

            batch_dict_copy = self.batch_dict.copy()
            batch_dict_copy.update({"render_attributes_list" : ['mask_attribute']})
            with torch.no_grad():
                render_results = self.renderer.render_batch(render_dict, [batch_dict_copy])
            pred_rgb = render_results['rgb'][0].clamp(0,1).permute(1,2,0).cpu().numpy()
            pred_depth = render_results['depth'][0].clamp(0,1).permute(1,2,0).cpu().numpy()
            pred_attribute = render_results['mask_attribute'][0].clamp(0,1).permute(1,2,0).cpu().numpy()
            # imageio.imwrite("./debug.png", (pred_rgb*255).astype(np.uint8))

            images.append((pred_rgb*255).astype(np.uint8))
            depths.append((pred_depth*255).astype(np.uint8))
        imageio.imwrite(os.path.join(self.out_dir, 'added_fg.png'), images[-1])

        imageio.mimwrite(os.path.join(self.out_dir, 'added_fg.mp4'),
                            images,
                            quality=8, fps=4)
        print()


    def draw_gs_trajectory(self, samp_num=10, gs_num=512):
        """
        Draw gs trajectory from time 0 to 1
        """
        cur_pts = self.gs_atlases_model.get_atlas('gs_fg').point_cloud.get_position(0)
        from vis_utils import farthest_point_sample
        pts_idx = farthest_point_sample(cur_pts[None], gs_num)[0]

        spatial_idx = torch.argsort(cur_pts[pts_idx][:,0])
        pts_idx = pts_idx[spatial_idx]

        import cv2
        from matplotlib import cm
        color_map = cm.get_cmap("jet")
        colors = np.array([np.array(color_map(i/max(1, float(gs_num - 1)))[:3]) * 255 for i in range(gs_num)], dtype=np.int32)

        imgs = []
        delta_ts = torch.linspace(0, self.num_imgs-1, samp_num).to(self.device) # right side is included
        for i in range(samp_num):
            cur_time = delta_ts[i]
            cur_pts = self.gs_atlases_model.get_atlas('gs_fg').point_cloud.get_position(cur_time)
            cur_pts = cur_pts[pts_idx]

            (uv, depth) = gs.project_point(
                cur_pts,
                self.batch_dict["intrinsic_matrix"].cuda(),
                self.batch_dict["extrinsic_matrix"].cuda(),
                self.batch_dict["width"], 
                self.batch_dict["height"],
                nearest=0.01
                )
            
            alpha_img = np.zeros([self.h, self.w, 3])
            traj_img = np.zeros([self.h, self.w, 3])

            for i in range(gs_num):
                color = colors[i] / 255
                alpha_img = cv2.circle(img=alpha_img, center=(int(uv[i][0]), int(uv[i][1])), color=[1,1,1], radius=5, thickness=-1)
                traj_img = cv2.circle(img=traj_img, center=(int(uv[i][0]), int(uv[i][1])), color=[float(color[0]), float(color[1]), float(color[2])], radius=5, thickness=-1)

            with torch.no_grad():
                render_dict = self.gs_atlases_model.forward(cur_time)
                render_results = self.renderer.render_batch(render_dict, [self.batch_dict])
                img = render_results['rgb'][0].permute(1,2,0).cpu().numpy()
            img = traj_img * alpha_img[...,:1] + img * (1-alpha_img[...,:1])
            imgs.append((img*255).astype(np.uint8))
        imageio.mimwrite('gs_trajectory.mp4', imgs, fps=4)
        print()


    def sample_pts_within_mask(self, mask, num_pts, return_normed=False, seed=None,
                               use_mask=False, reverse_mask=False, regular=False, interval=10):
        rng = np.random.RandomState(seed) if seed is not None else np.random
        if use_mask:
            if reverse_mask:
                mask = ~mask
            kernel = torch.ones(7, 7, device=self.device)
            mask = morph.erosion(mask.float()[None, None], kernel).bool().squeeze()  # Erosion
        else:
            mask = torch.ones_like(self.grid[..., 0], dtype=torch.bool)

        if regular:
            coords = self.grid[::interval, ::interval, :2][mask[::interval, ::interval]]
        else:
            coords_valid = self.grid[mask][..., :2]
            rand_inds = rng.choice(len(coords_valid), num_pts, replace=(num_pts > len(coords_valid)))
            coords = coords_valid[rand_inds]

        coords_normed = util.normalize_coords(coords, self.h, self.w)
        if return_normed:
            return coords, coords_normed
        else:
            return coords  # [num_pts, 2]


    def draw_pixel_trajectory(self, idx=0, use_mask=False, radius=5):
        mask = self.masks[idx]
        px1s = self.sample_pts_within_mask(mask, num_pts=100, seed=1234,
                                           use_mask=use_mask, reverse_mask=False,
                                           regular=True, interval=20)
        num_pts = len(px1s)
        
        ### render frame to get the gs index
        render_dict = self.gs_atlases_model.forward(idx)
        # selected_gs_idx = render_results['gs_idx'][0][mask > 0]
        # selected_gs_idx = selected_gs_idx[selected_gs_idx != -1]
        # selected_gs_idx = torch.unique(selected_gs_idx)

        (uv1, depth1) = self.renderer.project_point(
            render_dict["detached_position"],
            self.batch_dict["extrinsic_matrix"].cuda(),
            self.batch_dict["width"], 
            self.batch_dict["height"],
            nearest=0.01)

        ### get the motion of the selected gs
        # pos_0 = render_dict['position'][selected_gs_idx]
        
        kpts = [px1s.detach().cpu().numpy()]
        px_masks = [np.ones_like(px1s.detach().cpu().numpy()[:,0], dtype=bool)]
        img_query = (self.images[0].cpu().numpy() * 255).astype(np.uint8)
        ##### set kpt color list here
        set_max = range(128)
        colors = {m: i for i, m in enumerate(set_max)}
        colors = {m: (255 * np.array(plt.cm.hsv(i/float(len(colors))))[:3][::-1]).astype(np.int32)
                for m, i in colors.items()}
        center = np.median(kpts[0], axis=0, keepdims=True)
        coord_angle = np.arctan2(kpts[0][:, 1] - center[:, 1], kpts[0][:, 0] - center[:, 0])
        corr_color = np.int32(64 * coord_angle / np.pi) % 128  # [N]
        # color_list = tuple(colors[corr_color].tolist())  # [N,3]
        color_list = [colors[corr_color[i]].tolist() for i in range(num_pts)]

        imgs_list = [self.images[0].cpu().numpy()]
        out_imgs = []
        for fid in range(1, self.num_imgs):
            render_dict2 = self.gs_atlases_model.forward(fid)

            (uv2, depth2) = self.renderer.project_point(
                render_dict2["detached_position"],
                self.batch_dict["extrinsic_matrix"].cuda(),
                self.batch_dict["width"], 
                self.batch_dict["height"],
                nearest=0.01)
            
            render_dict.update({"pixel_flow": uv2-uv1})  # TODO add flow
            batch_dict_copy = self.batch_dict.copy()
            batch_dict_copy.update({"render_attributes_list" : ["pixel_flow"]+list(self.cfg.render_attributes.keys())})
            batch_dict_copy.update({"num_idx": 20})
            render_results = self.renderer.render_batch(render_dict, [batch_dict_copy])

            px1s_normed = util.normalize_coords(px1s, self.h, self.w)
            px1s_flow = F.grid_sample(render_results['pixel_flow'], px1s_normed[None,:,None,:], align_corners=True)
            ### TODO add occulusion by rendering weights
            px2s = px1s + px1s_flow[0,...,0].permute(1,0)
            px2s_mask = torch.ones_like(px2s[:,0], dtype=torch.bool)

            img_i = self.images[fid].cpu().numpy()
            kpts.append(px2s.detach().cpu().numpy())
            px_masks.append(px2s_mask.detach().cpu().numpy())
            imgs_list.append(img_i)
        
        from functools import reduce
        unioned_px_mask = reduce(np.logical_and, px_masks)
        kpts = [k[unioned_px_mask] for k in kpts]
        for i in range(1, self.num_imgs+1):
            img_query = self.images[0].cpu().numpy()
            # out = util.drawMatches(img_query, img_i, px1s.detach().cpu().numpy(), px2s.detach().cpu().numpy(),
            #                         num_vis=num_pts, mask=None, radius=radius)
            # img = util.drawTrajectory(img_i, kpts[:-10:-1], num_vis=num_pts, idx_vis=np.arange(10))
            # img = util.drawTrajectoryWithColor(img_i, kpts[:-10:-1], color_list=color_list, num_vis=num_pts, idx_vis=np.arange(10))
            start_id = max(0, i-30)
            tracks_2d = np.stack(kpts[start_id:i], axis=0)
        
            # img = util.draw_tracks_2d(imgs_list[i-1], tracks_2d, track_point_size=8, track_line_width=3)  # for rebuttal image vis
            img = util.draw_tracks_2d(imgs_list[i-1], tracks_2d, track_point_size=2, track_line_width=1)  # for rebuttal image vis
            out = np.concatenate([(img_query * 255).astype(np.uint8), img], axis=1)
            out_imgs.append(out)

        return out_imgs
            

        ### draw trajectory. This is used for video visualization
        if False:
            kpts = np.stack(kpts, axis=0)
            img = np.ones((self.h, self.w, 3), dtype=np.uint8)
            img = util.drawTrajectory(img, kpts[::-1], num_vis=num_pts)

        

    def get_attributes_dict(self, frame_idx):
        """
        Get the attributes of the atlas at frame_idx
        """
        return self.gs_atlases_model.forward(frame_idx)
    

    def get_interpolation_result(self, scaling=2, save_frames=True):
        images, depths = [], []
        dinos = []
        # for id in range(self.num_imgs):
        num_imgs = 15
        for id in torch.linspace(0, num_imgs-1, int(num_imgs*scaling-1)):
            with torch.no_grad():
                render_dict = self.gs_atlases_model.forward(id)
                batch_dict_copy = self.batch_dict.copy()
                batch_dict_copy.update({"render_attributes_list" : ['dino_attribute', 'mask_attribute']})
                render_results = self.renderer.render_batch(render_dict, [batch_dict_copy])
                pred_rgb = render_results['rgb'][0].permute(1,2,0).cpu().numpy()
                pred_depth = render_results['depth'][0].permute(1,2,0).cpu().numpy()
                pred_dino = render_results['dino_attribute'][0].permute(1,2,0).cpu().numpy()
            images.append((pred_rgb*255).astype(np.uint8))
            depths.append(pred_depth)
            dinos.append((pred_dino*255).astype(np.uint8))
        

        if save_frames:
            save_dir = os.path.join(self.out_dir, 'vis', 'inter_frames')
            os.makedirs(save_dir, exist_ok=True)
            for i, img in enumerate(images):
                imageio.imwrite(os.path.join(save_dir, f'{i:05d}.png'), img)
        
        imageio.mimwrite(os.path.join(self.out_dir, 'vis', f'interp_{scaling:02d}_render.mp4'),
                            images,
                            quality=8, fps=int(2*scaling))
        print()

    def get_correspondences_and_occlusion_masks_for_pixels(self, ids1, px1s, ids2, use_max_loc=False):
        px2s_list, occlusions_list = [], []
        for (id1, id2, px1) in zip(ids1, ids2, px1s):
            px1s_normed = util.normalize_coords(px1, self.h, self.w)  # [num_pts, 2]
            target_tracks = self.video_3d_flow.load_target_tracks(id1, [id2], dim=0)  # [NumT, NumPoints, 4]
            target_visibles, target_invisibles, target_confidences = \
                parse_tapir_track_info(target_tracks[..., 2], target_tracks[..., 3])
            target_tracks = target_tracks[..., :2]  # [NumT, NumPoints, 2]
            # resize_h, resize_w = math.ceil(self.h / 4), math.ceil(self.w / 4)
            resize_h, resize_w = math.ceil(self.h), math.ceil(self.w)
            target_tracks = target_tracks.reshape(1,resize_h,resize_w,2).permute(0,3,1,2).to(self.device)
            target_invisibles = target_invisibles.reshape(1,resize_h,resize_w,1).permute(0,3,1,2).float().to(self.device)
            px2s = F.grid_sample(target_tracks, px1s_normed[None,:,None,:], align_corners=True)  # [1,2,N,1]
            px2s_occ = F.grid_sample(target_invisibles, px1s_normed[None,:,None,:], align_corners=True)  # [1,1,N,1]
            px2s = px2s[0,...,0].permute(1,0)
            px2s_occ = px2s_occ[0,...,0].permute(1,0)  # [N,1]
            px2s_list.append(px2s)
            occlusions_list.append(px2s_occ)
        return px2s_list, occlusions_list

    def get_correspondences_and_occlusion_masks_for_pixels_v0(self, ids1, px1s, ids2, use_max_loc=False):
        px2s, occlusions = [], []
        for (id1, id2, px1) in zip(ids1, ids2, px1s):
            ### 
            px2, occlusion = self.get_correspondences_and_occlusion_masks_for_pixels_core(id1, px1, id2, use_max_loc)
            px2s.append(px2)
            occlusions.append(occlusion)
        # return torch.stack(px2s, dim=0), torch.stack(occlusions, dim=0)
        return px2s, occlusions

    def get_correspondences_and_occlusion_masks_for_pixels_core(self, ids1, px1s, ids2,
                                                           use_max_loc=False):
        # ids1: int, px1s: [num_pts, 2], ids2: int
        # return px2s: [num_pts,2], occlusion: [num_pts,1]
        render_dict = self.gs_atlases_model.forward(ids1)
        render_dict2 = self.gs_atlases_model.forward(ids2)

        track_gs = render_dict2["position"]
        render_dict.update({"track_gs": track_gs})  # TODO add flow
        batch_dict_copy = self.batch_dict.copy()
        batch_dict_copy.update({"render_attributes_list" : ["track_gs"]+list(self.cfg.render_attributes.keys())})
        render_results = self.renderer.render_batch(render_dict, [batch_dict_copy])

        predicted_track_gs = render_results['track_gs'].permute(0,2,3,1)  # [1, h, w, 3]
        predicted_track_2d = util.denormalize_coords(predicted_track_gs[...,:2], self.h, self.w)  # [1, h, w, 2]
        predicted_track_2d = predicted_track_2d.permute(0,3,1,2)  # [1, 2, h, w]

        normed_px1s = util.normalize_coords(px1s, self.h, self.w)  # [num_pts, 2]
        px2s_pred = F.grid_sample(predicted_track_2d, normed_px1s[None,None], align_corners=True)[0,:,0,:]  # [1,2,1,num_pts]
        px2s_pred = px2s_pred.permute(1,0)

        predicted_track_depth = render_results['track_gs'][:,2:3]  # [1, 1, h, w]
        depth_proj = F.grid_sample(predicted_track_depth, normed_px1s[None,None], align_corners=True)[0,:,0,:]  # [1,1,1,num_pts]
        depth_proj = depth_proj.permute(1,0)


        ##### directly render the depth
        batch_dict_copy = self.batch_dict.copy()
        render_results2 = self.renderer.render_batch(render_dict2, [batch_dict_copy])
        depth2 = render_results2['depth']  # [1, 1, h, w]
        px2s_pred_depth = F.grid_sample(depth2, px2s_pred[None,None], align_corners=True)[0,:,0,:]  # [1,1,1,num_pts]
        px2s_pred_depth = px2s_pred_depth.permute(1,0)
        occlusion = (px2s_pred_depth >= depth_proj).float()
        return px2s_pred, occlusion