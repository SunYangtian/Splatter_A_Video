import torch
from dataclasses import dataclass, field
from typing import Mapping, Optional, Union, Any
from omegaconf import DictConfig
from pytorch_msssim import ms_ssim

from pointrix.utils.base import BaseModule
from pointrix.utils.config import parse_structured
from pointrix.point_cloud import parse_point_cloud
from pointrix.model.loss import l1_loss, ssim, psnr
from pointrix.model.base_model import BaseModel, MODEL_REGISTRY
from pointrix.point_cloud import POINTSCLOUD_REGISTRY
import numpy as np
from pointrix.utils.dataset.dataset_utils import fov2focal, focal2fov
from pointrix.dataset.base_data import SimplePointCloud
from dynamic_gaussian_points import DynamicGaussianPointCloud
from dynamic_gaussian_with_base_point_cloud import DynamicGaussianWithBasePointCloud
# from lbs_gaussian_point import LBSGaussianPointCloud
# from dust3r_interface import Dust3R
from itertools import accumulate
import operator


class SingleAtlasModel(BaseModel):
    """
    A class for the Single Atlas Model.
    """
    @dataclass
    class Config:
        point_cloud: dict = field(default_factory=dict)
        lambda_dssim: float = 0.2

    cfg: Config

    def setup(self, gs_atlas_cfg, point_cloud=None, device="cuda"):
        self.gs_atlas_cfg = gs_atlas_cfg
        self.point_cloud = DynamicGaussianPointCloud(self.cfg.point_cloud, gs_atlas_cfg, point_cloud).to(device)
        self.render_attributes_list = list(gs_atlas_cfg.render_attributes.keys())

    def forward(self, ids, batch=None) -> dict:
        """
        Forward pass of the model.

        Parameters
        ----------
        batch : dict
            The batch of data.
        
        Returns
        -------
        dict
            The render results which will be the input of renderers.
        """
        position = self.point_cloud.get_position(ids)
        detached_position = self.point_cloud.get_position(ids, detach_pos=True)
        shs = self.point_cloud.get_shs
        opacity = self.point_cloud.get_opacity
        rotation = self.point_cloud.get_rotation(ids)
        scaling = self.point_cloud.get_scaling
        pos_poly_feat = self.point_cloud.get_pos_poly_feat
        pos_fourier_feat = self.point_cloud.get_pos_fourier_feat
        rot_poly_feat = self.point_cloud.get_rot_poly_feat
        rot_fourier_feat = self.point_cloud.get_rot_fourier_feat
        render_dict = {
            "position": position,
            "detached_position": detached_position,
            "opacity": opacity,
            "scaling": scaling,
            "rotation": rotation,
            "shs": shs,
            "pos_poly_feat": pos_poly_feat.reshape(-1, pos_poly_feat.shape[-1]*pos_poly_feat.shape[-2]),
            "pos_fourier_feat": pos_fourier_feat.reshape(-1, pos_fourier_feat.shape[-1]*pos_fourier_feat.shape[-2]),
            "rot_poly_feat": rot_poly_feat.reshape(-1, rot_poly_feat.shape[-1]*rot_poly_feat.shape[-2]),
            "rot_fourier_feat": rot_fourier_feat.reshape(-1, rot_fourier_feat.shape[-1]*rot_fourier_feat.shape[-2]),
        }
        for attr in self.render_attributes_list:
            if attr not in render_dict:
                render_dict[attr] = getattr(self.point_cloud, f"get_{attr}")
        return render_dict


class SingleAtlasWithBaseModel(BaseModel):
    """
    A class for the Single Atlas Model.
    """
    @dataclass
    class Config:
        point_cloud: dict = field(default_factory=dict)
        lambda_dssim: float = 0.2

    cfg: Config

    def setup(self, gs_atlas_cfg, base_point_seq=None, device="cuda"):
        self.gs_atlas_cfg = gs_atlas_cfg
        self.point_cloud = DynamicGaussianWithBasePointCloud(self.cfg.point_cloud, gs_atlas_cfg, base_point_seq).to(device)
        self.render_attributes_list = list(gs_atlas_cfg.render_attributes.keys())

    def forward(self, ids, batch=None) -> dict:
        """
        Forward pass of the model.

        Parameters
        ----------
        batch : dict
            The batch of data.
        
        Returns
        -------
        dict
            The render results which will be the input of renderers.
        """
        position = self.point_cloud.get_position(ids)
        detached_position = self.point_cloud.get_position(ids, detach_pos=True)
        shs = self.point_cloud.get_shs
        opacity = self.point_cloud.get_opacity
        rotation = self.point_cloud.get_rotation(ids)
        scaling = self.point_cloud.get_scaling
        pos_poly_feat = self.point_cloud.get_pos_poly_feat
        pos_fourier_feat = self.point_cloud.get_pos_fourier_feat
        rot_poly_feat = self.point_cloud.get_rot_poly_feat
        rot_fourier_feat = self.point_cloud.get_rot_fourier_feat
        render_dict = {
            "position": position,
            "detached_position": detached_position,
            "opacity": opacity,
            "scaling": scaling,
            "rotation": rotation,
            "shs": shs,
            "pos_poly_feat": pos_poly_feat.reshape(-1, pos_poly_feat.shape[-1]*pos_poly_feat.shape[-2]),
            "pos_fourier_feat": pos_fourier_feat.reshape(-1, pos_fourier_feat.shape[-1]*pos_fourier_feat.shape[-2]),
            "rot_poly_feat": rot_poly_feat.reshape(-1, rot_poly_feat.shape[-1]*rot_poly_feat.shape[-2]),
            "rot_fourier_feat": rot_fourier_feat.reshape(-1, rot_fourier_feat.shape[-1]*rot_fourier_feat.shape[-2]),
        }
        for attr in self.render_attributes_list:
            if attr not in render_dict:
                render_dict[attr] = getattr(self.point_cloud, f"get_{attr}")
        return render_dict
    

class SingleAtlasLBSModel(BaseModel):
    """
    A class for the Single Atlas Model.
    """
    @dataclass
    class Config:
        point_cloud: dict = field(default_factory=dict)
        lambda_dssim: float = 0.2

    cfg: Config

    def setup(self, gs_atlas_cfg, base_point_seq, device="cuda"):
        self.gs_atlas_cfg = gs_atlas_cfg
        self.point_cloud = LBSGaussianPointCloud(self.cfg.point_cloud, gs_atlas_cfg, base_point_seq).to(device)
        self.render_attributes_list = list(gs_atlas_cfg.render_attributes.keys())

    def forward(self, ids, batch=None) -> dict:
        """
        Forward pass of the model.

        Parameters
        ----------
        batch : dict
            The batch of data.
        
        Returns
        -------
        dict
            The render results which will be the input of renderers.
        """
        position = self.point_cloud.get_position(ids)
        # delta_pos = 
        
        
        detached_position = position.detach()



        shs = self.point_cloud.get_shs
        opacity = self.point_cloud.get_opacity
        rotation = self.point_cloud.get_rotation(ids)
        scaling = self.point_cloud.get_scaling
        pos_poly_feat = self.point_cloud.get_pos_poly_feat
        pos_fourier_feat = self.point_cloud.get_pos_fourier_feat
        rot_poly_feat = self.point_cloud.get_rot_poly_feat
        rot_fourier_feat = self.point_cloud.get_rot_fourier_feat
        render_dict = {
            "position": position,
            "detached_position": detached_position,
            "opacity": opacity,
            "scaling": scaling,
            "rotation": rotation,
            "shs": shs,
            "pos_poly_feat": pos_poly_feat.reshape(-1, pos_poly_feat.shape[-1]*pos_poly_feat.shape[-2]),
            "pos_fourier_feat": pos_fourier_feat.reshape(-1, pos_fourier_feat.shape[-1]*pos_fourier_feat.shape[-2]),
            "rot_poly_feat": rot_poly_feat.reshape(-1, rot_poly_feat.shape[-1]*rot_poly_feat.shape[-2]),
            "rot_fourier_feat": rot_fourier_feat.reshape(-1, rot_fourier_feat.shape[-1]*rot_fourier_feat.shape[-2]),
        }
        for attr in self.render_attributes_list:
            if attr not in render_dict:
                render_dict[attr] = getattr(self.point_cloud, f"get_{attr}")
        return render_dict


@MODEL_REGISTRY.register()
class FragModel(BaseModel):
    """
    A class for the Fragmentation Model.
    """
    @dataclass
    class Config:
        point_cloud: dict = field(default_factory=dict)
        lambda_dssim: float = 0.2

    cfg: Config

    def setup_bak(self, *gs_atlas_cfg_list, white_bg=True):
        self.gs_atlas_cfg_list = gs_atlas_cfg_list
        self.dust3r = Dust3R("pretrained_weights/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
                             niter=100)
        # Initialize the point cloud
        self.atlas_dict = {}
        for gs_atlas_cfg in self.gs_atlas_cfg_list:
            name = gs_atlas_cfg.name
            im_path1, im_path2 = gs_atlas_cfg.start_frame_path, gs_atlas_cfg.end_frame_path
            dust3r_dict = self.dust3r.infer(im_path1, im_path2)
            point_cloud = SimplePointCloud(dust3r_dict['pcd'].cpu().numpy(), dust3r_dict['color'].cpu().numpy(), normals=None)
            self.atlas_dict[name] = SingleAtlasModel(self.cfg, gs_atlas_cfg, point_cloud)
        self.focal_y_ratio = dust3r_dict['focals'].mean() / dust3r_dict['H']
        self.white_bg = white_bg 

    def setup_randInit(self, *gs_atlas_cfg_list, white_bg=True):
        self.gs_atlas_cfg_list = gs_atlas_cfg_list
        # Initialize the point cloud
        self.atlas_dict = {}
        point_cloud = None
        for gs_atlas_cfg in self.gs_atlas_cfg_list:
            name = gs_atlas_cfg.name
            self.atlas_dict[name] = SingleAtlasModel(self.cfg, gs_atlas_cfg, point_cloud)
        self.focal_y_ratio = 1.0
        self.white_bg = white_bg

    def setup(self, gs_atlas_cfg_list, base_point_seq_list, white_bg=True):
        base_point_seq = torch.cat(base_point_seq_list, dim=1)
        self.gs_atlas_cfg_list = gs_atlas_cfg_list
        self.atlas_dict = {}
        for gs_atlas_cfg in self.gs_atlas_cfg_list:
            name = gs_atlas_cfg.name
            if name == "gs_base":
                self.atlas_dict[name] = SingleAtlasWithBaseModel(self.cfg, gs_atlas_cfg, base_point_seq)
                # self.atlas_dict[name] = SingleAtlasWithBaseModel(self.cfg, gs_atlas_cfg, base_point_seq_list[1])
            elif name == "gs_fg":
                self.atlas_dict[name] = SingleAtlasLBSModel(self.cfg, gs_atlas_cfg, base_point_seq_list[0])
            elif name == "gs_bg":
                self.atlas_dict[name] = SingleAtlasLBSModel(self.cfg, gs_atlas_cfg, base_point_seq_list[1])
            else:
                raise ValueError(f"Unknown atlas name: {name}")
        self.focal_y_ratio = 1.0
        self.white_bg = white_bg


    def get_atlas(self, name):
        """
        Query the Gaussian Splatting Atlas by name.

        Parameters
        ----------
        name : str
            The name of the Gaussian Splatting Atlas.
        """
        return self.atlas_dict[name]


    def forward(self, ids, batch=None) -> dict:
        """
        Forward pass of the model.

        Parameters
        ----------
        batch : dict
            The batch of data.
        
        Returns
        -------
        dict
            The render results which will be the input of renderers.
        """


        atls_render_dict_list = [self.atlas_dict[name].forward(ids) for name in self.atlas_dict]
        render_dict = {}
        for atls_render_dict in atls_render_dict_list:
            for key, value in atls_render_dict.items():
                if key not in render_dict:
                    render_dict[key] = value
                else:
                    render_dict[key] = torch.cat([render_dict[key], value], dim=0)

        return render_dict
    

    def forward_single_atlas(self, ids, name) -> dict:
        """
        Forward pass of the model for a single atlas.

        Parameters
        ----------
        name : str
            The name of the Gaussian Splatting Atlas.
        
        Returns
        -------
        dict
            The render results which will be the input of renderers.
        """
        render_dict = self.atlas_dict[name].forward(ids)
        return render_dict

    
    def get_point_num_sep(self):
        """
        Get the number of points in the point cloud.
        """
        point_num_list = [0]+[len(self.atlas_dict[name].point_cloud) for name in self.atlas_dict]
        return list(accumulate(point_num_list, operator.add))
    
    def prepare_optimizer_dict(self, loss, render_results):
        """
        Prepare the optimizer dictionary.
        """
        optimizer_dict = {}
        ptnum = self.get_point_num_sep()
        for idx, gs_atlas_cfg in enumerate(self.gs_atlas_cfg_list):
            name = gs_atlas_cfg.name
            optimizer_dict[name] = {
                "viewspace_points": [x[ptnum[idx]:ptnum[idx+1]] for x in render_results["viewspace_points"]],
                "viewspace_points_grad": [x.grad[ptnum[idx]:ptnum[idx+1]] for x in render_results["viewspace_points"]],
                "visibility": render_results["visibility"][ptnum[idx]:ptnum[idx+1]],
                "radii": render_results["radii"][ptnum[idx]:ptnum[idx+1]],
                "white_bg": self.white_bg,
            }
            for x in optimizer_dict[name]["viewspace_points"]:
                x.retain_grad()
        return optimizer_dict
    
    def get_state_dict(self):
        additional_info = {k: v.get_state_dict() for k, v in self.atlas_dict.items()}
        return {**super().state_dict(), **additional_info}
    
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):

        for name, model in self.atlas_dict.items():
            model.load_state_dict(state_dict[name], strict)
            model.to('cuda')


