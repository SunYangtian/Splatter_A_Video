import torch
from torch import nn
from pytorch_msssim import ms_ssim
from dataclasses import dataclass

from pointrix.point_cloud import PointCloud, POINTSCLOUD_REGISTRY
from pointrix.utils.gaussian_points.gaussian_utils import (
    build_covariance_from_scaling_rotation,
    inverse_sigmoid,
    gaussian_point_init
)

from pointrix.utils.dataset.dataset_utils import fov2focal, focal2fov
import numpy as np
import imageio
from pointrix.dataset.base_data import SimplePointCloud
def depth2pcd(depth, shift=0.1):
    """
    Convert the depth map to point cloud.

    Parameters
    ----------
    depth: np.ndarray
        The depth map.
    """
    h, w = depth.shape
    i, j = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    z = depth + shift  ### add a shift value to avoid zero depth
    x = (j - w * 0.5) / (0.5 * w)  # normalize to [-1, 1]
    y = (i - h * 0.5) / (0.5 * h)
    pcd = np.stack([x, y, z], axis=-1)
    return pcd



@POINTSCLOUD_REGISTRY.register()
class DynamicGaussianPointCloud(PointCloud):
    """
    A class for Gaussian point cloud.

    Parameters
    ----------
    PointCloud : PointCloud
        The point cloud for initialisation.
    """
    @dataclass
    class Config(PointCloud.Config):
        max_sh_degree: int = 3
        lambda_dssim: float = 0.2

    cfg: Config

    def setup(self, gs_atlas_cfg, point_cloud=None):
        # ##### init point cloud use depth
        # depth_map = np.load(gs_atlas_cfg.start_depth_npy)
        # depth_map = depth_map + np.random.randn(*depth_map.shape) * 0.01
        # pcd = depth2pcd(depth_map)
        # image = imageio.imread(gs_atlas_cfg.start_frame_path)[..., :3] / 255.
        # mask = imageio.imread(gs_atlas_cfg.start_frame_mask_path) > 0
        # mask = np.ones_like(mask, dtype=bool)
        # if gs_atlas_cfg.reverse_mask:
        #     mask = ~mask
        # ##### use downsample to reduce the number of points
        # pcd = pcd[mask][::5]
        # colors = image[mask][::5]
        # point_cloud = SimplePointCloud(positions=pcd, colors=colors, normals=None)

        super().setup(point_cloud)
        self.gs_atlas_cfg = gs_atlas_cfg
        self.start_frame_id = gs_atlas_cfg.start_frame_id
        self.end_frame_id = gs_atlas_cfg.end_frame_id
        self.time_len = gs_atlas_cfg.num_images - 1
        # self.mask = torch.from_numpy(mask).to(self.position.device)

        # Activation funcitons
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

        scales, rots, opacities, features_rest = gaussian_point_init(
            position=self.position,
            max_sh_degree=self.cfg.max_sh_degree,
            init_opacity=0.01
        )

        ######## add dyanmic attributes
        num_points = len(self.position)
        self.poly_feature_dim = 4
        self.fourier_feature_dim = 4 * 2
        pos_poly_feat = torch.zeros((num_points, self.poly_feature_dim, 3))
        pos_fourier_feat = torch.zeros((num_points, self.fourier_feature_dim, 3))
        rot_poly_feat = torch.zeros((num_points, self.poly_feature_dim, 4))
        rot_fourier_feat = torch.zeros((num_points, self.fourier_feature_dim, 4))

        fused_color = self.features.unsqueeze(1)
        self.features = (
            nn.Parameter(
                fused_color.contiguous().requires_grad_(True)
            )
        )
        self.register_atribute("features_rest", features_rest)
        self.register_atribute("scaling", scales)
        self.register_atribute("rotation", rots)
        self.register_atribute("opacity", opacities)
        self.register_atribute("pos_poly_feat", pos_poly_feat)
        self.register_atribute("pos_fourier_feat", pos_fourier_feat)
        self.register_atribute("rot_poly_feat", rot_poly_feat)
        self.register_atribute("rot_fourier_feat", rot_fourier_feat)

        ######### add image attributes to GS
        for attrib_name, feature_dim in gs_atlas_cfg.render_attributes.items():
            if attrib_name in ["pos_poly_feat", "pos_fourier_feat", "rot_poly_feat", "rot_fourier_feat"]:
                continue
            attribute = torch.zeros((num_points, feature_dim))
            self.register_atribute(attrib_name, attribute)
            if attrib_name == "mask_attribute":
                self.mask_attribute_activation = torch.sigmoid
            if attrib_name == "dino_attribute":
                self.dino_attribute_activation = torch.sigmoid

    

    @property
    def get_opacity(self):
        return self.opacity_activation(self.opacity)

    @property
    def get_scaling(self):
        return self.scaling_activation(self.scaling)

    # @property
    # def get_rotation(self):
    #     return self.rotation_activation(self.rotation)

    def get_rotation(self, time):
        rotation = self.rotation
        normed_time = (time - self.start_frame_id) / self.time_len
        basis = torch.arange(self.poly_feature_dim).float().to(rotation.device)
        poly_basis = torch.pow(normed_time, basis)[None, :, None]

        # al * cos(lt) + bl * sin(lt)
        basis = torch.arange(self.fourier_feature_dim/2).float().to(rotation.device) + 1
        fourier_basis = [torch.cos(normed_time * basis * np.pi), torch.sin(normed_time * basis * np.pi)]
        fourier_basis = torch.cat(fourier_basis, dim=0)[None, :, None]
                            
        rotation = rotation \
            + torch.sum(self.rot_poly_feat * poly_basis, dim=1).detach() \
            + torch.sum(self.rot_fourier_feat * fourier_basis, dim=1).detach()
        return self.rotation_activation(rotation)


    @property
    def get_covariance(self, scaling_modifier=1):
        ### Not called in dptr
        return self.covariance_activation(
            self.get_scaling,
            scaling_modifier,
            self.get_rotation,
        )

    @property
    def get_shs(self):
        return torch.cat([
            self.features, self.features_rest,
        ], dim=1)

    def get_position(self, time, detach_pos=False):
        position = self.position
        normed_time = (time - self.start_frame_id) / self.time_len
        basis = torch.arange(self.poly_feature_dim).float().to(position.device)
        poly_basis = torch.pow(normed_time, basis)[None, :, None]

        # al * cos(lt) + bl * sin(lt)
        basis = torch.arange(self.fourier_feature_dim/2).float().to(position.device) + 1
        fourier_basis = [torch.cos(normed_time * basis * np.pi), torch.sin(normed_time * basis * np.pi)]
        fourier_basis = torch.cat(fourier_basis, dim=0)[None, :, None]
                            
        if detach_pos:
            return position.detach() + torch.sum(self.pos_poly_feat * poly_basis, dim=1) + torch.sum(self.pos_fourier_feat * fourier_basis, dim=1)
        else:
            return position \
                + torch.sum(self.pos_poly_feat * poly_basis, dim=1) \
                + torch.sum(self.pos_fourier_feat * fourier_basis, dim=1)
    
    @property
    def get_pos_poly_feat(self):
        return self.pos_poly_feat
    
    @property
    def get_pos_fourier_feat(self):
        return self.pos_fourier_feat
    
    @property
    def get_rot_poly_feat(self):
        return self.rot_poly_feat
    
    @property
    def get_rot_fourier_feat(self):
        return self.rot_fourier_feat
    
    @property
    def get_mask_attribute(self):
        return self.mask_attribute_activation(self.mask_attribute)
    
    @property
    def get_dino_attribute(self):
        return self.dino_attribute_activation(self.dino_attribute)


    def re_init(self, num_points):
        super().re_init(num_points)
        fused_color = self.features.unsqueeze(1)
        self.features = (
            nn.Parameter(
                fused_color.contiguous().requires_grad_(True)
            )
        )
        scales, rots, opacities, features_rest = gaussian_point_init(
            position=self.position,
            max_sh_degree=self.cfg.max_sh_degree,
        )
        self.register_atribute("features_rest", features_rest)
        self.register_atribute("scaling", scales)
        self.register_atribute("rotation", rots)
        self.register_atribute("opacity", opacities)

        ######## add dyanmic attributes
        num_points = len(self.position)
        pos_poly_feat = torch.zeros((num_points, self.poly_feature_dim, 3))
        self.register_atribute("pos_poly_feat", pos_poly_feat)
        pos_fourier_feat = torch.zeros((num_points, self.fourier_feature_dim, 3))
        self.register_atribute("pos_fourier_feat", pos_fourier_feat)
        rot_poly_feat = torch.zeros((num_points, self.poly_feature_dim, 4))
        self.register_atribute("rot_poly_feat", rot_poly_feat)
        rot_fourier_feat = torch.zeros((num_points, self.fourier_feature_dim, 4))
        self.register_atribute("rot_fourier_feat", rot_fourier_feat)

        ######### add image attributes to GS
        for attrib_name, feature_dim in self.gs_atlas_cfg.render_attributes.items():
            if attrib_name in ["pos_poly_feat", "pos_fourier_feat", "rot_poly_feat", "rot_fourier_feat"]:
                continue
            attribute = torch.zeros((num_points, feature_dim))
            self.register_atribute(attrib_name, attribute)
            if attrib_name == "mask_attribute":
                self.mask_attribute_activation = torch.sigmoid
            if attrib_name == "dino_attribute":
                self.dino_attribute_activation = torch.sigmoid