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
from pointrix.point_cloud.utils import get_random_feauture, get_random_points

from pointrix.utils.dataset.dataset_utils import fov2focal, focal2fov
import numpy as np
import imageio
from pointrix.dataset.base_data import SimplePointCloud
import math
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
class DynamicGaussianWithBasePointCloud(PointCloud):
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

    def setup(self, gs_atlas_cfg, base_point_seq=None):
        base_point_seq = torch.stack([x[~torch.isnan(x).any(dim=1)] for x in base_point_seq], dim=0)
        ### assume first frame is the base frame
        self.base_position = base_point_seq[0]
        self.delta_position = base_point_seq - self.base_position[None,...]
        # split for Lagrange interpolation
        if False:
            self.interp_order = 5
            self.interp_nodes = torch.linspace(0, len(self.delta_position)-1, self.interp_order+1).long().to(self.base_position.device)
            self.prod_withouti = lambda l, i: torch.prod(torch.stack([l[j] for j in range(len(l)) if j != i]))
        # cubic spline interpolation
        self.interval_num = math.ceil(len(base_point_seq) / 5)  # set a node every 5 frames
        self.intervals_idx = torch.linspace(0, len(self.delta_position)-1, self.interval_num+1).long().to(self.base_position.device)
        self.intervals = self.intervals_idx / (len(self.delta_position)-1)  # between 0 and 1
        delta_position = self.delta_position.cpu().numpy()  # [Nt, Np, 3]
        from scipy.interpolate import CubicSpline
        xx = self.intervals.cpu().numpy()
        coeff_list = []
        for yy in delta_position[self.intervals_idx.cpu().numpy()].transpose(1,0,2):
            cs = CubicSpline(xx, yy)
            coeff_list.append(cs.c)  # [4,interval_num]
        
        coeff_list = np.array(coeff_list)  # [Np, 4, interval_num, 3]
        self.cubic_coeff = torch.tensor(coeff_list).to(self.base_position.device).float()  # [Np, 4, interval_num, 3]

        # super().setup(point_cloud)
        ######################## setup of point cloud
        self.atributes = []
        position = self.base_position
        features = get_random_feauture(len(position), self.cfg.initializer.feat_dim)
        self.register_buffer('position', position)
        self.register_buffer('features', features)
        self.atributes.append({
            'name': 'position',
            'trainable': self.cfg.trainable,
        })
        self.atributes.append({
            'name': 'features',
            'trainable': self.cfg.trainable,
        })
        
        if self.cfg.trainable:
            self.position = nn.Parameter(
                position.contiguous().requires_grad_(False)
            )
            self.features = nn.Parameter(
                features.contiguous().requires_grad_(True)
            )

        self.prefix_name = self.cfg.unwarp_prefix + "."
        ######################## ########################

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
            # init_opacity=0.01
            init_opacity=0.5
        )

        ##### set scale threshold
        self.scale_threshold = 0.1

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

        # self.register_atribute("pos_lagrange_node", self.delta_position[self.interp_nodes].permute(1,0,2).reshape(-1, (self.interp_order+1)*3))
        self.register_atribute("pos_cubic_node", self.cubic_coeff.reshape(-1, 4*self.interval_num*3))

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
        # return torch.min(self.scaling_activation(self.scaling), torch.ones_like(self.scaling)*self.scale_threshold)
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
        
    # def get_position(self, time, detach_pos=False):
    #     time_index = time - self.start_frame_id
    #     if isinstance(time, torch.Tensor):
    #         time_index = time_index.item()
    #     delta_position = self.delta_position[time_index]
    #     return self.base_position + delta_position
    
    ##### for Lagrange
    def get_position_Larange(self, time, detach_pos=False):
        time_index = time - self.start_frame_id
        if isinstance(time, torch.Tensor):
            time_index = time_index.item()
        ### use Lagrange interpolation
        out_pos_list = []
        for ii in range(len(self.interp_nodes)):
            numerator = self.prod_withouti(time-self.interp_nodes, ii)
            denominator = self.prod_withouti(self.interp_nodes[ii]-self.interp_nodes, ii)
            out_pos_list.append(numerator/denominator * self.pos_lagrange_node.reshape(-1,self.interp_order+1,3)[:,ii])
        return self.position + torch.stack(out_pos_list).sum(dim=0)
    
    def get_position(self, time, detach_pos=False):
        # time_index = time - self.start_frame_id TODO pay attention to this !
        if isinstance(time, torch.Tensor):
            time = time.item()
        normed_time = time / (len(self.delta_position)-1)
        ### use cubic spline interpolation
        coeff = self.pos_cubic_node.reshape(-1, 4, self.interval_num, 3)
        # minus 1e-7 to avoid numeric error
        indices = torch.searchsorted(self.intervals, normed_time-1e-7, right=False) - 1
        indices = torch.clamp(indices, min=0)
        distances = normed_time - self.intervals[indices]  # [1,]
        pos = coeff[:,3, indices] + coeff[:,2, indices] * distances + \
            coeff[:,1, indices] * distances**2 + coeff[:,0, indices] * distances**3  # [Np, 1, 3]
        # return pos[:,0,:]
        return pos + self.position


        # normed_time = (time - self.start_frame_id) / self.time_len
        # basis = torch.arange(self.fourier_feature_dim/2).float().to(self.position.device) + 1
        # fourier_basis = [torch.cos(normed_time * basis * np.pi), torch.sin(normed_time * basis * np.pi)]
        # fourier_basis = torch.cat(fourier_basis, dim=0)[None, :, None]
        # return self.position + torch.stack(out_pos_list).sum(dim=0) + torch.sum(self.pos_fourier_feat * fourier_basis, dim=1)
        
    
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
            init_opacity=0.5
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

        # pos_lagrange_node = torch.zeros((num_points, (self.interp_order+1)*3))
        # self.register_atribute("pos_lagrange_node", pos_lagrange_node)
        cubic_coeff = torch.zeros((num_points, 4*self.interval_num*3))
        self.register_atribute("pos_cubic_node", cubic_coeff)

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