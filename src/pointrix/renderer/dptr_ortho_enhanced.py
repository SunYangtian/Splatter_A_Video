import torch
import numpy as np
import dptr.gs as gs
from torch import Tensor
from typing import Tuple
from jaxtyping import Float, Int

from typing import List
from dataclasses import dataclass
from pointrix.utils.base import BaseObject
from pointrix.utils.renderer.renderer_utils import RenderFeatures
from pointrix.utils.registry import Registry
from .dptr import RENDERER_REGISTRY

BLOCK_X = 16
BLOCK_Y = 16

def ewa_project_torch_impl(
    xyz,
    cov3d, 
    extr,
    xy,
    W, H,
    visible
):
    Wmat = extr[..., :3, :3]
    p = extr[..., :3, 3]

    t = torch.matmul(Wmat, xyz[..., None])[..., 0] + p
    rz = 1.0 / t[..., 2]
    rz2 = rz**2
    
    
    Jmat = torch.stack(
        [
            torch.stack([W/2+torch.zeros_like(rz), torch.zeros_like(rz), torch.zeros_like(rz)], dim=-1),
            torch.stack([torch.zeros_like(rz), H/2+torch.zeros_like(rz), torch.zeros_like(rz)], dim=-1),
        ],
        dim=-2,
    )
    
    T = Jmat @ Wmat
    cov3d_1 = torch.stack([cov3d[..., 0], cov3d[..., 1], cov3d[..., 2]], dim=-1)
    cov3d_2 = torch.stack([cov3d[..., 1], cov3d[..., 3], cov3d[..., 4]], dim=-1)
    cov3d_3 = torch.stack([cov3d[..., 2], cov3d[..., 4], cov3d[..., 5]], dim=-1)
    cov3d = torch.stack([cov3d_1, cov3d_2, cov3d_3], dim=-1)
    
    cov2d = T @ cov3d @ T.transpose(-1, -2)
    cov2d[..., 0, 0] = cov2d[:, 0, 0] + 0.3
    cov2d[..., 1, 1] = cov2d[:, 1, 1] + 0.3

    # Compute extent in screen space
    det = cov2d[..., 0, 0] * cov2d[..., 1, 1] - cov2d[..., 0, 1] ** 2
    det_mask = det != 0
        
    conic = torch.stack(
        [
            cov2d[..., 1, 1] / det,
            -cov2d[..., 0, 1] / det,
            cov2d[..., 0, 0] / det,
        ],
        dim=-1,
    )
    
    b = (cov2d[..., 0, 0] + cov2d[..., 1, 1]) / 2
    v1 = b + torch.sqrt(torch.clamp(b**2 - det, min=0.1))
    v2 = b - torch.sqrt(torch.clamp(b**2 - det, min=0.1))
    radius = torch.ceil(3.0 * torch.sqrt(torch.max(v1, v2)))
    
    # get tiles
    top_left = torch.zeros_like(xy, dtype=torch.int, device=xyz.device)
    bottom_right = torch.zeros_like(xy, dtype=torch.int, device=xyz.device)
    top_left[:, 0] = ((xy[:, 0] - radius) / BLOCK_X)
    top_left[:, 1] = ((xy[:, 1] - radius) / BLOCK_Y)
    bottom_right[:, 0] = ((xy[:, 0] + radius + BLOCK_X - 1) / BLOCK_X)
    bottom_right[:, 1] = ((xy[:, 1] + radius + BLOCK_Y - 1) / BLOCK_Y)
    
    tile_bounds = torch.zeros(2, dtype=torch.int, device=xyz.device)
    tile_bounds[0] = (W + BLOCK_X - 1) / BLOCK_X
    tile_bounds[1] = (H + BLOCK_Y - 1) / BLOCK_Y
    
    tile_min = torch.stack(
        [
            torch.clamp(top_left[..., 0], 0, tile_bounds[0]),
            torch.clamp(top_left[..., 1], 0, tile_bounds[1]),
        ],
        -1,
    )
    tile_max = torch.stack(
        [
            torch.clamp(bottom_right[..., 0], 0, tile_bounds[0]),
            torch.clamp(bottom_right[..., 1], 0, tile_bounds[1]),
        ],
        -1,
    )
    
    tiles_tmp = tile_max - tile_min
    tiles = tiles_tmp[..., 0] * tiles_tmp[..., 1]
    
    mask = torch.logical_and(tiles != 0, det_mask)
    mask = torch.logical_and(visible, mask)
    
    conic = torch.nan_to_num(conic)
    radius = torch.nan_to_num(radius)
    tiles = torch.nan_to_num(tiles)

    conic = conic * mask.float()[..., None]
    radius = radius * mask.float()
    tiles = tiles * mask.float()
    
    return conic, radius.int(), tiles.int()


@RENDERER_REGISTRY.register()
class DPTROrthoEnhancedRender(BaseObject):
    """
    A class for rendering point clouds using DPTR.

    Parameters
    ----------
    cfg : dict
        The configuration dictionary.
    white_bg : bool
        Whether the background is white or not.
    device : str
        The device to use.
    update_sh_iter : int, optional
        The iteration to update the spherical harmonics degree, by default 1000.
    """

    @dataclass
    class Config:
        update_sh_iter: int = 1000
        max_sh_degree: int = 3
        densify_abs_grad_enable: bool = False
    
    cfg: Config

    def setup(self, white_bg, device, **kwargs):
        self.active_sh_degree = 0
        self.device = device
        super().setup(white_bg, device, **kwargs)
        self.bg_color = 1. if white_bg else 0.

    def project_point(
        self, 
        xyz: Float[Tensor, "P 3"],
        extr: Float[Tensor, "3 4"],
        W: int,
        H: int,
        nearest: float = 0.2,
        extent: float = 1.3,
    ) -> Tuple:
        """
        Project a point cloud into the image plane.

        Parameters
        ----------
        xyz : Float[Tensor, "P 3"]
            The point cloud.
        extr : Float[Tensor, "3 4"]
            The extrinsic matrix.
        W : int
            The width of the image.
        H : int
            The height of the image.
        nearest : float, optional
            The nearest point, by default 0.2
        extent : float, optional
            The extent, by default 1.3

        Returns
        -------
        Tuple
            The uv and depth.
        """
        R = extr[:3, :3]
        t = extr[:3, -1].unsqueeze(dim=1)

        pt_cam = torch.matmul(R, xyz.t()) + t  # 3 x P

        depth = pt_cam[2]
        uv = (pt_cam[:2] + 1.) * torch.tensor([[W], [H]], device=pt_cam.device) / 2

        uv = uv.t() - 0.5

        depth = torch.nan_to_num(depth)
        near_mask = depth <= nearest
        extent_mask_x = torch.logical_or(uv[:, 0] < (1 - extent) * W * 0.5, 
                                        uv[:, 0] > (1 + extent) * W * 0.5)
        extent_mask_y = torch.logical_or(uv[:, 1] < (1 - extent) * H * 0.5, 
                                        uv[:, 1] > (1 + extent) * H * 0.5)
        extent_mask = torch.logical_or(extent_mask_x, extent_mask_y)
        mask = torch.logical_or(near_mask, extent_mask)

        uv_masked = uv.clone()
        depth_masked = depth.clone()
        uv_masked[:, 0][mask] = 0
        uv_masked[:, 1][mask] = 0
        depth_masked[mask] = 0

        return uv_masked, depth_masked.unsqueeze(-1)
        

    def render_iter(self,
                    FovX,
                    FovY,
                    height,
                    width,
                    extrinsic_matrix,
                    intrinsic_matrix,
                    camera_center,
                    position,
                    opacity,
                    scaling,
                    rotation,
                    shs,
                    scaling_modifier=1.0,
                    render_xyz=False,
                    **kwargs) -> dict:
        """
        Render the point cloud for one iteration

        Parameters
        ----------
        FovX : float
            The field of view in the x-axis.
        FovY : float
            The field of view in the y-axis.
        height : float
            The height of the image.
        width : float
            The width of the image.
        world_view_transform : torch.Tensor
            The world view transformation matrix.
        full_proj_transform : torch.Tensor
            The full projection transformation matrix.
        camera_center : torch.Tensor
            The camera center.
        position : torch.Tensor
            The position of the point cloud.
        opacity : torch.Tensor
            The opacity of the point cloud.
        scaling : torch.Tensor
            The scaling of the point cloud.
        rotation : torch.Tensor
            The rotation of the point cloud.
        shs : torch.Tensor
            The spherical harmonics of the point cloud.
        scaling_modifier : float, optional
            The scaling modifier, by default 1.0
        render_xyz : bool, optional
            Whether to render the xyz or not, by default False

        Returns
        -------
        dict
            The rendered image, the viewspace points, 
            the visibility filter, the radii, the xyz, 
            the color, the rotation, the scales, and the xy.
        """
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        # import pdb; pdb.set_trace()
        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
    
        # direction = (position.cuda() -
        #              camera_center.repeat(position.shape[0], 1).cuda())
        # direction = direction / direction.norm(dim=1, keepdim=True)
        direction = torch.zeros_like(position).cuda()
        direction[:, 2] = 1.0
        rgb = gs.compute_sh(shs, 3, direction)

        # (uv, depth) = gs.project_point(
        #     position,
        #     intrinsic_matrix.cuda(),
        #     extrinsic_matrix.cuda(),
        #     width, height)

        enable_ortho_projection = kwargs.get("enable_ortho_projection", True)
        if enable_ortho_projection:
            (uv, depth) = self.project_point(
                position, 
                extrinsic_matrix.cuda(), 
                width, 
                height,
                nearest=0.01)
        else:
            (uv, depth) = gs.project_point(
                position,
                intrinsic_matrix.cuda(),
                extrinsic_matrix.cuda(),
                width, height)

        visible = depth != 0

        # compute cov3d
        cov3d = gs.compute_cov3d(scaling, rotation, visible)

        # ewa project
        if enable_ortho_projection:
            (conic, radius, tiles_touched) = ewa_project_torch_impl(
                position,
                cov3d,
                extrinsic_matrix.cuda(),
                uv,
                width,
                height,
                visible.squeeze(-1)
            )
        else:
            (conic, radius, tiles_touched) = gs.ewa_project(
                position,
                cov3d,
                intrinsic_matrix.cuda(),
                extrinsic_matrix.cuda(),
                uv,
                width,
                height,
                visible
            )

        # sort
        (gaussian_ids_sorted, tile_range) = gs.sort_gaussian(
            uv, depth, width, height, radius, tiles_touched
        )

        Render_Features = RenderFeatures(rgb=rgb)
        render_features = Render_Features.combine()

        ndc = torch.zeros_like(uv, requires_grad=True)
        abs_ndc = torch.zeros_like(uv, requires_grad=True)
        # alpha blending
        try:
            ndc.retain_grad()
            abs_ndc.retain_grad()
        except:
            raise ValueError("ndc does not have grad")

        bg_color = kwargs.get("bg_color", self.bg_color)
        num_idx = kwargs.get("num_idx", 10)
        rendered_features, ncontrib, gs_idx = gs.alpha_blending_enhanced(
            uv, conic, opacity, render_features,
            gaussian_ids_sorted, tile_range, bg_color, 
            width, height, 
            ndc,
            abs_ndc,
            K=num_idx
        )
        rendered_features_split = Render_Features.split(rendered_features)

        ##### render for depth
        bg_color = 1.0
        rendered_features_depth = gs.alpha_blending(
            uv, conic, opacity, depth,
            gaussian_ids_sorted, tile_range, bg_color, width, height, ndc.detach()
        )
        rendered_features_split.update({"depth": rendered_features_depth})


        render_attributes_list = kwargs.get("render_attributes_list", [])

        # process flow seperately

        if len(render_attributes_list) > 0:
            attributes_dict = {x : kwargs[x] for x in render_attributes_list}
            Render_Features_extend = RenderFeatures(**attributes_dict)

            render_features_extend = Render_Features_extend.combine()

            bg_color = 0.0
            rendered_features_extent = gs.alpha_blending(
                uv, conic, opacity.detach(), render_features_extend,
                gaussian_ids_sorted, tile_range, bg_color, width, height, ndc.detach()
            )
            rendered_features_split.update(Render_Features_extend.split(rendered_features_extent))

        return {"rendered_features_split": rendered_features_split,
                "viewspace_points": abs_ndc if self.cfg.densify_abs_grad_enable else ndc,
                "visibility_filter": radius > 0,
                "radii": radius,
                "gs_idx": gs_idx
                }

    def render_batch(self, render_dict: dict, batch: List[dict]) -> dict:
        """
        Render the batch of point clouds.

        Parameters
        ----------
        render_dict : dict
            The render dictionary.
        batch : List[dict]
            The batch data.

        Returns
        -------
        dict
            The rendered image, the viewspace points, 
            the visibility filter, the radii, the xyz, 
            the color, the rotation, the scales, and the xy.
        """
        rendered_features = {}
        viewspace_points = []
        visibilitys = []
        radii = []
        gs_idx = []

        for b_i in batch:
            b_i.update(render_dict)
            render_results = self.render_iter(**b_i)
            for feature_name in render_results["rendered_features_split"].keys():
                if feature_name not in rendered_features:
                    rendered_features[feature_name] = []
                rendered_features[feature_name].append(
                    render_results["rendered_features_split"][feature_name])

            viewspace_points.append(render_results["viewspace_points"])
            visibilitys.append(
                render_results["visibility_filter"].unsqueeze(0))
            radii.append(render_results["radii"].unsqueeze(0))
            gs_idx.append(render_results["gs_idx"].unsqueeze(0))

        for feature_name in rendered_features.keys():
            rendered_features[feature_name] = torch.stack(
                rendered_features[feature_name], dim=0)

        return {**rendered_features,
                "viewspace_points": viewspace_points,
                "visibility": torch.cat(visibilitys).any(dim=0),
                "radii": torch.cat(radii, 0).max(dim=0).values,
                "gs_idx": torch.cat(gs_idx, 0)
                }
    
    def update_sh_degree(self, step):
        if step % self.cfg.update_sh_iter == 0:
            if self.active_sh_degree < self.cfg.max_sh_degree:
                self.active_sh_degree += 1

    def load_state_dict(self, state_dict):
        self.active_sh_degree = state_dict["active_sh_degree"]

    def state_dict(self):
        return {"active_sh_degree": self.active_sh_degree}
