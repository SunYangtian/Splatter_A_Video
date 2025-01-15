#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from dataclasses import dataclass

from pointrix.utils.base import BaseObject
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from .dptr import RENDERER_REGISTRY

@RENDERER_REGISTRY.register()
class GaussianSplattingRender(BaseObject):
    @dataclass
    class Config:
        update_sh_iter: int = 1000
        max_sh_degree: int = 3
    
    cfg: Config
    
    """
    A class for rendering point clouds using Gaussian splatting.

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
    def setup(self, white_bg, device, **kwargs):
        self.active_sh_degree = 0
        self.device = device
        bg_color = [1, 1, 1] if white_bg else [0, 0, 0]
        self.bg_color = torch.tensor(
            bg_color, dtype=torch.float32, device=self.device)

    def render_iter(self,
                    FovX,
                    FovY,
                    height,
                    width,
                    world_view_transform,
                    full_proj_transform,
                    camera_center,
                    position,
                    opacity,
                    scaling,
                    rotation,
                    shs,
                    scaling_modifier=1.0,
                    render_xyz=False,
                    **kwargs,) -> dict:
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
        screenspace_points = torch.zeros_like(
            position,
            dtype=position.dtype,
            requires_grad=True,
            device="cuda"
        ) + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(FovX * 0.5)
        tanfovy = math.tan(FovY * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(height),
            image_width=int(width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=world_view_transform.cuda(),
            projmatrix=full_proj_transform.cuda(),
            sh_degree=self.active_sh_degree,
            campos=camera_center.cuda(),
            prefiltered=False,
            # computer_xyz=render_xyz,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = position
        means2D = screenspace_points
        cov3D_precomp = None

        # trbfscale = pc._time_scale_params
        # trbfdistanceoffset = pc.timestamp_final
        # trbfdistance =  trbfdistanceoffset / torch.exp(trbfscale)
        # trbfoutput = torch.exp(-1*trbfdistance.pow(2))

        # opacity = opacity * trbfoutput

        colors_precomp = None

        # keep up with dptr
        rendered_features_split = {}

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        (
            # num_rendered,
            rendered_image,
            # opacity,
            # depth,
            # render_xyz,
            radii,
        ) = rasterizer(
            means3D=means3D.contiguous(),
            means2D=means2D.contiguous(),
            shs=shs.contiguous(),
            colors_precomp=colors_precomp,
            opacities=opacity.contiguous(),
            scales=scaling.contiguous(),
            rotations=rotation.contiguous(),
            cov3D_precomp=cov3D_precomp
        )
        rendered_features_split['rgb'] = rendered_image

        # import pdb; pdb.set_trace()
        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {"rendered_features_split": rendered_features_split,
                # "opacity": opacity,
                # "render_xyz": render_xyz,
                "viewspace_points": screenspace_points,
                "visibility_filter": radii > 0,
                "radii": radii,
                "xyz": means3D,
                "color": shs,
                "rot": rotation,
                "scales": scaling,
                "xy": means2D, }

    def render_batch(self, render_dict, batch):
        """
        Render a batch of views.

        Parameters
        ----------
        render_dict : dict
            The rendering dictionary.
        batch : list
            The batch of views.
        """
        renders = []
        viewspace_points = []
        visibilitys = []
        radiis = []
        for b_i in batch:
            b_i.update(render_dict)
            render_results = self.render_iter(**b_i)
            renders.append(render_results["rendered_features_split"]['rgb'])
            viewspace_points.append(render_results["viewspace_points"])
            visibilitys.append(
                render_results["visibility_filter"].unsqueeze(0))
            radiis.append(render_results["radii"].unsqueeze(0))

        radii = torch.cat(radiis, 0).max(dim=0).values
        visibility = torch.cat(visibilitys).any(dim=0)
        images = torch.stack(renders)

        render_results = {
            "rgb": images,
            "radii": radii,
            "visibility": visibility,
            "viewspace_points": viewspace_points,
        }

        return render_results

    def update_sh_degree(self, step):
        if step % self.cfg.update_sh_iter == 0:
            if self.active_sh_degree < self.cfg.max_sh_degree:
                self.active_sh_degree += 1

    def load_state_dict(self, state_dict):
        self.active_sh_degree = state_dict["active_sh_degree"]

    def state_dict(self):
        return {"active_sh_degree": self.active_sh_degree}
