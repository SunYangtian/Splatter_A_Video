import torch
import numpy as np
import dptr.gs as gs

from typing import List
from dataclasses import dataclass
from pointrix.utils.base import BaseObject
from pointrix.utils.renderer.renderer_utils import RenderFeatures
from pointrix.utils.registry import Registry
RENDERER_REGISTRY = Registry("RENDERER", modules=["pointrix.renderer"])

@RENDERER_REGISTRY.register()
class DPTRRender(BaseObject):
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
    
    cfg: Config

    def setup(self, white_bg, device, **kwargs):
        self.active_sh_degree = 0
        self.device = device
        super().setup(white_bg, device, **kwargs)
        self.bg_color = 1. if white_bg else 0.

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
    
        direction = (position.cuda() -
                     camera_center.repeat(position.shape[0], 1).cuda())
        direction = direction / direction.norm(dim=1, keepdim=True)
        rgb = gs.compute_sh(shs, 3, direction)

        (uv, depth) = gs.project_point(
            position,
            intrinsic_matrix.cuda(),
            extrinsic_matrix.cuda(),
            width, height,
            nearest=0.01
            )
        
        # img = torch.zeros((height, width), dtype=torch.float32).to(self.device)
        # uv_ = uv.clone()
        # uv_[:,0] = torch.clamp(uv_[:,0], 0, width-1)
        # uv_[:,1] = torch.clamp(uv_[:,1], 0, height-1)
        # img[uv_[:, 1].long(), uv_[:, 0].long()] = 1
        # imageio.imwrite("debug2.png", (img.cpu().numpy()*255).astype(np.uint8))


        visible = depth != 0

        # compute cov3d
        cov3d = gs.compute_cov3d(scaling, rotation, visible)

        # ewa project
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

        Render_Features = RenderFeatures(rgb=rgb, depth=depth)
        if "pixel_flow" in kwargs:
            setattr(Render_Features, "pixel_flow", kwargs["pixel_flow"])
        render_features = Render_Features.combine()

        ndc = torch.zeros_like(uv, requires_grad=True)
        # alpha blending
        try:
            ndc.retain_grad()
        except:
            raise ValueError("ndc does not have grad")

        rendered_features = gs.alpha_blending(
            uv, conic, opacity, render_features,
            gaussian_ids_sorted, tile_range, self.bg_color, width, height, ndc
        )
        rendered_features_split = Render_Features.split(rendered_features)

        return {"rendered_features_split": rendered_features_split,
                "viewspace_points": ndc,
                "visibility_filter": radius > 0,
                "radii": radius
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

        for feature_name in rendered_features.keys():
            rendered_features[feature_name] = torch.stack(
                rendered_features[feature_name], dim=0)

        return {**rendered_features,
                "viewspace_points": viewspace_points,
                "visibility": torch.cat(visibilitys).any(dim=0),
                "radii": torch.cat(radii, 0).max(dim=0).values
                }
    
    def update_sh_degree(self, step):
        if step % self.cfg.update_sh_iter == 0:
            if self.active_sh_degree < self.cfg.max_sh_degree:
                self.active_sh_degree += 1

    def load_state_dict(self, state_dict):
        self.active_sh_degree = state_dict["active_sh_degree"]

    def state_dict(self):
        return {"active_sh_degree": self.active_sh_degree}
