import torch
from dataclasses import dataclass, field
from typing import Mapping, Optional, Union, Any
from omegaconf import DictConfig
from pytorch_msssim import ms_ssim

from pointrix.utils.base import BaseModule
from pointrix.utils.config import parse_structured
from pointrix.point_cloud import parse_point_cloud
from .loss import l1_loss, ssim, psnr
# from .lpips_pytorch import lpips
from pointrix.utils.registry import Registry

MODEL_REGISTRY = Registry("MODEL", modules=["pointrix.model"])
MODEL_REGISTRY.__doc__ = ""


@MODEL_REGISTRY.register()
class BaseModel(BaseModule):
    """
    Base class for all models.

    Parameters
    ----------
    cfg : Optional[Union[dict, DictConfig]]
        The configuration dictionary.
    datapipeline : BaseDataPipeline
        The data pipeline which is used to initialize the point cloud.
    device : str, optional
        The device to use, by default "cuda".
    """
    @dataclass
    class Config:
        point_cloud: dict = field(default_factory=dict)
        lambda_dssim: float = 0.2

    cfg: Config

    def setup(self, datapipeline, device="cuda"):
        self.point_cloud = parse_point_cloud(self.cfg.point_cloud,
                                             datapipeline).to(device)
        self.point_cloud.set_prefix_name("point_cloud")
        self.device = device

    def forward(self, batch=None) -> dict:
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
        render_dict = {
            "position": self.point_cloud.position,
            "opacity": self.point_cloud.get_opacity,
            "scaling": self.point_cloud.get_scaling,
            "rotation": self.point_cloud.get_rotation,
            "shs": self.point_cloud.get_shs,
        }
        return render_dict

    def get_loss_dict(self, render_results, batch) -> dict:
        """
        Get the loss dictionary.

        Parameters
        ----------
        render_results : dict
            The render results which is the output of the renderer.
        batch : dict
            The batch of data which contains the ground truth images.
        
        Returns
        -------
        dict
            The loss dictionary which contain loss for backpropagation.
        """
        gt_images = torch.stack(
            [batch[i]["image"].to(self.device) for i in range(len(batch))],
            dim=0
        )
        L1_loss = l1_loss(render_results['rgb'], gt_images)
        ssim_loss = 1.0 - ssim(render_results['rgb'], gt_images)
        loss = (
            (1.0 - self.cfg.lambda_dssim) * L1_loss
        ) + (
            self.cfg.lambda_dssim * ssim_loss
        )
        loss_dict = {"loss": loss,
                     "L1_loss": L1_loss,
                     "ssim_loss": ssim_loss}
        return loss_dict

    def get_optimizer_dict(self, loss_dict, render_results, white_bg) -> dict:
        """
        Get the optimizer dictionary which will be 
        the input of the optimizer update model

        Parameters
        ----------
        loss_dict : dict
            The loss dictionary.
        render_results : dict
            The render results which is the output of the renderer.
        white_bg : bool
            The white background flag.
        """
        optimizer_dict = {"loss": loss_dict["loss"],
                          "viewspace_points": render_results['viewspace_points'],
                          "visibility": render_results['visibility'],
                          "radii": render_results['radii'],
                          "white_bg": white_bg}
        return optimizer_dict

    @torch.no_grad()
    def get_metric_dict(self, render_results, batch) -> dict:
        """
        Get the metric dictionary.

        Parameters
        ----------
        render_results : dict
            The render results which is the output of the renderer.
        batch : dict
            The batch of data which contains the ground truth images.
        
        Returns
        -------
        dict
            The metric dictionary which contains the metrics for evaluation.
        """
        gt_images = torch.clamp(torch.stack(
            [batch[i]["image"].to(self.device) for i in range(len(batch))],
            dim=0
        ), 0.0, 1.0)
        rgb = torch.clamp(render_results['rgb'], 0.0, 1.0)
        L1_loss = l1_loss(rgb, gt_images).mean().double()
        psnr_test = psnr(rgb, gt_images).mean().double()
        ssims_test = ms_ssim(
            rgb, gt_images, data_range=1, size_average=True
        ).mean().item()
        # lpips_test = lpips(
        #     render_results['images'], 
        #     gt_images,
        #     net_type='vgg'
        # ).mean().item()
        metric_dict = {"L1_loss": L1_loss,
                       "psnr": psnr_test,
                       "ssims": ssims_test,
                    #    "lpips": lpips_test,
                       "gt_images": gt_images,
                       "images": rgb,
                       "rgb_file_name": batch[0]["camera"].rgb_file_name}
        
        if 'depth' in render_results:
            depth = render_results['depth']
            metric_dict['depth'] = depth

        return metric_dict
    
    def load_ply(self, path):
        """
        Load the ply model for point cloud.

        Parameters
        ----------
        path : str
            The path of the ply file.
        """
        self.point_cloud.load_ply(path)
    
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        num_pts = state_dict.pop('num_pts')

        if num_pts != len(self.point_cloud):
            self.point_cloud.re_init(num_pts)

        return super().load_state_dict(state_dict, strict, assign)

    def get_state_dict(self):
        additional_info = {'num_pts': len(self.point_cloud)}
        return {**super().state_dict(), **additional_info}



        
