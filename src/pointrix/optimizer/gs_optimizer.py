import torch
from torch import Tensor
from torch.optim import Optimizer
from dataclasses import dataclass
from typing import Tuple, List

from pointrix.utils.config import C

from .optimizer import BaseOptimizer
from pointrix.model.base_model import BaseModel

from pointrix.utils.gaussian_points.gaussian_utils import (
    inverse_sigmoid,
    build_rotation,
)
from .optimizer import OPTIMIZER_REGISTRY


@OPTIMIZER_REGISTRY.register()
class GaussianSplattingOptimizer(BaseOptimizer):
    @dataclass
    class Config:
        # Densification
        control_module: str = "point_cloud"
        percent_dense: float = 0.01
        split_num: int = 2
        densify_stop_iter: int = 15000
        densify_start_iter: int = 500
        prune_interval: int = 100
        duplicate_interval: int = 100
        opacity_reset_interval: int = 3000
        densify_grad_threshold: float = 0.0002
        min_opacity: float = 0.005
        verbose: bool = False

    cfg: Config
    
    """
    Optimizer for Gaussian splatting, which can be not only used to optimize the parameter
    of the point cloud, but also to densify, prune and split the point cloud.

    Parameters
    ----------
    optimizer : Optimizer
        The optimizer which is used to update parameters.
    point_cloud : GSPointCloud
        The point cloud which will be optimized.
    cfg : dict
        The configuration.
    cameras_extent : float
        The radius of the cameras.
    """

    def setup(self, optimizer: Optimizer, model: BaseModel, cameras_extent: float) -> None:
        self.optimizer = optimizer
        self.point_cloud = getattr(model, self.cfg.control_module)
        self.device = self.point_cloud.device
        if len(optimizer.param_groups) > 1:
            self.base_param_settings = {
                'params': torch.tensor([0.0], dtype=torch.float)
            }
            self.base_param_settings.update(**self.optimizer.defaults)
        else:
            self.base_param_settings = None  # type: ignore

        self.cameras_extent = cameras_extent

        # Densification setup
        num_points = len(self.point_cloud)
        self.max_radii2D = torch.zeros(num_points).to(self.device)
        self.percent_dense = self.cfg.percent_dense
        self.pos_gradient_accum = torch.zeros((num_points, 1)).to(self.device)
        self.denom = torch.zeros((num_points, 1)).to(self.device)
        self.opacity_deferred = False

        self.step = 0
        self.update_hypers()

    def update_hypers(self) -> None:
        """
        Update the hyperparameters of the optimizer.
        """
        self.split_num = int(C(self.cfg.split_num, 0, self.step))
        self.prune_interval = int(C(self.cfg.prune_interval, 0, self.step))
        self.duplicate_interval = int(
            C(self.cfg.duplicate_interval, 0, self.step))
        self.opacity_reset_interval = int(
            C(self.cfg.opacity_reset_interval, 0, self.step))
        self.densify_grad_threshold = C(
            self.cfg.densify_grad_threshold, 0, self.step)
        self.min_opacity = C(self.cfg.min_opacity, 0, self.step)
        self.step += 1

    @torch.no_grad()
    def update_structure(self, visibility: Tensor, viewspace_grad: Tensor,
                         radii: Tensor, white_bg: bool = False) -> None:
        """
        Update the structure of the point cloud.

        Parameters
        ----------
        visibility : torch.Tensor
            The visibility of the points.
        viewspace_grad : torch.Tensor
            The gradient in the view space.
        radii : torch.Tensor
            The radii of the point cloud
        white_bg : bool
            Whether the background is white.
        """
        if self.step < self.cfg.densify_stop_iter:
            # Keep track of max radii in image-space for pruning
            self.max_radii2D[visibility] = torch.max(
                self.max_radii2D[visibility],
                radii[visibility]
            )
            self.pos_gradient_accum[visibility] += torch.norm(
                viewspace_grad[visibility, :2],
                dim=-1,
                keepdim=True
            )
            self.denom[visibility] += 1

            if self.step > self.cfg.densify_start_iter:
                self.densification(self.step)

            # Reset opacity with a delay incase of validation right after reset opacity
            if self.opacity_deferred:
                self.opacity_deferred = False
                self.reset_opacity()

            if self.step % self.opacity_reset_interval == 0 or (white_bg and self.step == self.cfg.densify_start_iter):
                self.opacity_deferred = True

    def update_model(self, viewspace_points: Tensor,
                     visibility: Tensor, radii: Tensor, white_bg: bool, **kwargs) -> None:
        """
        Update the model parameter with the loss, 
        and update the structure with viewspace points, 
        visibility, radii and white background.

        you need call backward first, 
        then call this function to update the model.

        Parameters
        ----------
        loss : torch.Tensor
            The loss tensor.
        viewspace_points : torch.Tensor
            The view space points.
        visibility : torch.Tensor
            The visibility of the points.
        radii : torch.Tensor
            The radii of the point cloud.
        white_bg : bool
            Whether the background is white.
        """
        with torch.no_grad():
            viewspace_grad = self.accumulate_viewspace_grad(viewspace_points)
            self.update_structure(visibility, viewspace_grad, radii, white_bg)
            self.update_hypers()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

    def densification(self, step: int) -> None:
        """
        Densify the point cloud.

        Parameters
        ----------
        step : int
            The current step.
        """
        ### set a threshold.
        if step % self.duplicate_interval == 0:
            grads = self.pos_gradient_accum / self.denom
            grads[grads.isnan()] = 0.0
            self.densify_clone(grads)
            self.densify_split(grads)
        if step % self.prune_interval == 0:
            self.prune(step)
        torch.cuda.empty_cache()

    def reset_opacity(self) -> None:
        """
        Reset the opacity of the point cloud.        
        """
        opc = self.point_cloud.get_opacity
        opacities_new = inverse_sigmoid(
            torch.min(opc, torch.ones_like(opc)*0.01)
        )
        self.point_cloud.replace(
            {"opacity": opacities_new},
            self.optimizer
        )

    def generate_clone_mask(self, grads: Tensor) -> Tensor:
        """
        Generate the mask for cloning.

        Parameters
        ----------
        grads : torch.Tensor
            The gradients.

        Returns
        -------
        torch.Tensor
            The mask for cloning.
        """
        scaling = self.point_cloud.get_scaling
        cameras_extent = self.cameras_extent
        max_grad = self.densify_grad_threshold

        mask = torch.where(torch.norm(
            grads, dim=-1) >= max_grad, True, False)
        mask = torch.logical_and(
            mask,
            torch.max(
                scaling,
                dim=1
            ).values <= self.percent_dense*cameras_extent
        )
        return mask

    def generate_split_mask(self, grads: Tensor) -> Tensor:
        """
        Generate the mask for splitting.

        Parameters
        ----------
        grads : torch.Tensor
            The gradients.
        """
        scaling = self.point_cloud.get_scaling
        cameras_extent = self.cameras_extent
        max_grad = self.densify_grad_threshold

        num_points = len(self.point_cloud)
        padded_grad = torch.zeros((num_points), device=self.device)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        mask = torch.where(padded_grad >= max_grad, True, False)

        mask = torch.logical_and(
            mask,
            torch.max(
                scaling,
                dim=1
            ).values > self.percent_dense*cameras_extent
        )
        return mask

    def new_pos_scale(self, mask: Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate new position and scaling for splitting.

        Parameters
        ----------
        mask : torch.Tensor
            The mask for splitting.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The new position and scaling.
        """
        scaling = self.point_cloud.get_scaling
        position = self.point_cloud.position
        rotation = self.point_cloud.rotation
        split_num = self.split_num

        stds = scaling[mask].repeat(split_num, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(
            rotation[mask]
        ).repeat(split_num, 1, 1)
        new_pos = (
            torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
        ) + (
            position[mask].repeat(split_num, 1)
        )
        new_scaling = self.point_cloud.scaling_inverse_activation(
            scaling[mask].repeat(split_num, 1) / (0.8*split_num)
        )
        return new_pos, new_scaling

    def densify_clone(self, grads: Tensor) -> None:
        """
        Densify the point cloud by cloning.

        Parameters
        ----------
        grads : torch.Tensor
            The gradients.
        """
        num1 = len(self.point_cloud)
        mask = self.generate_clone_mask(grads)
        atributes = self.point_cloud.select_atributes(mask)
        self.point_cloud.extand_points(atributes, self.optimizer)
        num2 = len(self.point_cloud)
        self.reset_densification_state()
        if self.cfg.verbose:
            print("=== increase [clone] points from {} to {} ===".format(num1, num2))

    def densify_split(self, grads: Tensor) -> None:
        """
        Densify the point cloud by splitting.

        Parameters
        ----------
        grads : torch.Tensor
            The gradients.
        """
        num1 = len(self.point_cloud)
        mask = self.generate_split_mask(grads)
        new_pos, new_scaling = self.new_pos_scale(mask)
        atributes = self.point_cloud.select_atributes(mask)

        # Replace position and scaling from selected atributes
        atributes["position"] = new_pos
        atributes["scaling"] = new_scaling

        # Update rest of atributes
        for key, value in atributes.items():
            # Skip position and scaling, since they are already updated
            if key == "position" or key == "scaling":
                continue
            # Create a tuple of n_dim ones
            sizes = [1 for _ in range(len(value.shape))]
            sizes[0] = self.split_num
            sizes = tuple(sizes)

            # Repeat selected atributes in the fist dimension
            atributes[key] = value.repeat(*sizes)

        self.point_cloud.extand_points(atributes, self.optimizer)
        self.reset_densification_state()

        # TODO: need to remove unused operation
        prune_filter = torch.cat((mask, torch.zeros(self.split_num * mask.sum(),
                                                    device=self.device, dtype=bool)))
        valid_points_mask = ~prune_filter
        self.point_cloud.remove_points(valid_points_mask, self.optimizer)
        self.prune_postprocess(valid_points_mask)
        num2 = len(self.point_cloud)
        if self.cfg.verbose:
            print("=== increase [split] points from {} to {} ===".format(num1, num2))

    def prune(self, step: int) -> None:
        """
        Prune the point cloud.

        Parameters
        ----------
        step : int
            The current step.
        """
        # TODO: fix me
        num1 = len(self.point_cloud)
        size_threshold = 20 if step > self.opacity_reset_interval else None
        cameras_extent = self.cameras_extent

        prune_filter = (
            self.point_cloud.get_opacity < self.min_opacity
        ).squeeze()
        if size_threshold:
            big_points_vs = self.max_radii2D > size_threshold
            big_points_ws = self.point_cloud.get_scaling.max(
                dim=1).values > 0.1 * cameras_extent
            prune_filter = torch.logical_or(prune_filter, big_points_vs)
            prune_filter = torch.logical_or(prune_filter, big_points_ws)

        valid_points_mask = ~prune_filter
        self.point_cloud.remove_points(valid_points_mask, self.optimizer)
        self.prune_postprocess(valid_points_mask)
        num2 = len(self.point_cloud)
        if self.cfg.verbose:
            print("=== decrease [prune] points from {} to {} ===".format(num1, num2))

    def prune_postprocess(self, valid_points_mask):
        """
        Postprocess after pruning.

        Parameters
        ----------
        valid_points_mask : torch.Tensor
            The mask for valid points.
        """
        self.pos_gradient_accum = self.pos_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def reset_densification_state(self) -> None:
        """
        Reset the densification state.
        """
        num_points = len(self.point_cloud)
        self.pos_gradient_accum = torch.zeros(
            (num_points, 1), device=self.device)
        self.denom = torch.zeros((num_points, 1), device=self.device)
        self.max_radii2D = torch.zeros((num_points), device=self.device)

    @torch.no_grad()
    def accumulate_viewspace_grad(self, viewspace_points: Tensor) -> Tensor:
        """
        Accumulate viewspace gradients for batch.

        Parameters
        ----------
        viewspace_points : torch.Tensor
            The view space points.

        Returns
        -------
        torch.Tensor
            The viewspace gradients.
        """
        # Accumulate viewspace gradients for batch
        viewspace_grad = torch.zeros_like(
            viewspace_points[0]
        )
        for vp in viewspace_points:
            viewspace_grad += vp.grad

        return viewspace_grad
