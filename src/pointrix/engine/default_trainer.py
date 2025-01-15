import os
import random
from tqdm import tqdm
from typing import Any, Optional, Union, List
from dataclasses import dataclass, field

import torch
from torch import nn
from pathlib import Path
from pointrix.renderer import parse_renderer
from pointrix.dataset import parse_data_pipeline
from pointrix.utils.config import parse_structured
from pointrix.optimizer import parse_optimizer, parse_scheduler
from pointrix.model import parse_model
from pointrix.logger import parse_writer
from pointrix.hook import parse_hooks
from pointrix.exporter.novel_view import test_view_render, novel_view_render

from torch.utils.tensorboard import SummaryWriter


class DefaultTrainer:
    """
    The default trainer class for training and testing the model.

    Parameters
    ----------
    cfg : dict
        The configuration dictionary.
    exp_dir : str
        The experiment directory.
    device : str, optional
        The device to use, by default "cuda".
    """
    @dataclass
    class Config:
        # Modules
        model: dict = field(default_factory=dict)
        optimizer: dict = field(default_factory=dict)
        renderer: dict = field(default_factory=dict)
        scheduler: Optional[dict] = field(default_factory=dict)
        writer: dict = field(default_factory=dict)
        hooks: dict = field(default_factory=dict)
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

    def __init__(self, cfg: Config, exp_dir: Path, device: str = "cuda", tb_logger=False) -> None:
        super().__init__()
        self.exp_dir = exp_dir
        self.device = device

        self.start_steps = 1
        self.global_step = 0

        # build config
        self.cfg = parse_structured(self.Config, cfg)
        # build hooks
        self.hooks = parse_hooks(self.cfg.hooks)
        self.call_hook("before_run")
        # build datapipeline
        self.datapipeline = parse_data_pipeline(self.cfg.dataset)

        # build render and point cloud model
        self.white_bg = self.datapipeline.white_bg
        self.renderer = parse_renderer(
            self.cfg.renderer, white_bg=self.white_bg, device=device)

        self.model = parse_model(
            self.cfg.model, self.datapipeline, device=device)

        # build optimizer and scheduler
        cameras_extent = self.datapipeline.training_dataset.radius
        self.schedulers = parse_scheduler(self.cfg.scheduler,
                                          cameras_extent if self.cfg.spatial_lr_scale else 1.
                                          )
        self.optimizer = parse_optimizer(self.cfg.optimizer,
                                         self.model,
                                         cameras_extent=cameras_extent)

        # build logger and hooks
        if tb_logger:
            self.logger = parse_writer(self.cfg.writer, exp_dir)
        else:
            self.logger = None

    def train_step(self, batch: List[dict]) -> None:
        """
        The training step for the model.

        Parameters
        ----------
        batch : dict
            The batch data.
        """
        render_dict = self.model(batch)
        render_results = self.renderer.render_batch(render_dict, batch)
        # #### debug
        # from PIL import Image
        # import numpy as np
        # Image.fromarray((render_results['rgb'][0].permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)).save("./debug.png")
        self.loss_dict = self.model.get_loss_dict(render_results, batch)
        self.loss_dict['loss'].backward()
        self.optimizer_dict = self.model.get_optimizer_dict(self.loss_dict,
                                                            render_results,
                                                            self.white_bg)

    @torch.no_grad()
    def validation(self):
        self.val_dataset_size = len(self.datapipeline.validation_dataset)
        for i in range(0, self.val_dataset_size):
            self.call_hook("before_val_iter")
            batch = self.datapipeline.next_val(i)
            render_dict = self.model(batch)
            render_results = self.renderer.render_batch(render_dict, batch)
            self.metric_dict = self.model.get_metric_dict(render_results, batch)
            self.call_hook("after_val_iter")

    def test(self, model_path=None) -> None:
        """
        The testing method for the model.
        """
        # self.model.load_ply(model_path)
        self.load_model(model_path)
        self.model.to(self.device)
        self.renderer.active_sh_degree = 3
        test_view_render(self.model, self.renderer,
                         self.datapipeline, output_path=self.cfg.output_path)
        novel_view_render(self.model, self.renderer,
                          self.datapipeline, output_path=self.cfg.output_path, 
                          novel_view_list=["Spiral"])

    def train_loop(self) -> None:
        """
        The training loop for the model.
        """
        loop_range = range(self.start_steps, self.cfg.max_steps+1)
        self.global_step = self.start_steps
        self.call_hook("before_train")
        for iteration in loop_range:
            self.call_hook("before_train_iter")
            batch = self.datapipeline.next_train(self.global_step)
            self.renderer.update_sh_degree(iteration)
            self.schedulers.step(self.global_step, self.optimizer)
            self.train_step(batch)
            self.optimizer.update_model(**self.optimizer_dict)
            self.call_hook("after_train_iter")
            self.global_step += 1
            if (iteration+1) % self.cfg.val_interval == 0 or iteration+1 == self.cfg.max_steps:
                self.call_hook("before_val")
                self.validation()
                self.call_hook("after_val")
        self.call_hook("after_train")

    def call_hook(self, fn_name: str, **kwargs) -> None:
        """
        Call the hook method.

        Parameters
        ----------
        fn_name : str
            The hook method name.
        kwargs : dict
            The keyword arguments.
        """
        for hook in self.hooks:
            # support adding additional custom hook methods
            if hasattr(hook, fn_name):
                try:
                    getattr(hook, fn_name)(self, **kwargs)
                except TypeError as e:
                    raise TypeError(f'{e} in {hook}') from None

    def load_model(self, path: Path = None) -> None:
        if path is None:
            path = os.path.join(self.exp_dir,
                                "chkpnt" + str(self.global_step) + ".pth")
        data_list = torch.load(path)
        for k, v in data_list.items():
            print(f"Loaded {k} from checkpoint")
            # get arrtibute from model
            arrt = getattr(self, k)
            if hasattr(arrt, 'load_state_dict'):
                arrt.load_state_dict(v)
            else:
                setattr(self, k, v)

    def save_model(self, path: Path = None) -> None:
        if path is None:
            path = os.path.join(self.exp_dir,
                                "chkpnt" + str(self.global_step) + ".pth")
        data_list = {
            "global_step": self.global_step,
            "optimizer": self.optimizer.state_dict(),
            "model": self.model.get_state_dict(),
            "renderer": self.renderer.state_dict()
        }
        torch.save(data_list, path)
