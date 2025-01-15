import os
import random
import numpy as np
from tqdm import tqdm
from typing import Any, Optional, Union
from dataclasses import dataclass, field

import torch
import imageio
from torch import nn
from operator import methodcaller

from pointrix.utils.losses import l1_loss
from pointrix.utils.system import mkdir_p
from pointrix.utils.gaussian_points.gaussian_utils import psnr
from pointrix.utils.visuaize import visualize_depth, visualize_rgb


@torch.no_grad()
def test_view_render(model, renderer, datapipeline, output_path, device='cuda'):
    """
    Render the test view and save the images to the output path.

    Parameters
    ----------
    model : BaseModel
        The point cloud model.
    renderer : Renderer
        The renderer object.
    datapipeline : DataPipeline
        The data pipeline object.
    output_path : str
        The output path to save the images.
    """
    l1_test = 0.0
    psnr_test = 0.0
    val_dataset = datapipeline.validation_dataset
    val_dataset_size = len(val_dataset)
    progress_bar = tqdm(
        range(0, val_dataset_size),
        desc="Validation progress",
        leave=False,
    )

    mkdir_p(os.path.join(output_path, 'test_view'))

    for i in range(0, val_dataset_size):
        b_i = val_dataset[i]
        atributes_dict = model(b_i)
        atributes_dict.update(b_i)
        image_name = os.path.basename(b_i['camera'].rgb_file_name)
        render_results = renderer.render_iter(**atributes_dict)
        gt_image = torch.clamp(b_i['image'].to("cuda").float(), 0.0, 1.0)
        image = torch.clamp(
            render_results['rendered_features_split']['rgb'], 0.0, 1.0)

        for feat_name, feat in render_results['rendered_features_split'].items():
            visual_feat = eval(f"visualize_{feat_name}")(feat.squeeze())
            if not os.path.exists(os.path.join(output_path, f'test_view_{feat_name}')):
                os.makedirs(os.path.join(
                    output_path, f'test_view_{feat_name}'))
            imageio.imwrite(os.path.join(
                output_path, f'test_view_{feat_name}', image_name), visual_feat)

        l1_test += l1_loss(image, gt_image).mean().double()
        psnr_test += psnr(image, gt_image).mean().double()
        progress_bar.update(1)
    progress_bar.close()
    l1_test /= val_dataset_size
    psnr_test /= val_dataset_size
    print(f"Test results: L1 {l1_test:.5f} PSNR {psnr_test:.5f}")


def novel_view_render(model, renderer, datapipeline, output_path, novel_view_list=["Dolly", "Zoom", "Spiral"], device='cuda'):
    """
    Render the novel view and save the images to the output path.

    Parameters
    ----------
    model : BaseModel
        The point cloud model.
    renderer : Renderer
        The renderer object.
    datapipeline : DataPipeline
        The data pipeline object.
    output_path : str
        The output path to save the images.
    novel_view_list : list, optional
        The list of novel views to render, by default ["Dolly", "Zoom", "Spiral"]
    """
    cameras = datapipeline.training_cameras
    print("Rendering Novel view ...............")
    for novel_view in novel_view_list:
        novel_view_camera_list = cameras.generate_camera_path(50, novel_view)

        images_dict = {}
        for i, camera in enumerate(novel_view_camera_list):

            atributes_dict = model(camera)
            render_dict = {
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
            render_dict.update(atributes_dict)
            render_results = renderer.render_iter(**render_dict)

            for feat_name, feat in render_results['rendered_features_split'].items():
                visual_feat = eval(f"visualize_{feat_name}")(feat.squeeze())
                if not os.path.exists(os.path.join(output_path, f'{novel_view}_{feat_name}')):
                    os.makedirs(os.path.join(
                        output_path, f'{novel_view}_{feat_name}'))
                imageio.imwrite(os.path.join(
                    output_path, f'{novel_view}_{feat_name}', "{:0>3}.png".format(i)), visual_feat)
                images_dict.setdefault(feat_name, []).append(visual_feat)
        for feat_name, images in images_dict.items():
            imageio.mimwrite(os.path.join(
                output_path, f'{novel_view}_{feat_name}.mp4'), images, fps=10)
        
    

