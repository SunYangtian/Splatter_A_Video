import os
import json
import numpy as np
from PIL import Image
from typing import Any, Dict, List
from pathlib import Path

from pointrix.camera.camera import Camera
from pointrix.dataset.base_data import BaseReFormatData, DATA_FORMAT_REGISTRY
from pointrix.utils.dataset.dataset_utils import fov2focal, focal2fov
from pointrix.logger.writer import Logger
from pointrix.dataset.base_data import SimplePointCloud


@DATA_FORMAT_REGISTRY.register()
class ImageDepthReFormat(BaseReFormatData):
    """
    The foundational classes for formating the nerf_synthetic data.

    Parameters
    ----------
    data_root: Path
        The root of the data.
    split: str
        The split of the data.
    cached_image: bool
        Whether to cache the image in memory.
    scale: float
        The scene scale of data.
    """
    def __init__(self,
                 data_root: Path,
                 split: str = 'train',
                 scale: float = 1.0,
                 cached_image: bool = True):
        super().__init__(data_root, split, cached_image)

    def load_camera(self, split: str) -> List[Camera]:
        """
        The function for loading the camera typically requires user customization.

        Parameters
        ----------
        split: The split of the data.
        """
        
        c2w = np.eye(4)
        c2w[:3,3] = np.array([0, 0, 0.0])  # set the translation  [z = 1 + 1]
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        # R is stored transposed due to 'glm' in CUDA code
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]

        image_path = self.data_root
        image = np.array(Image.open(image_path))

        fovx = np.pi / 2.0
        fovy = focal2fov(fov2focal(fovx, image.shape[1]), image.shape[0])
        FovY = fovy
        FovX = fovx

        camera = Camera(idx=0, R=R, T=T, width=image.shape[1], height=image.shape[0],
                        rgb_file_name=image_path, fovX=FovX, fovY=FovY, bg=1.0)
        
        cameras = [camera]

        return cameras

    def load_image_filenames(self, cameras: List[Camera], split) -> List[Path]:
        """
        The function for loading the image files names typically requires user customization.

        Parameters
        ----------
        split: The split of the data.
        """
        image_filenames = []
        for camera in cameras:
            image_filenames.append(camera.rgb_file_name)
        return image_filenames

    def load_metadata(self, split) -> Dict[str, Any]:
        """
        The function for loading other information that is required for the dataset typically requires user customization.

        Parameters
        ----------
        split: The split of the data.
        """
        meta_data = {}        
        return meta_data
    
    def load_pointcloud(self):
        depth_folder = os.path.join(os.path.dirname(self.data_root), "depth_npy")
        if os.path.exists(depth_folder):
            depth_file_names = sorted(os.listdir(os.path.join(self.data_root, depth_folder)))
            depth = np.load(os.path.join(depth_folder, depth_file_names[0]))
            points = self.depth2pcd(depth).reshape(-1,3)

            image_path = self.data_root
            colors = np.array(Image.open(image_path)).reshape(-1,3) / 255.

            pcd = SimplePointCloud(points, colors, normals=None)
            return pcd
        else:
            return None
    
    def depth2pcd(self, depth):
        """
        Convert the depth map to point cloud.

        Parameters
        ----------
        depth: np.ndarray
            The depth map.
        focal: float
            The focal length.
        """
        h, w = depth.shape
        fovx = np.pi / 2.0  # same with load camera !
        focal = fov2focal(fovx, w)
        i, j = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        z = depth + 0.5  ### add a shift value to avoid zero depth
        x = (j - w * 0.5) * z / focal
        y = (i - h * 0.5) * z / focal
        pcd = np.stack([x, y, z], axis=-1)
        ### convert to opengl coordinate
        pcd[...,1] *= -1
        pcd[...,2] *= -1
        ### depth [0,1] -> [0,-1]
        return pcd
