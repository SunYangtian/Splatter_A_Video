import math
import torch
from torch import Tensor
from torch import nn
import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray
from typing import Union, List
from dataclasses import dataclass, field
# from pointrix.camera.camera_utils import se3_exp_map
from pointrix.utils.pose import se3_exp_map


@dataclass()
class Camera:
    """
    Camera class used in Pointrix

    Parameters
    ----------
    idx: int
        The index of the camera.
    width: int
        The width of the image.
    height: int
        The height of the image.
    R: Float[Tensor, "3 3"]
        The rotation matrix of the camera.
    T: Float[Tensor, "3 1"]
        The translation vector of the camera.
    fx: float
        The focal length of the camera in x direction.
    fy: float
        The focal length of the camera in y direction.
    cx: float
        The center of the image in x direction.
    cy: float
        The center of the image in y direction.
    fovX: float
        The field of view of the camera in x direction.
    fovY: float
        The field of view of the camera in y direction.
    bg: float
        The background color of the camera.
    rgb_file_name: str
        The path of the image.
    radius: float
        The radius of the camera.
    scene_scale: float
        The scale of the scene.

    Notes
    -----
    fx, fy, cx, cy and fovX, fovY are mutually exclusive. 
    If fx, fy, cx, cy are provided, fovX, fovY will be calculated from them. 
    If fovX, fovY are provided, fx, fy, cx, cy will be calculated from them.

    Examples
    --------
    >>> idx = 1
    >>> width = 800
    >>> height = 600
    >>> R = np.eye(3)
    >>> T = np.zeros(3)
    >>> focal_length_x = 800
    >>> focal_length_y = 800
    >>> camera = Camera(idx=idx, R=R, T=T, width=width, height=height, rgb_file_name='1_rgb.png',
                        fx=focal_length_x, fy=focal_length_y, cx=width/2, cy=height/2, bg=0.0, scene_scale=1.0)
    """
    idx: int
    width: int
    height: int
    R: Union[Float[Tensor, "3 3"], NDArray]
    T: Union[Float[Tensor, "3 1"], NDArray]
    fx: Union[float, None] = None
    fy: Union[float, None] = None
    cx: Union[float, None] = None
    cy: Union[float, None] = None
    fovX: Union[float, None] = None
    fovY: Union[float, None] = None
    bg: float = 0.0
    rgb_file_name: str = None
    rgb_file_path: str = None
    radius: float = 0.0
    fid: float = 0.0
    scene_scale: float = 1.0
    _world_view_transform: Float[Tensor, "4 4"] = field(init=False)
    _projection_matrix: Float[Tensor, "4 4"] = field(init=False)
    _intrinsics_matrix: Float[Tensor, "3 3"] = field(init=False)
    _full_proj_transform: Float[Tensor, "4 4"] = field(init=False)
    _camera_center: Float[Tensor, "3"] = field(init=False)

    def __post_init__(self):
        assert (self.fx is not None and self.fy is not None and self.cx is not None and self.cy is not None
                or self.fovX is not None and self.fovY is not None), "Either fx, fy, cx, cy or fovX, fovY must be provided"
        if self.fx is None:
            self.fx = self.width / (2 * np.tan(self.fovX * 0.5))
            self.fy = self.height / (2 * np.tan(self.fovY * 0.5))
            self.cx = self.width / 2
            self.cy = self.height / 2
        elif self.fovX is None:
            # TODO: check if this is correct
            self.fovX = 2 * math.atan(self.width / (2 * self.fx))
            self.fovY = 2 * math.atan(self.height / (2 * self.fy))

        if not isinstance(self.R, Tensor):
            self.R = torch.tensor(self.R)
        if not isinstance(self.T, Tensor):
            self.T = torch.tensor(self.T)
        self._extrinsic_matrix = self.getWorld2View(
            self.R, self.T, scale=self.scene_scale
        )
        self._world_view_transform = self._extrinsic_matrix.transpose(0, 1)
        self._projection_matrix = self.getProjectionMatrix(
            self.fovX, self.fovY).transpose(0, 1)
        self._full_proj_transform = (self.world_view_transform.unsqueeze(
            0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self._intrinsic_matrix = self._get_intrinsics(
            self.fx, self.fy, self.cx, self.cy)
        self._camera_center = self.world_view_transform.inverse()[3, :3]

    def load2device(self, device: str):
        """
        Load all the parameters of the camera to the device.

        Parameters
        ----------
        device: str
            The device to load the parameters to.

        Notes
        -----
            This function will load part of the parameters of the camera to the device.
        """
        self._world_view_transform = self._world_view_transform.to(device)
        self._projection_matrix = self._projection_matrix.to(device)
        self._full_proj_transform = self._full_proj_transform.to(device)
        self._intrinsic_matrix = self._intrinsic_matrix.to(device)
        self._camera_center = self._camera_center.to(device)

    @staticmethod
    def getWorld2View(
        R: Float[Tensor, "3 3"],
        t: Float[Tensor, "3 1"],
        scale: float = 1.0,
        translate: Float[Tensor, "3"] = torch.tensor([0., 0., 0.])
    ) -> Float[Tensor, "4 4"]:
        """
        Get the world to view transform.

        Parameters
        ----------
        R: Float[Tensor, "3 3"]
            The rotation matrix of the camera.
        t: Float[Tensor, "3 1"]
            The translation vector of the camera.
        scale: float
            The scale of the scene.
        translate: Float[Tensor, "3"]
            The translation vector of the scene.

        Returns
        -------
        Rt: Float[Tensor, "4 4"]
            The world to view transform.

        Notes
        -----
        only used in the camera class

        """
        Rt = torch.zeros((4, 4))
        Rt[:3, :3] = R.transpose(0, 1)
        Rt[:3, 3] = t
        Rt[3, 3] = 1.0

        C2W = torch.linalg.inv(Rt)
        cam_center = C2W[:3, 3]
        cam_center = (cam_center + translate) * scale
        C2W[:3, 3] = cam_center
        Rt = torch.linalg.inv(C2W)
        return Rt.float()

    @staticmethod
    def _get_intrinsics(fx: float, fy: float, cx: float, cy: float) -> Float[Tensor, "3 3"]:
        """
        Get the intrinsics matrix.

        Parameters
        ----------
        fx: float
            The focal length of the camera in x direction.
        fy: float
            The focal length of the camera in y direction.
        cx: float
            The center of the image in x direction.
        cy: float
            The center of the image in y direction.

        Returns
        -------
        intrinsics: Float[Tensor, "3 3"]
            The intrinsics matrix.
        Notes
        -----
        only used in the camera class
        """
        return torch.tensor([fx, fy, cx, cy], dtype=torch.float32)

    @staticmethod
    def getProjectionMatrix(fovX: float, fovY: float, znear: float = 0.01, zfar: float = 100) -> Float[Tensor, "4 4"]:
        """
        Get the projection matrix.

        Parameters
        ----------
        fovX: float
            The field of view of the camera in x direction.
        fovY: float
            The field of view of the camera in y direction.
        znear: float
            The near plane of the camera.
        zfar: float
            The far plane of the camera.

        Returns
        -------
        P: Float[Tensor, "4 4"]
            The projection matrix.
        Notes
        -----
        only used in the camera class

        """
        tanHalfFovY = np.tan((fovY / 2))
        tanHalfFovX = np.tan((fovX / 2))

        top = tanHalfFovY * znear
        bottom = -top
        right = tanHalfFovX * znear
        left = -right

        P = torch.zeros(4, 4)

        z_sign = 1.0

        P[0, 0] = 2.0 * znear / (right - left)
        P[1, 1] = 2.0 * znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)

        return P

    @property
    def world_view_transform(self) -> Float[Tensor, "4 4"]:
        """
        Get the world to view transform from the camera.

        Returns
        -------
        _world_view_transform: Float[Tensor, "4 4"]
            The world to view transform.

        Notes
        -----
        property of the camera class

        """
        return self._world_view_transform

    @property
    def projection_matrix(self) -> Float[Tensor, "4 4"]:
        """
        Get the projection matrix from the camera.

        Returns
        -------
        _projection_matrix: Float[Tensor, "4 4"]

        Notes
        -----
        property of the camera class

        """
        return self._projection_matrix

    @property
    def full_proj_transform(self) -> Float[Tensor, "4 4"]:
        """
        Get the full projection matrix from the camera.

        Returns
        -------
        _full_proj_transform: Float[Tensor, "4 4"]

        Notes
        -----
        property of the camera class

        """
        return self._full_proj_transform

    @property
    def intrinsic_matrix(self) -> Float[Tensor, "3 3"]:
        """
        Get the intrinsics matrix from the camera.

        Returns
        -------
        _intrinsic_matrix: Float[Tensor, "3 3"]

        Notes
        -----
        property of the camera class

        """
        return self._intrinsic_matrix
    
    @property
    def extrinsic_matrix(self) -> Float[Tensor, "4 4"]:
        """
        Get the extrinsic matrix from the camera.

        Returns
        -------
        _extrinsic_matrix: Float[Tensor, "4 4"]

        Notes
        -----
        property of the camera class

        """
        return self._extrinsic_matrix

    @property
    def camera_center(self) -> Float[Tensor, "1 3"]:
        """
        Get the camera center from the camera.

        Returns
        -------
        _camera_center: Float[Tensor, "1 3"]

        Notes
        -----
        property of the camera class

        """
        return self._camera_center

    @property
    def image_height(self) -> int:
        """
        Get the image height from the camera.

        Returns
        -------
        height: int
            The image height.

        Notes
        -----
        property of the camera class

        """
        return self.height

    @property
    def image_width(self) -> int:
        """
        Get the image width from the camera.

        Returns
        -------
        width: int
            The image width.

        Notes
        -----
        property of the camera class

        """
        return self.width


@dataclass()
class TrainableCamera(Camera):
    """
    Trainable Camera class used in Pointrix

    Parameters
    ----------
    idx: int
        The index of the camera.
    width: int
        The width of the image.
    height: int
        The height of the image.
    R: Float[Tensor, "3 3"]
        The rotation matrix of the camera.
    T: Float[Tensor, "3 1"]
        The translation vector of the camera.
    fx: float
        The focal length of the camera in x direction.
    fy: float
        The focal length of the camera in y direction.
    cx: float
        The center of the image in x direction.
    cy: float
        The center of the image in y direction.
    fovX: float
        The field of view of the camera in x direction.
    fovY: float
        The field of view of the camera in y direction.
    bg: float
        The background color of the camera.
    rgb_file_name: str
        The path of the image.
    radius: float
        The radius of the camera.
    scene_scale: float
        The scale of the scene.

    Notes
    -----
    fx, fy, cx, cy and fovX, fovY are mutually exclusive. 
    If fx, fy, cx, cy are provided, fovX, fovY will be calculated from them. 
    If fovX, fovY are provided, fx, fy, cx, cy will be calculated from them.

    Examples
    --------
    >>> idx = 1
    >>> width = 800
    >>> height = 600
    >>> R = np.eye(3)
    >>> T = np.zeros(3)
    >>> focal_length_x = 800
    >>> focal_length_y = 800
    >>> camera = TrainableCamera(idx=idx, R=R, T=T, width=width, height=height, rgb_file_name='1_rgb.png',
                        fx=focal_length_x, fy=focal_length_y, cx=width/2, cy=height/2, bg=0.0, scene_scale=1.0)
    """

    def __post_init__(self):
        super().__post_init__()
        self._omega = nn.Parameter(torch.zeros(6).requires_grad_(True))

    @property
    def param_groups(self) -> List:
        """
        Get the parameter groups of the camera.

        Returns
        -------
        param_groups: List

        Notes
        -----
        property of the camera class

        """
        return [self._omega]

    @property
    def _exp_factor(self) -> Float[Tensor, "4 4"]:
        """
        Get the exponential fix factor of the camera.

        Returns
        -------
        exp_factor: Float[Tensor, "4 4"]

        Notes
        -----
        property of the camera class

        """
        return se3_exp_map(self._omega.view(1, 6)).view(4, 4)

    @property
    def world_view_transform(self) -> Float[Tensor, "4 4"]:
        """
        Get the fixed world to view transform from the camera.

        Returns
        -------
        _world_view_transform: Float[Tensor, "4 4"]
            The world to view transform.

        Notes
        -----
        property of the camera class

        """
        return self._world_view_transform @ self._exp_factor

    @property
    def full_proj_transform(self) -> Float[Tensor, "4 4"]:
        """
        Get the full projection matrix from the camera.

        Returns
        -------
        _full_proj_transform: Float[Tensor, "4 4"]

        Notes
        -----
        property of the camera class

        """
        return (self.world_view_transform.unsqueeze(0).bmm(self._projection_matrix.unsqueeze(0))).squeeze(0)

    def load2device(self, device):
        """
        Load all the parameters of the camera to the device.

        Parameters
        ----------
        device: str
            The device to load the parameters to.

        Notes
        -----
            This function will load part of the parameters of the camera to the device.
        """
        self._world_view_transform = self._world_view_transform.to(device)
        self._projection_matrix = self._projection_matrix.to(device)
        self._full_proj_transform = self._full_proj_transform.to(device)
        self._intrinsic_matrix = self._intrinsic_matrix.to(device)
        self._camera_center = self._camera_center.to(device)
        self._omega = self._omega.to(device)


class Cameras:
    """
    Cameras class used in Pointrix, which are used to generate camera paths.

    Parameters
    ----------
    camera_list: List[Camera]
        The list of the cameras.

    Examples
    --------
    >>> width = 800
    >>> height = 600
    >>> R = np.eye(3)
    >>> T = np.zeros(3)
    >>> focal_length_x = 800
    >>> focal_length_y = 800
    >>> camera1 = Camera(idx=1, R=R, T=T, width=width, height=height, rgb_file_name='1_rgb.png',
                        fx=focal_length_x, fy=focal_length_y, cx=width/2, cy=height/2, bg=0.0, scene_scale=1.0)
    >>> camera2 = Camera(idx=2, R=R, T=T, width=width, height=height, rgb_file_name='1_rgb.png', 
                        fx=focal_length_x, fy=focal_length_y, cx=width/2, cy=height/2, bg=0.0, scene_scale=1.0)
    >>> cameras = Cameras([camera1, camera2])
    """

    def __init__(self, camera_list: List[Camera]):
        self.camera_type = camera_list[0].__class__
        self.cameras = camera_list
        self.num_cameras = len(camera_list)
        self.Rs = torch.stack([cam.R for cam in camera_list], dim=0)
        self.Ts = torch.stack([cam.T for cam in camera_list], dim=0)
        self.projection_matrices = torch.stack(
            [cam.projection_matrix for cam in camera_list], dim=0)
        self.world_view_transforms = torch.stack(
            [cam.world_view_transform for cam in camera_list], dim=0)
        self.full_proj_transforms = torch.stack(
            [cam.full_proj_transform for cam in camera_list], dim=0)
        self.camera_centers = torch.stack(
            [cam.camera_center for cam in camera_list], dim=0)  # (N, 3)

        self.translate, self.radius = self.get_translate_radius()

    def __len__(self):
        return self.num_cameras

    def __getitem__(self, index):
        return self.cameras[index]

    def get_translate_radius(self):
        """
        Get the translate and radius of the cameras.

        Returns
        -------
        translate: Float[Tensor, "3"]
            The translation vector of the cameras.
        radius: float
            The radius of the cameras.
        """
        avg_cam_center = torch.mean(self.camera_centers, dim=0, keepdims=True)
        dist = torch.linalg.norm(
            self.camera_centers - avg_cam_center, dim=1, keepdims=True)
        diagonal = torch.max(dist)
        center = avg_cam_center[0]
        radius = diagonal * 1.1
        translate = -center

        return translate, radius

    def generate_camera_path(self, num_frames: int, mode: str = "Dolly") -> List[Camera]:
        """
        Generate the camera path.

        Parameters
        ----------
        num_frames: int
            The number of frames of the camera path.

        Returns
        -------
        camera_path: Float[Tensor, "num_frames 4 4"]
            The camera path.
        """
        SE3_poses = torch.zeros(self.num_cameras, 3, 4)
        for i in range(self.num_cameras):
            SE3_poses[i, :3, :3] = self.cameras[i].R
            SE3_poses[i, :3, 3] = self.cameras[i].T

        mean_pose = torch.mean(SE3_poses[:, :, 3], 0)

        # Select the best idx for rendering
        render_idx = 0
        best_dist = 1000000000
        for iidx in range(SE3_poses.shape[0]):
            cur_dist = torch.mean((SE3_poses[iidx, :, 3] - mean_pose) ** 2)
            if cur_dist < best_dist:
                best_dist = cur_dist
                render_idx = iidx
        # if self.cameras[render_idx].near_far is not None and self.cameras[render_idx].near_far.any():
        #     sc = self.cameras[render_idx].near_far[0] * 0.75
        # else:
        sc = 1

        c2w = SE3_poses.cpu().detach().numpy()[render_idx]

        fovX = self.cameras[render_idx].fovX
        fovY = self.cameras[render_idx].fovY
        width = self.cameras[render_idx].width
        height = self.cameras[render_idx].height
        focalx = width / (2 * np.tan(fovX * 0.5))
        focaly = height / (2 * np.tan(fovY * 0.5))

        if mode == "Dolly":
            return self.dolly(c2w, [focalx, focaly], width, height, sc=sc, length=SE3_poses.shape[0], num_frames=num_frames)
        elif mode == "Zoom":
            return self.zoom(c2w, [focalx, focaly], width, height, sc=sc, length=SE3_poses.shape[0], num_frames=num_frames)
        elif mode == "Spiral":
            return self.spiral(c2w, [focalx, focaly], width, height, sc=sc, length=SE3_poses.shape[0], num_frames=num_frames)
        elif mode == "Circle":
            return self.circle([focalx, focaly], width, height, sc=sc, length=SE3_poses.shape[0], num_frames=num_frames)

    def pose_to_cam(self, poses, focals, width, height):
        """
        Generate the camera path from poses.

        Parameters
        ----------
        poses: Float[Tensor, "num_frames 3 4"]
            The poses of the camera path.
        focals: Float[Tensor, "num_frames"]
            The focal lengths of the camera path.
        width: int
            The width of the image.
        height: int 
            The height of the image.
        """
        camera_list = []
        for idx in range(focals.shape[0]):
            pose = poses[idx]
            focal = focals[idx]

            R = pose[:3, :3]
            T = pose[:3, 3]

            fovX = 2 * math.atan(width / (2 * focal))
            fovY = 2 * math.atan(height / (2 * focal))

            cam = self.camera_type(idx=idx, R=R, T=T, width=width, height=height, rgb_file_name='',
                                   fovX=fovX, fovY=fovY, bg=0.0, scene_scale=1.0)
            camera_list.append(cam)
        return camera_list

    def dolly(self, c2w, focal, width, height, sc, length, num_frames):
        """
        Generate the camera path with dolly zoom.

        Parameters
        ----------
        c2w: Float[Tensor, "3 4"]
            The camera to world transform.
        focal: list[float]
            The focal length of the camera.
        sc: float
            The scale of the scene.
        length: int
            The length of the camera path.
        num_frames: int
            The number of frames of the camera path.

        Returns
        -------
        camera_path: Float[Tensor, "num_frames 4 4"]
            The camera path.
        """
        # TODO: how to define the max_disp
        max_disp = 2.0

        max_trans = max_disp / focal[0] * sc
        dolly_poses = []
        dolly_focals = []
        # Dolly zoom
        for i in range(num_frames):
            x_trans = 0.0
            y_trans = 0.0
            z_trans = max_trans * 2.5 * i / float(30 // 2)
            i_pose = np.concatenate(
                [
                    np.concatenate(
                        [np.eye(3), np.array([x_trans, y_trans, z_trans])[
                            :, np.newaxis]],
                        axis=1,
                    ),
                    np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :],
                ],
                axis=0,
            )
            i_pose = np.linalg.inv(i_pose)
            ref_pose = np.concatenate(
                [c2w[:3, :4], np.array([0.0, 0.0, 0.0, 1.0])[
                    np.newaxis, :]], axis=0
            )
            render_pose = np.dot(ref_pose, i_pose)
            dolly_poses.append(render_pose[:3, :])
            new_focal = focal[0] - focal[0] * 0.1 * z_trans / max_trans / 2.5
            dolly_focals.append(new_focal)
        dolly_poses = np.stack(dolly_poses, 0)[:, :3]
        dolly_focals = np.stack(dolly_focals, 0)

        return self.pose_to_cam(dolly_poses, dolly_focals, width, height)

    def zoom(self, c2w, focal, width, height, sc, length, num_frames):
        """
        Generate the camera path with zoom.

        Parameters
        ----------
        c2w: Float[Tensor, "3 4"]
            The camera to world transform.
        focal: list[float]
            The focal length of the camera.
        width: int
            The width of the image.
        height: int
            The height of the image.
        sc: float
            The scale of the scene.
        length: int
            The length of the camera path.
        num_frames: int
            The number of frames of the camera path.

        Returns
        -------
        camera_path: Float[Tensor, "num_frames 4 4"]
            The camera path.
        """
        # TODO: how to define the max_disp
        max_disp = 20.0

        max_trans = max_disp / focal[0] * sc
        zoom_poses = []
        zoom_focals = []
        # Zoom in
        # Zoom in
        for i in range(num_frames):
            x_trans = 0.0
            y_trans = 0.0
            # z_trans = max_trans * np.sin(2.0 * np.pi * float(i) / float(num_novelviews)) * args.z_trans_multiplier
            z_trans = max_trans * 2.5 * i / float(30 // 2)
            i_pose = np.concatenate(
                [
                    np.concatenate(
                        [np.eye(3), np.array([x_trans, y_trans, z_trans])[
                            :, np.newaxis]],
                        axis=1,
                    ),
                    np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :],
                ],
                axis=0,
            )

            # torch.tensor(np.linalg.inv(i_pose)).float()
            i_pose = np.linalg.inv(i_pose)

            ref_pose = np.concatenate(
                [c2w[:3, :4], np.array([0.0, 0.0, 0.0, 1.0])[
                    np.newaxis, :]], axis=0
            )

            render_pose = np.dot(ref_pose, i_pose)
            zoom_poses.append(render_pose[:3, :])
            zoom_focals.append(focal[0])

        zoom_poses = np.stack(zoom_poses, 0)[:, :3]
        zoom_focals = np.stack(zoom_focals, 0)
        return self.pose_to_cam(zoom_poses, zoom_focals, width, height)

    def spiral(self, c2w, focal, width, height, sc, length, num_frames):
        """
        Generate the camera path with spiral.

        Parameters
        ----------
        c2w: Float[Tensor, "3 4"]
            The camera to world transform.
        focal: list[float]
            The focal length of the camera.
        width: int
            The width of the image.
        height: int
            The height of the image.
        sc: float
            The scale of the scene.
        length: int
            The length of the camera path.
        num_frames: int
            The number of frames of the camera path.
        """
        # TODO: how to define the max_disp
        max_disp = 120.0

        max_trans = max_disp / focal[0] * sc

        spiral_poses = []
        spiral_focals = []
        # Rendering teaser. Add translation.
        for i in range(num_frames):
            x_trans = max_trans * 1.5 * \
                np.sin(2.0 * np.pi * float(i) / float(60)) * 2.0
            y_trans = (
                max_trans
                * 1.5
                * (np.cos(2.0 * np.pi * float(i) / float(60)) - 1.0)
                * 2.0
                / 3.0
            )
            z_trans = 0.0

            i_pose = np.concatenate(
                [
                    np.concatenate(
                        [np.eye(3), np.array([x_trans, y_trans, z_trans])[
                            :, np.newaxis]],
                        axis=1,
                    ),
                    np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :],
                ],
                axis=0,
            )

            i_pose = np.linalg.inv(i_pose)

            ref_pose = np.concatenate(
                [c2w[:3, :4], np.array([0.0, 0.0, 0.0, 1.0])[
                    np.newaxis, :]], axis=0
            )

            render_pose = np.dot(ref_pose, i_pose)
            # output_poses.append(np.concatenate([render_pose[:3, :], hwf], 1))
            spiral_poses.append(render_pose[:3, :])
            spiral_focals.append(focal[0])
        spiral_poses = np.stack(spiral_poses, 0)[:, :3]
        spiral_focals = np.stack(spiral_focals, 0)
        return self.pose_to_cam(spiral_poses, spiral_focals, width, height)

    def circle(self, focal, width, height, sc, length, num_frames):
        """
        Generate the camera path with circle.

        Parameters
        ----------
        c2w: Float[Tensor, "3 4"]
            The camera to world transform.
        focal: list[float]
            The focal length of the camera.
        width: int
            The width of the image.
        height: int
            The height of the image.
        sc: float
            The scale of the scene.
        length: int
            The length of the camera path.
        num_frames: int
            The number of frames of the camera path.
        """
        camera_list = []

        def trans_t(t): return torch.Tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, t],
            [0, 0, 0, 1]]).float()

        def rot_phi(phi): return torch.Tensor([
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1]]).float()

        def rot_theta(th): return torch.Tensor([
            [np.cos(th), 0, -np.sin(th), 0],
            [0, 1, 0, 0],
            [np.sin(th), 0, np.cos(th), 0],
            [0, 0, 0, 1]]).float()

        def rot_yaw(yaw): return torch.Tensor([
            [np.cos(yaw), -np.sin(yaw), 0, 0],
            [np.sin(yaw), np.cos(yaw), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]).float()

        def pose_spherical(theta, phi, yaw, radius):
            c2w = trans_t(radius)
            c2w = rot_phi(phi/180.*np.pi) @ c2w
            c2w = rot_theta(theta/180.*np.pi) @ c2w
            c2w = rot_yaw(yaw/180.*np.pi) @ c2w
            c2w = torch.Tensor(
                np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
            return c2w

        render_poses = torch.stack([pose_spherical(-4, -90., angle, 4.0)
                                   for angle in np.linspace(-180, 180, 100+1)[:-1]], 0)

        for idx, poses in enumerate(render_poses):
            matrix = np.linalg.inv(np.array(poses))
            R = -np.transpose(matrix[:3, :3])
            R[:, 0] = -R[:, 0]
            T = -matrix[:3, 3]

            fovX = 2 * math.atan(width / (2 * focal[0]))
            fovY = 2 * math.atan(height / (2 * focal[0]))

            cam = self.camera_type(idx=idx, R=R, T=T, width=width, height=height, rgb_file_name='',
                                   fovX=fovX, fovY=fovY, bg=0.0, scene_scale=1.0)
            camera_list.append(cam)
        return camera_list
