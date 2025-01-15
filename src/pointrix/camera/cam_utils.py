import numpy as np
from scipy.spatial.transform import Rotation as R
import math
import torch

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 1 / tanHalfFovX
    P[1, 1] = 1 / tanHalfFovY
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

from pointrix.camera.camera import Camera
def construct_canonical_camera(width, height, c2w=None):
    if c2w is None:
        c2w = np.eye(4)
        c2w[:3,3] = np.array([0, 0, 0.0])  #
    # # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
    # c2w[:3, 1:3] *= -1
    ## Use opencv cooradinates here. Look at +Z

    # get the world-to-camera transform and set R, T
    w2c = np.linalg.inv(c2w)
    # R is stored transposed due to 'glm' in CUDA code
    R = np.transpose(w2c[:3, :3])
    T = w2c[:3, 3]

    fovx = np.pi / 2.0
    fovy = focal2fov(fov2focal(fovx, width), height)
    FovY = fovy
    FovX = fovx
    camera = Camera(idx=0, R=R, T=T, width=width, height=height,
                    fovX=FovX, fovY=FovY, bg=1.0)
    return camera

def construct_canonical_camera_from_focal(width, height, focal):
    c2w = np.eye(4)
    c2w[:3,3] = np.array([0, 0, 0.0])  #

    # get the world-to-camera transform and set R, T
    w2c = np.linalg.inv(c2w)
    # R is stored transposed due to 'glm' in CUDA code
    R = np.transpose(w2c[:3, :3])
    T = w2c[:3, 3]

    fovx = focal2fov(focal, width)
    fovy = focal2fov(focal, height)
    FovY = fovy
    FovX = fovx
    camera = Camera(idx=0, R=R, T=T, width=width, height=height,
                    fovX=FovX, fovY=FovY, bg=1.0)
    return camera

def construct_canonical_camera_from_Zdistance(width, height, Zdistance=0.0):
    c2w = np.eye(4)
    camera_distance = Zdistance
    c2w[:3,3] = np.array([0, 0, -1*camera_distance])

    # get the world-to-camera transform and set R, T
    w2c = np.linalg.inv(c2w)
    # R is stored transposed due to 'glm' in CUDA code
    R = np.transpose(w2c[:3, :3])
    T = w2c[:3, 3]

    fovx = np.arctan(1.0 / (camera_distance+1.0)) * 2.0
    fovy = focal2fov(fov2focal(fovx, width), height)
    FovY = fovy
    FovX = fovx
    camera = Camera(idx=0, R=R, T=T, width=width, height=height,
                    fovX=FovX, fovY=FovY, bg=1.0)
    return camera




class MiniCam2:
    def __init__(self, c2w, width, height, fovy, fovx, znear, zfar, fid, to_opengl=False):
        # c2w (pose) should be in NeRF convention.

        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.fid = fid
        self.c2w = c2w

        if to_opengl:
            c2w[:3,1:3] *= -1
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])
        T = w2c[:3,3]

        trans=np.array([0.0, 0.0, 0.0])
        scale=1.0

        self.extrinsic_matrix = torch.tensor(getWorld2View2(R, T, trans, scale))
        self.world_view_transform = self.extrinsic_matrix.transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1).cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        ##### set extrinsic / intrinsic matrix
        self.fx = width / (2 * np.tan(self.FoVx * 0.5))
        self.fy = height / (2 * np.tan(self.FoVy * 0.5))
        self.cx = width / 2
        self.cy = height / 2
        self.intrinsic_matrix = self._get_intrinsics(
                    self.fx, self.fy, self.cx, self.cy)
        

    @staticmethod
    def _get_intrinsics(fx: float, fy: float, cx: float, cy: float):
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


def dot(x, y):
    if isinstance(x, np.ndarray):
        return np.sum(x * y, -1, keepdims=True)
    else:
        return torch.sum(x * y, -1, keepdim=True)


def length(x, eps=1e-20):
    if isinstance(x, np.ndarray):
        return np.sqrt(np.maximum(np.sum(x * x, axis=-1, keepdims=True), eps))
    else:
        return torch.sqrt(torch.clamp(dot(x, x), min=eps))


def safe_normalize(x, eps=1e-20):
    return x / length(x, eps)


def look_at(campos, target, opengl=True):
    # campos: [N, 3], camera/eye position
    # target: [N, 3], object to look at
    # return: [N, 3, 3], rotation matrix
    if not opengl:
        # camera forward aligns with -z
        forward_vector = safe_normalize(target - campos)
        up_vector = np.array([0, 1, 0], dtype=np.float32)
        right_vector = safe_normalize(np.cross(forward_vector, up_vector))
        up_vector = safe_normalize(np.cross(right_vector, forward_vector))
    else:
        # camera forward aligns with +z
        forward_vector = safe_normalize(campos - target)
        up_vector = np.array([0, 1, 0], dtype=np.float32)
        right_vector = safe_normalize(np.cross(up_vector, forward_vector))
        up_vector = safe_normalize(np.cross(forward_vector, right_vector))
    R = np.stack([right_vector, up_vector, forward_vector], axis=1)
    return R


# elevation & azimuth to pose (cam2world) matrix
def orbit_camera(elevation, azimuth, radius=1, is_degree=True, target=None, opengl=True):
    # radius: scalar
    # elevation: scalar, in (-90, 90), from +y to -y is (-90, 90)
    # azimuth: scalar, in (-180, 180), from +z to +x is (0, 90)
    # return: [4, 4], camera pose matrix
    if is_degree:
        elevation = np.deg2rad(elevation)
        azimuth = np.deg2rad(azimuth)
    x = radius * np.cos(elevation) * np.sin(azimuth)
    y = - radius * np.sin(elevation)
    z = radius * np.cos(elevation) * np.cos(azimuth)
    if target is None:
        target = np.zeros([3], dtype=np.float32)
    campos = np.array([x, y, z]) + target  # [3]
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = look_at(campos, target, opengl)
    T[:3, 3] = campos
    return T


class OrbitCamera:
    def __init__(self, W, H, r=2, fovy=60, near=0.01, far=100, center=[0,0,0]):
        self.W = W
        self.H = H
        self.radius = r  # camera distance from center
        self.fovy = np.deg2rad(fovy)  # deg 2 rad
        self.near = near
        self.far = far
        self.center = np.array(center, dtype=np.float32)  # look at this point
        self.rot = R.from_matrix(np.eye(3))
        # self.rot = R.from_matrix(np.array([[1., 0., 0.,],
        #                                    [0., 0., -1.],
        #                                    [0., 1., 0.]]))
        self.up = np.array([0, 1, 0], dtype=np.float32)  # need to be normalized!
        self.side = np.array([1, 0, 0], dtype=np.float32)

    @property
    def fovx(self):
        return 2 * np.arctan(np.tan(self.fovy / 2) * self.W / self.H)

    @property
    def campos(self):
        return self.pose[:3, 3]

    # pose (c2w)
    @property
    def pose(self):
        # first move camera to radius
        res = np.eye(4, dtype=np.float32)
        res[2, 3] = self.radius  # opengl convention...
        # rotate
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return res

    # view (w2c)
    @property
    def view(self):
        return np.linalg.inv(self.pose)

    # projection (perspective)
    @property
    def perspective(self):
        y = np.tan(self.fovy / 2)
        aspect = self.W / self.H
        return np.array(
            [
                [1 / (y * aspect), 0, 0, 0],
                [0, -1 / y, 0, 0],
                [
                    0,
                    0,
                    -(self.far + self.near) / (self.far - self.near),
                    -(2 * self.far * self.near) / (self.far - self.near),
                ],
                [0, 0, -1, 0],
            ],
            dtype=np.float32,
        )

    # intrinsics
    @property
    def intrinsics(self):
        focal = self.H / (2 * np.tan(self.fovy / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2], dtype=np.float32)

    @property
    def mvp(self):
        return self.perspective @ np.linalg.inv(self.pose)  # [4, 4]

    def orbit(self, dx, dy):
        # rotate along camera up/side axis!
        side = self.rot.as_matrix()[:3, 0]
        up = self.rot.as_matrix()[:3, 1]
        rotvec_x = up * np.radians(-0.05 * dx)
        rotvec_y = side * np.radians(-0.05 * dy)
        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        # pan in camera coordinate system (careful on the sensitivity!)
        self.center += 0.0001 * self.rot.as_matrix()[:3, :3] @ np.array([-dx, -dy, dz])
