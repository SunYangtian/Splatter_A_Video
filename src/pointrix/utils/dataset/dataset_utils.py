import functools
import logging
import math
import numpy as np
from collections.abc import Mapping
from torch.utils.data import Dataset
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def getNerfppNorm(Rs, Ts):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for R, T in zip(Rs, Ts):
        W2C = getWorld2View2(R, T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

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

def force_full_init(old_func: Callable) -> Any:
    @functools.wraps(old_func)
    def wrapper(obj: object, *args, **kwargs):
        # The instance must have `full_init` method.
        if not hasattr(obj, 'full_init'):
            raise AttributeError(f'{type(obj)} does not have full_init '
                                 'method.')
        # If instance does not have `_fully_initialized` attribute or
        # `_fully_initialized` is False, call `full_init` and set
        # `_fully_initialized` to True
        if not getattr(obj, '_fully_initialized', False):
            print(
                f'Attribute `_fully_initialized` is not defined in '
                f'{type(obj)} or `type(obj)._fully_initialized is '
                'False, `full_init` will be called and '
                f'{type(obj)}._fully_initialized will be set to True',
                logger='current',
                level=logging.WARNING)
            obj.full_init()  # type: ignore
            obj._fully_initialized = True  # type: ignore

        return old_func(obj, *args, **kwargs)

    return wrapper
