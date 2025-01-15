import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

def to8b(x): return (255 * np.clip(x, 0, 1)).astype(np.uint8)

def visualize_depth(depth, minmax=None, cmap=cv2.COLORMAP_JET, tensorboard=False):
    """
    depth: (H, W)
    """
    if type(depth) is not np.ndarray:
        depth = depth.detach().cpu().numpy()

    x = np.nan_to_num(depth)  # change nan to 0
    if minmax is None:
        mi = np.min(x[x > 0])  # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi, ma = minmax

    x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    x = (255 * x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_)  # (3, H, W)
    if tensorboard:
        return x_
    x = to8b(x_.detach().cpu().numpy().transpose(1, 2, 0))
    return x

def visualize_rgb(rgb):
    """
    rgb: (H, W, 3)
    """
    rgb = torch.clamp(rgb, 0.0, 1.0)
    rgb = rgb.detach().cpu().numpy()
    return to8b(rgb.transpose(1, 2, 0))