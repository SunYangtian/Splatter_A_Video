import torch
from torch import Tensor
from jaxtyping import Float
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def psnr(img1:Float[Tensor, "H W C"], img2:Float[Tensor, "H W C"]):
    """
    Compute the PSNR between two images.

    Parameters
    ----------
    img1 : torch.Tensor
        The first image.
    img2 : torch.Tensor
        The second image.
    """
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def l1_loss(network_output:Float[Tensor, "H ..."], gt:Float[Tensor, "H ..."]):
    """
    Compute the L1 loss between the network output and the ground truth.

    Parameters
    ----------
    network_output : torch.Tensor
        The network output.
    gt : torch.Tensor
        The ground truth.
    
    Returns
    -------
    torch.Tensor
        The L1 loss.
    """
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output:Float[Tensor, "H ..."], gt:Float[Tensor, "H ..."]):
    """
    Compute the L2 loss between the network output and the ground truth.

    Parameters
    ----------
    network_output : torch.Tensor
        The network output.
    gt : torch.Tensor
        The ground truth.
    
    Returns
    -------
    torch.Tensor
        The L2 loss.
    """
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1:Float[Tensor, "H W C"], img2:Float[Tensor, "H W C"], window_size=11, size_average=True):
    """
    Compute the SSIM between two images.

    Parameters
    ----------
    img1 : torch.Tensor
        The first image.
    img2 : torch.Tensor
        The second image.
    window_size : int, optional
        The window size, by default 11
    size_average : bool, optional
        Whether to average the SSIM or not, by default True
    """
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    