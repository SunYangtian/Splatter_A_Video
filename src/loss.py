import torch
from torch import nn
import math

# copy from MiDaS
def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


def mse_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))
    res = prediction - target
    image_loss = torch.sum(mask * res * res, (1, 2))

    return reduction(image_loss, 2 * M)

def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)


class MSELoss(nn.Module):
    def __init__(self, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        return mse_loss(prediction, target, mask, reduction=self.__reduction)


class GradientLoss(nn.Module):
    def __init__(self, scales=4, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step],
                                   mask[:, ::step, ::step], reduction=self.__reduction)

        return total
    

class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction='batch-based'):
        super().__init__()

        self.__data_loss = MSELoss(reduction=reduction)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None

    def forward(self, prediction, target, mask):

        scale, shift = compute_scale_and_shift(prediction, target, mask)
        self.__prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        total = self.__data_loss(self.__prediction_ssi, target, mask)
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(self.__prediction_ssi, target, mask)

        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)
# end copy



import torch
import torch.nn.functional as F

def depth_correlation_loss(gt_depth, rendered_depth, patch_size, num_patches):
    """
    Compute the depth correlation loss between the ground truth and rendered depth maps.

    Args:
        gt_depth (torch.Tensor): The ground truth depth map. [H, W, 1]
        rendered_depth (torch.Tensor): The rendered depth map. [H, W, 1]
        patch_size (int): The size of the patches to sample.
        num_patches (int): The number of patches to sample.
    """

    # Find the dimensions of the depth maps
    height, width, _ = gt_depth.size()
    grid_i, grid_j = torch.meshgrid([torch.arange(patch_size), torch.arange(patch_size)], indexing='ij')
    grid = torch.stack([grid_i, grid_j], dim=-1).float().to(gt_depth.device)  # [patch_size, patch_size, 2]

    
    # Sample patches from the depth maps and compute correlations
    ii = torch.randint(0, height - patch_size, (num_patches,))  # [N,]
    jj = torch.randint(0, width - patch_size, (num_patches,))
    sampled_indexes = torch.stack([ii, jj], dim=1).to(gt_depth.device)  # [N, 2]
    sampled_indexes = sampled_indexes[:, None, None, :] + grid[None]  # [N, patch_size, patch_size, 2]

    # Extract the patches from the depth maps
    sampled_indexes = sampled_indexes[:, :, :, 0] * width + sampled_indexes[:, :, :, 1]  # [N, patch_size, patch_size]
    sampled_indexes = sampled_indexes.reshape(num_patches, -1).long()
    sampled_gt_patches = torch.gather(gt_depth.reshape(1,height*width).repeat(num_patches,1), dim=1, index=sampled_indexes)

    sampled_rendered_patches = torch.gather(rendered_depth.reshape(1,height*width).repeat(num_patches,1), dim=1, index=sampled_indexes)

    pcc = (sampled_rendered_patches*sampled_gt_patches).mean(dim=1) - sampled_rendered_patches.mean(dim=1)*sampled_gt_patches.mean(dim=1)
    pcc = pcc / (sampled_rendered_patches.std(dim=1) * sampled_gt_patches.std(dim=1))

    return 1 - pcc.mean()


def depth_loss_dpt(pred_depth, gt_depth, weight=None):
    """
    :param pred_depth:  (H, W)
    :param gt_depth:    (H, W)
    :param weight:      (H, W)
    :return:            scalar
    """
    
    t_pred = torch.median(pred_depth)
    s_pred = torch.mean(torch.abs(pred_depth - t_pred))

    t_gt = torch.median(gt_depth)
    s_gt = torch.mean(torch.abs(gt_depth - t_gt))

    pred_depth_n = (pred_depth - t_pred) / s_pred
    gt_depth_n = (gt_depth - t_gt) / s_gt

    if weight is not None:
        loss = F.mse_loss(pred_depth_n, gt_depth_n, reduction='none')
        loss = loss * weight
        loss = loss.sum() / (weight.sum() + 1e-8)
    else:
        loss = F.mse_loss(pred_depth_n, gt_depth_n)
    return loss