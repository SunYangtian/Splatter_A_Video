import os
import torch
import numpy as np

from simple_knn._C import distCUDA2

def inverse_sigmoid(x):
    return torch.log(x/(1-x))


def build_rotation(r):
    norm = torch.sqrt(r[:, 0]*r[:, 0] + r[:, 1]*r[:, 1] +
                      r[:, 2]*r[:, 2] + r[:, 3]*r[:, 3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R


def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):

    s = scaling_modifier * scaling

    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float)

    R = build_rotation(rotation)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    # L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    L = L @ L.transpose(1, 2)

    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float)

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]

    return uncertainty


def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def gaussian_point_init(position, max_sh_degree, init_opacity=0.1):
    num_points = len(position)
    avg_dist = torch.clamp_min(
        distCUDA2(position.cuda()), 
        0.0000001
    )[..., None].cpu()
    # position_np = position.detach().cpu().numpy()
    # Build the nearest neighbors model
    # from sklearn.neighbors import NearestNeighbors

    # k = 3
    # nn_model = NearestNeighbors(
    #     n_neighbors=k + 1, 
    #     algorithm="auto", 
    #     metric="euclidean"
    # ).fit(position_np)
    
    # distances, indices = nn_model.kneighbors(position_np)
    # distances = distances[:, 1:].astype(np.float32)
    # distances = torch.from_numpy(distances)
    # avg_dist = distances.mean(dim=-1, keepdim=True)
    
    # avg_dist = torch.clamp_min(avg_dist, 0.0000001)
    scales = torch.log(torch.sqrt(avg_dist)).repeat(1, 3)
    # scales = torch.log(torch.ones_like(avg_dist) * 0.03).repeat(1, 3)  # TODO check this
    rots = torch.zeros((num_points, 4))
    rots[:, 0] = 1

    init_one = torch.ones(
        (num_points, 1),
        dtype=torch.float32
    )
    opacities = inverse_sigmoid(init_opacity * init_one)
    # opacities = inverse_sigmoid(0.01 * init_one)
    features_rest = torch.zeros(
        (num_points, (max_sh_degree+1) ** 2 - 1, 3),
        dtype=torch.float32
    )

    return scales, rots, opacities, features_rest