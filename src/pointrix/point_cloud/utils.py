
import torch
import numpy as np
from pointrix.logger.writer import Logger

def RGB2SH(rgb):
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0

def unwarp_name(name, prefix="point_cloud."):
    return name.replace(prefix, "")

def get_random_points(num_points, radius):
    pos = np.random.random((num_points, 3)) * 2 * radius - radius
    pos = torch.from_numpy(pos).float()
    return pos

def get_random_feauture(num_points, feat_dim):
    feart = np.random.random((num_points, feat_dim)) / 255.0
    feart = torch.from_numpy(feart).float()
    return feart

def points_init(init_cfg, point_cloud):
    init_type = init_cfg.init_type
    
    if init_type == 'random' and point_cloud is None:
        num_points = init_cfg.num_points
        Logger.log("Number of points at initialisation : ", num_points)
        pos = get_random_points(num_points, init_cfg.radius)
        features = get_random_feauture(num_points, init_cfg.feat_dim)

    elif init_type == 'positive_Z_random' and point_cloud is None:
        num_points = init_cfg.num_points
        Logger.log("Number of points at initialisation : ", num_points)
        pos = get_random_points(num_points, init_cfg.radius)
        features = get_random_feauture(num_points, init_cfg.feat_dim)
        pos[:, 2] = pos[:, 2] + 1  # +1 for opencv coordinate
        
    else:
        Logger.log("Number of points at initialisation : ", point_cloud.positions.shape[0])
        pos = np.asarray(point_cloud.positions)
        pos = torch.from_numpy(pos).float()
        features = RGB2SH(torch.tensor(np.asarray(point_cloud.colors)).float())
        
        if "random" in init_type:
            num_points = init_cfg.num_points
            print("Extend the initialiased point with random : ", num_points)
            max_dis = torch.abs(pos).max().item()
            pos_ext = get_random_points(num_points, max_dis * init_cfg.radius)
            features_ext = get_random_feauture(num_points, features.shape[1])
            
            pos = torch.cat((pos, pos_ext), dim=0)
            features = torch.cat((features, features_ext), dim=0)
            
    return pos, features