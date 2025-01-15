import os
import glob
import json
import imageio
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import multiprocessing as mp
from util import normalize_coords, gen_grid_np


def get_sample_weights(flow_stats):
    sample_weights = {}
    for k in flow_stats.keys():
        sample_weights[k] = {}
        total_num = np.array(list(flow_stats[k].values())).sum()
        for j in flow_stats[k].keys():
            sample_weights[k][j] = 1. * flow_stats[k][j] / total_num
    return sample_weights


class GSRAFTExhaustiveDataset(Dataset):
    def __init__(self, args, max_interval=None):
        self.args = args
        self.seq_dir = args.data_dir
        self.seq_name = os.path.basename(self.seq_dir.rstrip('/'))
        self.img_dir = os.path.join(self.seq_dir, 'color')
        self.flow_dir = os.path.join(self.seq_dir, 'raft_exhaustive')
        img_names = sorted(os.listdir(self.img_dir))
        self.num_imgs = min(self.args.num_imgs, len(img_names))
        self.base_idx = self.args.base_idx
        self.img_names = img_names[self.base_idx:self.num_imgs+self.base_idx]

        h, w, _ = imageio.imread(os.path.join(self.img_dir, img_names[0])).shape
        self.h, self.w = h, w
        max_interval = self.num_imgs - 1 if not max_interval else max_interval
        self.max_interval = mp.Value('i', max_interval)
        self.num_pts = self.args.num_pts
        self.grid = gen_grid_np(self.h, self.w)
        flow_stats = json.load(open(os.path.join(self.seq_dir, 'flow_stats.json')))
        self.sample_weights = get_sample_weights(flow_stats)

    def __len__(self):
        return self.num_imgs * 100000

    def set_max_interval(self, max_interval):
        self.max_interval.value = min(max_interval, self.num_imgs - 1)

    def increase_max_interval_by(self, increment):
        curr_max_interval = self.max_interval.value
        self.max_interval.value = min(curr_max_interval + increment, self.num_imgs - 1)

    def __getitem__(self, idx):
        cached_flow_pred_dir = os.path.join('out', '{}_{}'.format(self.args.expname, self.seq_name), 'flow')
        cached_flow_pred_files = sorted(glob.glob(os.path.join(cached_flow_pred_dir, '*')))
        flow_error_file = os.path.join(os.path.dirname(cached_flow_pred_dir), 'flow_error.txt')
        if os.path.exists(flow_error_file):
            flow_error = np.loadtxt(flow_error_file)
            id1_sample_weights = flow_error / np.sum(flow_error)
            id1 = np.random.choice(self.num_imgs, p=id1_sample_weights)
        else:
            id1 = idx % self.num_imgs

        img_name1 = self.img_names[id1]
        max_interval = min(self.max_interval.value, self.num_imgs - 1)
        img2_candidates = sorted(list(self.sample_weights[img_name1].keys()))
        img2_candidates = img2_candidates[max(self.base_idx+id1-max_interval, self.base_idx):min(self.base_idx+id1+max_interval, self.base_idx+self.num_imgs - 1)]

        # sample more often from i-1 and i+1
        id2s = np.array([self.img_names.index(n) for n in img2_candidates])
        sample_weights = np.array([self.sample_weights[img_name1][i] for i in img2_candidates])
        sample_weights /= np.sum(sample_weights)
        sample_weights[np.abs(id2s - id1) <= 1] = 0.5
        sample_weights /= np.sum(sample_weights)

        img_name2 = np.random.choice(img2_candidates, p=sample_weights)
        id2 = self.img_names.index(img_name2)
        frame_interval = abs(id1 - id2)

        # read image, flow and confidence
        img1 = imageio.imread(os.path.join(self.img_dir, img_name1)) / 255.

        flow_file = os.path.join(self.flow_dir, '{}_{}.npy'.format(img_name1, img_name2))
        flow = np.load(flow_file)
        mask_file = flow_file.replace('raft_exhaustive', 'raft_masks').replace('.npy', '.png')
        masks = imageio.imread(mask_file) / 255.

        coord1 = self.grid
        coord2 = self.grid + flow

        cycle_consistency_mask = masks[..., 0] > 0
        occlusion_mask = masks[..., 1] > 0

        if frame_interval == 1:
            mask = np.ones_like(cycle_consistency_mask)
        else:
            mask = cycle_consistency_mask | occlusion_mask

        if mask.sum() == 0:
            invalid = True
            mask = np.ones_like(cycle_consistency_mask)
        else:
            invalid = False

        pair_weight = np.cos((frame_interval - 1.) / max_interval * np.pi / 2)

        pts1 = torch.from_numpy(coord1).float().reshape(-1,2)
        pts2 = torch.from_numpy(coord2).float().reshape(-1,2)

        weights = torch.from_numpy(mask).float().reshape(-1,1) * pair_weight

        gt_rgb1 = torch.from_numpy(img1).float().reshape(-1,3)

        if invalid:
            weights = torch.zeros_like(weights)

        data = {'ids1': id1,
                'ids2': id2,
                'pts1': pts1,  # [n_pts, 2]
                'pts2': pts2,  # [n_pts, 2]
                'gt_rgb1': gt_rgb1,  # [n_pts, 3]
                'weights': weights,  # [n_pts, 1]
                }
        return data