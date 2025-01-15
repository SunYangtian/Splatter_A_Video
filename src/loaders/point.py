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
from tqdm import tqdm
import pathlib


def get_sample_weights(flow_stats):
    sample_weights = {}
    for k in flow_stats.keys():
        sample_weights[k] = {}
        total_num = np.array(list(flow_stats[k].values())).sum()
        for j in flow_stats[k].keys():
            sample_weights[k][j] = 1. * flow_stats[k][j] / total_num
    return sample_weights

from scipy.ndimage import zoom

def extract_depth(base_dir):
    if isinstance(base_dir, str):
        base_dir = pathlib.Path(base_dir)
    result_dir = base_dir / "BA_full"
    result_f_list = sorted(list(result_dir.glob("*.npz")))

    K0 = None
    cam_c2ws, Ks, depths = [], [], []
    for tmp_i, tmp_result_f in enumerate(tqdm(result_f_list)):
        tmp_idx = int(tmp_result_f.stem)  # 0002.npz
        assert tmp_idx == tmp_i, f"{tmp_idx}, {tmp_i}"

        # disp (192, 384) float32
        # R (3, 3) float32
        # t (3,) float32
        # K (3, 3) float32
        # img (192, 384, 3) float32
        # mask_motion (192, 384) float32
        # uncertainty_pred (2, 192, 384) float32
        # gt_K (3, 3) float32
        tmp_info = np.load(tmp_result_f)
        disp = tmp_info["disp"]
        depth = 1 / (disp + 1e-8)
        cam_c2w = np.eye(4)
        cam_c2w[:3, :3] = tmp_info["R"]
        cam_c2w[:3, 3] = tmp_info["t"]
        K = np.eye(4)
        K[:3, :3] = tmp_info["K"]  # this is important

        if K0 is None:
            K0 = K
        else:
            assert np.sum(np.abs(K0 - K)) < 1e-5, f"{K0}, {K}"

        cam_c2ws.append(cam_c2w)
        Ks.append(K)
        depths.append(depth)

    ##### scale depth ######
    depths = np.stack(depths, axis=0)
    depths = depths / depths.max()

    res_dict = {"c2w": cam_c2ws, "K": Ks, "depth": depths}
    return res_dict

def extract_mask_edge(mask, kernel_size=5):
    import cv2
    # 创建一个卷积核（kernel）用于腐蚀和膨胀操作
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # 腐蚀操作
    eroded = cv2.erode(mask, kernel, iterations=1)
    # 膨胀操作
    dilated = cv2.dilate(mask, kernel, iterations=1)
    edges = dilated - eroded
    margin = 5
    edges[:margin, :] = edges[-margin:, :] = edges[:, :margin] = edges[:, -margin:] = 255
    return edges.astype(np.uint8)
        
def extract_mask_edge(mask, kernel_size=5):
    import cv2
    # 创建一个卷积核（kernel）用于腐蚀和膨胀操作
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # 腐蚀操作
    eroded = cv2.erode(mask, kernel, iterations=1)
    # 膨胀操作
    dilated = cv2.dilate(mask, kernel, iterations=1)
    edges = dilated - eroded
    margin = 5
    edges[:margin, :] = edges[-margin:, :] = edges[:, :margin] = edges[:, -margin:] = 255
    return edges.astype(np.uint8)

class PointRAFTExhaustiveDataset(Dataset):
    def __init__(self, args, max_interval=None):
        self.args = args
        self.seq_dir = args.data_dir
        self.seq_name = os.path.basename(self.seq_dir.rstrip('/'))
        self.img_dir = os.path.join(self.seq_dir, 'color')
        self.flow_dir = os.path.join(self.seq_dir, 'raft_exhaustive')
        self.depth_dir = os.path.join(self.seq_dir, 'marigold', 'depth_npy')
        self.casualSAM_dir = '/mnt/sdc/syt/dataset/DAVIS_processed_one_step_pose_depth/kangaroo/casual_sam'
        self.casualSAM_res = extract_depth(self.casualSAM_dir)
        img_names = sorted(os.listdir(self.img_dir))
        self.num_imgs = min(self.args.num_imgs, len(img_names))
        self.img_names = img_names[:self.num_imgs]

        img = imageio.imread(os.path.join(self.img_dir, img_names[0]))
        ### downscale template image. Cancelled
        # img = zoom(img, zoom=(1/self.args.down_scale, 1/self.args.down_scale, 1), order=2)
        h, w, _ = img.shape
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
        cached_flow_pred_files = sorted(glob.glob(os.path.join(cached_flow_pred_dir, '*.npy')))
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
        img2_candidates = img2_candidates[max(id1 - max_interval, 0):min(id1 + max_interval, self.num_imgs - 1)]

        # sample more often from i-1 and i+1
        id2s = np.array([self.img_names.index(n) for n in img2_candidates])
        sample_weights = np.array([self.sample_weights[img_name1][i] for i in img2_candidates])
        sample_weights /= np.sum(sample_weights)
        sample_weights[np.abs(id2s - id1) <= 1] = 0.5
        sample_weights /= np.sum(sample_weights)

        img_name2 = np.random.choice(img2_candidates, p=sample_weights)
        id2 = self.img_names.index(img_name2)

        frame_interval = abs(id1 - id2)

        # # ######## This is for debugging
        # id1, id2 = 0, 10
        # img_name1, img_name2 = self.img_names[id1], self.img_names[id2]

        # read image, flow and confidence
        img1 = imageio.imread(os.path.join(self.img_dir, img_name1)) / 255.
        img2 = imageio.imread(os.path.join(self.img_dir, img_name2)) / 255.

        # downscale image
        # scale_factor = (1/self.args.down_scale, 1/self.args.down_scale, 1)
        # img1 = zoom(img1, zoom=scale_factor, order=2)
        # img2 = zoom(img2, zoom=scale_factor, order=2)

        flow_file = os.path.join(self.flow_dir, '{}_{}.npy'.format(img_name1, img_name2))
        flow = np.load(flow_file)
        mask_file = flow_file.replace('raft_exhaustive', 'raft_masks').replace('.npy', '.png')
        masks = imageio.imread(mask_file) / 255.

        # read depth
        # depth1 = np.load(os.path.join(self.depth_dir, img_name1[:-4] + '_pred.npy'))
        # depth2 = np.load(os.path.join(self.depth_dir, img_name2[:-4] + '_pred.npy'))
        # depth1 = self.casualSAM_res["depth"][id1]
        # depth2 = self.casualSAM_res["depth"][id2]
        # tgt_h, tgt_w = masks.shape[:2]
        # depth1 = cv2.resize(
        #                         depth1, (tgt_w, tgt_h), interpolation=cv2.INTER_NEAREST
        #                     )
        # depth2 = cv2.resize(
        #                         depth2, (tgt_w, tgt_h), interpolation=cv2.INTER_NEAREST
        #                     )

        # downscale flow, masks, depth
        # flow = zoom(flow, zoom=scale_factor, order=2)
        # flow = flow / self.args.down_scale  # flow is special. Value need change.
        # masks = zoom(masks, zoom=scale_factor, order=0)  # nearest interpolation
        # depth1 = zoom(depth1, zoom=scale_factor[0], order=2)  # depth shape [H,W]
        # depth2 = zoom(depth2, zoom=scale_factor[0], order=2)  # depth shape [H,W]

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

        ##### the edge depth might be wrong !!! Remove them !
        if False:
            fg_mask_file = os.path.join(self.img_dir, img_name1).replace('color', 'mask').replace('.jpg', '.png')
            fg_mask = ((imageio.imread(fg_mask_file)[..., :3].sum(axis=-1) > 0) * 255).astype(np.uint8)
            edge_mask = extract_mask_edge(fg_mask)
            mask = cv2.bitwise_and(~edge_mask, (mask*255).astype(np.uint8)) > 0

        if len(cached_flow_pred_files) > 0 and self.args.use_error_map:
            cached_flow_pred_file = cached_flow_pred_files[id1]
            assert img_name1 + '_' in cached_flow_pred_file
            sup_flow_file = os.path.join(self.flow_dir, os.path.basename(cached_flow_pred_file))
            pred_flow = np.load(cached_flow_pred_file)
            sup_flow = np.load(sup_flow_file)
            error_map = np.linalg.norm(pred_flow - sup_flow, axis=-1)
            error_map = cv2.GaussianBlur(error_map, (5, 5), 0)
            if False:
                ###### save error map for visualization
                save_path = cached_flow_pred_file.replace('npy', 'png')
                imageio.imwrite(save_path, (error_map/error_map.max()*255).astype(np.uint8))
            error_selected = error_map[mask]
            prob = error_selected / np.sum(error_selected)
            select_ids_error = np.random.choice(mask.sum(), self.num_pts, replace=(mask.sum() < self.num_pts), p=prob)
            select_ids_random = np.random.choice(mask.sum(), self.num_pts, replace=(mask.sum() < self.num_pts))
            select_ids = np.random.choice(np.concatenate([select_ids_error, select_ids_random]), self.num_pts,
                                          replace=False)
        else:
            if self.args.use_count_map:
                count_map = imageio.imread(os.path.join(self.seq_dir, 'count_maps', img_name1.replace('.jpg', '.png')))
                pixel_sample_weight = 1 / np.sqrt(count_map + 1.)
                ##### downscale count map
                # pixel_sample_weight = zoom(pixel_sample_weight, zoom=scale_factor[0], order=2)
                pixel_sample_weight = pixel_sample_weight[mask]
                pixel_sample_weight /= pixel_sample_weight.sum()
                select_ids = np.random.choice(mask.sum(), self.num_pts, replace=(mask.sum() < self.num_pts),
                                              p=pixel_sample_weight)
            else:
                select_ids = np.random.choice(mask.sum(), self.num_pts, replace=(mask.sum() < self.num_pts))

        
        pair_weight = np.cos((frame_interval - 1.) / max_interval * np.pi / 2)

        coord1 = self.grid
        coord2 = self.grid + flow

        pts1 = torch.from_numpy(coord1[mask][select_ids]).float()
        pts2 = torch.from_numpy(coord2[mask][select_ids]).float()
        pts2_normed = normalize_coords(pts2, self.h, self.w)[None, None]
        # pts1_depth = torch.from_numpy(depth1[mask][select_ids]).float()
        # pts2_depth = torch.from_numpy(depth2[mask][select_ids]).float()
        # pts2_depth = F.grid_sample(torch.from_numpy(depth2).float()[None,None], pts2_normed, align_corners=True).squeeze().T
        
        ######### use the whole image instead of the selected points
        covisible_mask = torch.from_numpy(cycle_consistency_mask[mask][select_ids]).float()[..., None]
        weights = torch.ones_like(covisible_mask) * pair_weight
        # weights[covisible_mask == 0.] = 0

        if invalid:
            weights = torch.zeros_like(weights)

        gt_rgb1 = torch.from_numpy(img1[mask][select_ids]).float()
        # gt_rgb2 = F.grid_sample(torch.from_numpy(img2).float().permute(2, 0, 1)[None], pts2_normed,
                                # align_corners=True).squeeze().T

        flow_normed = flow / np.array([self.w, self.h])[None,None] * 2  # range in [-1, 1]
        data = {'ids1': id1,
                'ids2': id2,
                'pts1': pts1,  # [n_pts, 2]
                'pts2': pts2,  # [n_pts, 2]
                # 'pts1_depth': pts1_depth,  # [n_pts]
                # 'pts2_depth': pts2_depth,  # [n_pts]
                'gt_rgb1': gt_rgb1,  # [n_pts, 3]
                # 'gt_rgb2': gt_rgb2,
                'pts1_all': torch.from_numpy(coord1).float(), # [H, W, 2]
                'pts2_all': torch.from_numpy(coord2).float(), # [H, W, 2]
                'gt_img1' : torch.from_numpy(img1).float(), # [H, W, 3]
                'gt_img2' : torch.from_numpy(img2).float(), # [H, W, 3]
                'covisible_mask': covisible_mask,  # [H, W]
                # 'depth1_all' : torch.from_numpy(depth1).float(), # [H, W]
                # 'depth2_all' : torch.from_numpy(depth2).float(), # [H, W]
                'weights': weights,  # [n_pts, 1]
                # 'gt_flow': torch.from_numpy(flow).float(), # [H, W, 2]
                # 'gt_flow_normed' : torch.from_numpy(flow_normed).float(), # [H, W, 2]
                # 'select_mask' : torch.from_numpy(mask).bool(), # [H, W]
                # 'select_ids' : torch.from_numpy(select_ids).long(), # [num_pts]
                }
        return data