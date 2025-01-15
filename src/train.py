import os
import subprocess
import random
import datetime
import shutil
import numpy as np
import torch
import torch.utils.data
import torch.distributed as dist
from config import config_parser
from tensorboardX import SummaryWriter
from loaders.create_training_dataset import get_training_dataset
from trainer_fragGS import FragTrainer
torch.manual_seed(1234)
# from gui import GUI
# import dearpygui.dearpygui as dpg


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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


def train(args):
    # seq_name = os.path.basename(args.data_dir.rstrip('/'))
    seq_name = args.seq_name
    out_dir = os.path.join(args.save_dir, '{}_{}'.format(args.expname, seq_name))
    os.makedirs(out_dir, exist_ok=True)
    print('optimizing for {}...\n output is saved in {}'.format(seq_name, out_dir))

    args.out_dir = out_dir

    # save the args and config files
    f = os.path.join(out_dir, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            if not arg.startswith('_'):
                attr = getattr(args, arg)
                file.write('{} = {}\n'.format(arg, attr))

    if args.config:
        f = os.path.join(out_dir, 'config.txt')
        if not os.path.isfile(f):
            shutil.copy(args.config, f)

    log_dir = 'logs/{}_{}'.format(args.expname, seq_name)
    writer = SummaryWriter(log_dir)

    g = torch.Generator()
    g.manual_seed(args.loader_seed)
    dataset, data_sampler = get_training_dataset(args, max_interval=args.start_interval)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=args.num_pairs,
                                              worker_init_fn=seed_worker,
                                              generator=g,
                                              num_workers=args.num_workers,
                                              sampler=data_sampler,
                                              shuffle=True if data_sampler is None else False,
                                              pin_memory=True)

    # get trainer
    trainer = FragTrainer(args)
    # trainer = BaseTrainer(args)

    if False:
        # trainer.draw_gs_trajectory(samp_num=10, gs_num=512)
        use_mask = True
        track_imgs = trainer.draw_pixel_trajectory(use_mask=use_mask, radius=4)
        save_dir = os.path.join(trainer.out_dir, 'tracking')
        os.makedirs(save_dir, exist_ok=True)
        import imageio
        w = track_imgs[0].shape[1]
        track_imgs = [x[:,w//2:] for x in track_imgs]
        save_name = "tracking.mp4" if use_mask else "tracking_no_mask.mp4"
        imageio.mimwrite(os.path.join(save_dir, save_name), track_imgs, fps=15)
        print()

    if False:
        trainer.render_video(save_frames=True)

    if False:
        trainer.render_part(fg=True, threshold=0.5)
        print()

    if False:
        ### for cow
        delta_pos = torch.tensor([[0.6, -0.4, 0.1]], device='cuda')
        trainer.add_fg(delta_pos, scale=0.8, threshold=0.9)
        # delta_pos = torch.tensor([[-0.4, 0.1, -0.6]], device='cuda')
        # trainer.add_fg(delta_pos, scale=1.2, threshold=0.9)
        ### for camel
        # delta_pos = torch.tensor([[0.4, 0.3, -0.2]], device='cuda')
        # trainer.add_fg(delta_pos, scale=0.6, threshold=0.9)

        print()

    if False:
        trainer.get_interpolation_result(scaling=4)
    
    if False:
        # trainer.optimize_appearance_from_img("000000_edit_mask.png")
        mask_path = os.path.join(args.data_dir, "masks", "00000.png")
        # edited_img_path = os.path.join(args.data_dir, "sketch_1.png")
        edited_img_path = os.path.join("nips_cow.png")
        trainer.optimize_appearance_from_mask(mask_path, edited_img_path)

    ##### This code is for canonical space visualization
    #### track-everything's canonical space
    if False:
        trainer.save_canonical_rgba_volume(num_pts=5000000, sample_points_from_frames=True)
    if False:
        pts_canonical_np, colors_np, mask_np = trainer.save_canonical_points(start_id=0, end_id=dataset.num_imgs, step=10)
        if False:
            masks = [extract_mask_edge((m*255).astype(np.uint8), kernel_size=3) == 0 for m in mask_np]
            points = [p[m.reshape(-1)] for p, m in zip(pts_canonical_np, masks)]
            colors = [c[m] for c, m in zip(colors_np, masks)]
            points = np.concatenate(points, axis=0)
            colors = np.concatenate(colors, axis=0)
        # print()
        import trimesh
        trimesh.PointCloud(pts_canonical_np.reshape(-1,3), colors=colors_np.reshape(-1,3)).export("./debug_all.ply")

    ###### This part is for NVS
    if False:
        trainer.get_nvs_rendered_imgs()
        trainer.get_stereo_rendered_imgs()

    ###### config gui for visualization
    from dataclasses import dataclass
    @dataclass
    class GUIArgs:
        gui: bool = args.gui
        W: int = 512
        H: int = 512
        radius: float = 2
        fovy: float = 60

    start_step = trainer.step + 1
    step = start_step
    epoch = 0
    if args.gui:
        gui_args = GUIArgs()
        gui = GUI(gui_args, [])
        gui.gaussian_trainer = trainer
        while dpg.is_dearpygui_running():
            for batch in data_loader:
                gui.test_step()
                dpg.render_dearpygui_frame()
                if gui.training and step < args.num_iters + start_step + 1:
                    trainer.train_one_step(step, batch)
                    trainer.log(writer, step)

                    step += 1

                    dataset.set_max_interval(args.start_interval + step // 2000)

                    if step >= args.num_iters + start_step + 1:
                        break

                epoch += 1
                if args.distributed:
                    data_sampler.set_epoch(epoch)
    else:
        while step < args.num_iters + start_step + 1:
            for batch in data_loader:
                trainer.train_one_step(step, batch)
                trainer.log(writer, step)

                step += 1

                dataset.set_max_interval(args.start_interval + step // 2000)

                if step >= args.num_iters + start_step + 1:
                    break


if __name__ == '__main__':
    args = config_parser()

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    train(args)

