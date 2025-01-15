import argparse
import fnmatch
import os
import os.path as osp
from glob import glob
from typing import Literal

import cv2
import imageio.v2 as iio
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Pipeline, pipeline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
UINT16_MAX = 65535


models = {
    # "depth-anything": "LiheYoung/depth-anything-large-hf",
    # "depth-anything-v2": "depth-anything/Depth-Anything-V2-Large-hf",
    "depth-anything": "/mnt/sdb/syt/pretrained_weights/depth-anything-large-hf",
    "depth-anything-v2": "/mnt/sdb/syt/pretrained_weights/Depth-Anything-V2-Large-hf",
}


def get_pipeline(model_name: str):
    pipe = pipeline(task="depth-estimation", model=models[model_name], device=DEVICE)
    print(f"{model_name} model loaded.")
    return pipe


def to_uint16(disp: np.ndarray):
    disp_min = disp.min()
    disp_max = disp.max()

    if disp_max - disp_min > np.finfo("float").eps:
        disp_uint16 = UINT16_MAX * (disp - disp_min) / (disp_max - disp_min)
    else:
        disp_uint16 = np.zeros(disp.shape, dtype=disp.dtype)
    disp_uint16 = disp_uint16.astype(np.uint16)
    return disp_uint16


def get_depth_anything_disp(
    pipe: Pipeline,
    img_file: str,
    ret_type: Literal["uint16", "float"] = "float",
):

    image = Image.open(img_file)
    disp = pipe(image)["predicted_depth"]
    disp = torch.nn.functional.interpolate(
        disp.unsqueeze(1), size=image.size[::-1], mode="bicubic", align_corners=False
    )
    disp = disp.squeeze().cpu().numpy()
    if ret_type == "uint16":
        return to_uint16(disp)
    elif ret_type == "float":
        return disp
    else:
        raise ValueError(f"Unknown return type {ret_type}")


def save_disp_from_dir(
    model_name: str,
    img_dir: str,
    out_dir: str,
    matching_pattern: str = "*",
):
    img_files = sorted(glob(osp.join(img_dir, "*.jpg"))) + sorted(
        glob(osp.join(img_dir, "*.png"))
    )
    img_files = [
        f for f in img_files if fnmatch.fnmatch(osp.basename(f), matching_pattern)
    ]
    if osp.exists(out_dir) and len(glob(osp.join(out_dir, "*.png"))) == len(img_files):
        print(f"Raw {model_name} depth maps already computed for {img_dir}")
        return

    pipe = get_pipeline(model_name)
    os.makedirs(out_dir, exist_ok=True)
    for img_file in tqdm(img_files, f"computing {model_name} depth maps"):
        disp = get_depth_anything_disp(pipe, img_file, ret_type="uint16")
        out_file = osp.join(out_dir, osp.splitext(osp.basename(img_file))[0] + ".png")
        iio.imwrite(out_file, disp)


def align_monodepth_with_metric_depth(
    metric_depth_dir: str,
    input_monodepth_dir: str,
    output_monodepth_dir: str,
    matching_pattern: str = "*",
):
    print(
        f"Aligning monodepth in {input_monodepth_dir} with metric depth in {metric_depth_dir}"
    )
    mono_paths = sorted(glob(f"{input_monodepth_dir}/{matching_pattern}"))
    img_files = [osp.basename(p) for p in mono_paths]
    os.makedirs(output_monodepth_dir, exist_ok=True)
    if len(os.listdir(output_monodepth_dir)) == len(img_files):
        print(f"Founds {len(img_files)} files in {output_monodepth_dir}, skipping")
        return

    for f in tqdm(img_files):
        imname = os.path.splitext(f)[0]
        metric_path = osp.join(metric_depth_dir, imname + ".npy")
        mono_path = osp.join(input_monodepth_dir, imname + ".png")

        mono_disp_map = iio.imread(mono_path) / UINT16_MAX
        metric_disp_map = np.load(metric_path)
        ms_colmap_disp = metric_disp_map - np.median(metric_disp_map) + 1e-8
        ms_mono_disp = mono_disp_map - np.median(mono_disp_map) + 1e-8

        scale = np.median(ms_colmap_disp / ms_mono_disp)
        shift = np.median(metric_disp_map - scale * mono_disp_map)

        aligned_disp = scale * mono_disp_map + shift

        min_thre = min(1e-6, np.quantile(aligned_disp, 0.01))
        # set depth values that are too small to invalid (0)
        aligned_disp[aligned_disp < min_thre] = 0.0
        out_file = osp.join(output_monodepth_dir, imname + ".npy")
        np.save(out_file, aligned_disp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="depth-anything",
        help="depth model to use, one of [depth-anything, depth-anything-v2]",
    )
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--out_raw_dir", type=str, required=True)
    parser.add_argument("--out_aligned_dir", type=str, default=None)
    parser.add_argument("--sparse_dir", type=str, default=None)
    parser.add_argument("--metric_dir", type=str, default=None)
    parser.add_argument("--matching_pattern", type=str, default="*")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    assert args.model in [
        "depth-anything",
        "depth-anything-v2",
    ], f"Unknown model {args.model}"
    save_disp_from_dir(
        args.model, args.img_dir, args.out_raw_dir, args.matching_pattern
    )

    align_monodepth_with_metric_depth(
        args.metric_dir,
        args.out_raw_dir,
        args.out_aligned_dir,
        args.matching_pattern,
    )


if __name__ == "__main__":
    main()
