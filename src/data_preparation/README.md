Our approach relies on the below works for data preprocessing.
1. Monocular depth estimation [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) and metric depth estimation [Unidepth](https://github.com/lpiccinelli-eth/UniDepth). The metric depth is used to initialize the 3D point flow, together with flow.

    - Change the $UNIDEPTH_PATH and $UNIDEPTH_CKPT_PATH in `compute_metric_depth.py`.

    - Change the $depth-anything-v2 path in `compute_depth.py`

```
python compute_metric_depth.py --img_dir $data_root$/images --depth_dir $data_root$/unidepth_disp --intrins-file $data_root$/unidepth_intrins.json

python compute_depth.py --img_dir $data_root$/images --out_raw_dir $data_root$/depth_anything_v2 --out_aligned_dir $data_root$/aligned_depth_anything_v2 --model depth-anything-v2 --metric_dir $data_root$/unidepth_disp
```

[Marigold](https://github.com/prs-eth/Marigold) is also requried in current implementation when training.


2. Flow estimation [TAPIR](https://github.com/google-deepmind/tapnet).
    - Download [pretrained model](https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt)
    - Extract dense frame correspondence
    ```
    python compute_tracks_torch.py --image_dir $data_root$/images --mask_dir $data_root$/masks --out_dir $data_root$/bootstapir --model_type bootstapir --ckpt_dir {pretrained_model_dir}

    ```

The initialized point flow should look like
<table>
  <tr>
    <td><img src="../../imgs/video.gif" alt="video" width="300"></td>
    <td><img src="../../imgs/point_render.gif" alt="point" width="300"></td>
  </tr>
</table>


### Custom Video

The video should be extracted to frames first. Then object masks are extracted.


The dataset is prepared in the following format
```
- data_root
    - images
        - 0000.png
        - ...
    - masks
        - 0000.png
        - ...
    - aligned_depth_anything_v2
        - 0000.npy
    - marigold
        - depth_npy
            - 0000_pred.npy
            - ...
    - bootstapir
        - 0000_0000.npy
        - 0000_0001.npy
        - ...
    - unidepth_disp (not used in training)
    - depth_anything_v2 (not used in training)
```

### DAVIS dataset

We will also provide scripts for data processing of DAVID video.