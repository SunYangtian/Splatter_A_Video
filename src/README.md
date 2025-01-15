## Environment
```
conda create -n splatter_a_video python=3.10
conda activate splatter_a_video
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install configargparse tensorboardX tensorboard imageio opencv-python matplotlib tqdm scipy pytorch_msssim jaxtyping plyfile omegaconf tabulate rich kornia
```

```
cd submodules/simple-knn
python setup.py install

cd submodules/dptr
python setup.py install
```

Install [pytorch3d](https://github.com/facebookresearch/pytorch3d).


## Data preparation
Please follow this [instruction]((src/data_preparation/README.md)).


## Training
python train.py --config configs/config.txt --seq_name $seq_name --num_imgs 250