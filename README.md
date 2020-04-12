# Unet and Unet++: multiple classification using Pytorch

This repository contains code for a multiple classification image segmentation model based on [UNet](https://arxiv.org/pdf/1505.04597.pdf) and [UNet++](https://arxiv.org/abs/1807.10165)


## Usage

#### Note : Use Python 3

### Dataset
make sure to put the files as the following structure:
```
data
├── images
|   ├── 0a7e06.jpg
│   ├── 0aab0a.jpg
│   ├── 0b1761.jpg
│   ├── ...
|
└── masks
    ├── 0a7e06.png
    ├── 0aab0a.png
    ├── 0b1761.png
    ├── ...
```
mask is a single-channel category index. For example, your dataset has three categories, mask should be 8-bit images with value 0,1,2 as the categorical value, this image looks black.

### Demo dataset
You can download the demo dataset from [here](https://drive.google.com/open?id=13vwNHeIVLPEsMevd0M9kLreBrAd257c0) to data/

### Training
```bash
python train.py
```

### inference
```base
python inference.py -m ./data/checkpoints/epoch_10.pth -i ./data/test/input -o ./data/test/output
```
If you want to highlight your mask with color, you can
```bash
python inference_color.py -m ./data/checkpoints/epoch_10.pth -i ./data/test/input -o ./data/test/output
```

## Tensorboard
You can visualize in real time the train and val losses, along with the model predictions with tensorboard:
```bash
tensorboard --logdir=runs
```

