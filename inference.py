# -*- coding: utf-8 -*-
# @Time    : 2020-02-26 17:53
# @Author  : Zonas
# @Email   : zonas.wang@gmail.com
# @File    : inference.py
"""

"""
import argparse
import logging
import os
import os.path as osp
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from unet import NestedUNet
from unet import UNet
from utils.dataset import BasicDataset
from config import UNetConfig

cfg = UNetConfig()

def inference_one(net,
                  image,
                  device,
                  scale_factor=1,
                  out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess(image, scale_factor))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
        if cfg.deepsupervision:
            output = output[-1]

        if cfg.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(image.size[1]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        mask = probs.squeeze().cpu().numpy()

    return mask > out_threshold


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', dest='input', type=str, default='',
                        help='Directory of input images')
    parser.add_argument('--output', '-o', dest='output', type=str, default='',
                        help='Directory of ouput images')
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)

    return parser.parse_args()


def mask_to_image(mask, idx):
    mask = mask[idx]
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    args = get_args()

    input_imgs = os.listdir(args.input)

    if cfg.deepsupervision:
        net = eval(cfg.model)(cfg)
    else:
        net = eval(cfg.model)(cfg)

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    for i, img_name in tqdm(enumerate(input_imgs)):
        logging.info("\nPredicting image {} ...".format(img_name))

        img_path = osp.join(args.input, img_name)
        img = Image.open(img_path)

        mask = inference_one(net=net,
                             image=img,
                             scale_factor=cfg.scale,
                             out_threshold=args.mask_threshold,
                             device=device)
        img_name_no_ext = osp.splitext(img_name)[0]
        output_img_dir = osp.join(args.output, img_name_no_ext)
        os.makedirs(output_img_dir, exist_ok=True)

        if cfg.n_classes == 1:
            image_idx = Image.fromarray((mask * 255).astype(np.uint8))
            image_idx.save(osp.join(output_img_dir, img_name))
        else:
            ids = mask.shape[0]
            for idx in range(ids):
                img_name_idx = img_name_no_ext + "_" + str(idx) + ".png"
                image_idx = mask_to_image(mask, idx)
                image_idx.save(osp.join(output_img_dir, img_name_idx))

