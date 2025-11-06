#
# Copyright (C) 2024, TRASE
# Technical University of Munich CVG
# All rights reserved.
#
# TRASE is heavily based on other research. Consider citing their works as well.
# 3D Gaussian Splatting: https://github.com/graphdeco-inria/gaussian-splatting
# Deformable-3D-Gaussians: https://github.com/ingra14m/Deformable-3D-Gaussians
# gaussian-grouping: https://github.com/lkeab/gaussian-grouping
# SAGA: https://github.com/Jumpat/SegAnyGAussians
# SC-GS: https://github.com/yihua7/SC-GS
# 4d-gaussian-splatting: https://github.com/fudan-zvg/4d-gaussian-splatting
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting
# GRAPHDECO research group, https://team.inria.fr/graphdeco
#


import os
import cv2
import torch
from tqdm import tqdm
from argparse import ArgumentParser
from sklearn.decomposition import PCA
from segment_anything import (SamAutomaticMaskGenerator, SamPredictor,
                              sam_model_registry)
import numpy as np
from glob import glob
from bitarray import bitarray

if __name__ == '__main__':
    
    parser = ArgumentParser(description="Extract SAM features and masks")
    
    parser.add_argument("--img_path", default='./data/HyperNeRF/split-cookie/rgb/1x', type=str)
    parser.add_argument("--output", default='./data/HyperNeRF/split-cookie', type=str)
    parser.add_argument("--sam_checkpoint_path", default="./dependency/sam_vit_h_4b8939.pth", type=str)
    parser.add_argument("--sam_arch", default="vit_h", type=str)
    parser.add_argument("--iou_th", default=0.88, type=float)
    parser.add_argument("--stability_score_th", default=0.95, type=float)
    parser.add_argument("--downsample_mask", default=1, type=int)
    parser.add_argument("--save_to_tensor", action='store_true', default=False)

    args = parser.parse_args()
    
    print("Initializing SAM...")
    model_type = args.sam_arch
    sam = sam_model_registry[model_type](checkpoint=args.sam_checkpoint_path).to('cuda')
    predictor = SamPredictor(sam)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=args.iou_th, 
        box_nms_thresh=0.7,
        stability_score_thresh=args.stability_score_th, 
        crop_n_layers=0,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=100,
    )
    
    output_masks = os.path.join(args.output, "masks")
    
    os.makedirs(output_masks, exist_ok=True)
    downsample_manually = False
    if args.downsample_mask != 1:
        downsample_manually = True
        
    print("Extracting features and mask...")
    
    for path in tqdm(sorted(os.listdir(args.img_path))):
        try:
            name = path.split('.')[0]
            img = cv2.imread(os.path.join(args.img_path, path))
            if downsample_manually:
                img = cv2.resize(img,dsize=(img.shape[1] // args.downsample_mask, img.shape[0] // args.downsample_mask),fx=1,fy=1,interpolation=cv2.INTER_LINEAR)
            ## origina method
            masks = mask_generator.generate(img)
            mask_list = []
            for m in masks:
                m_score = torch.from_numpy(m['segmentation']).float().to('cuda')
                if len(m_score.unique()) < 2:
                    continue
                else:
                    mask_list.append(m_score.bool())
            masks = torch.stack(mask_list, dim=0)
            if args.save_to_tensor:
                ## Save binary masks to tensor (this could use a lot of space)
                torch.save(masks, os.path.join(output_masks, name + '.pt'))
            else:
                ## Save binary masks as bitarray
                N, H, W = masks.shape
                masks = {
                    'masks': bitarray(masks.reshape(-1).cpu().numpy().tolist()),
                    'N': N, 
                    'H': H, 
                    'W': W, 
                }
                torch.save(masks, os.path.join(output_masks, name + '.pt'))
        except AttributeError:
            print(f"{path} is not an image")
        except Exception as X:
            print(X)
        
        