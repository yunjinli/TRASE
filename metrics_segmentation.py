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


from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
import numpy as np

def compute_acc(pred, gt):
    same = np.sum(pred == gt)
    
    num_pixels = gt.size

    return same / num_pixels

def compute_iou(pred, gt):
    intersection = np.sum(np.logical_and(pred, gt))
    union = np.sum(np.logical_or(pred, gt))

    if (union == 0):
        iou = 0
    else:
        iou = intersection / union
    return iou

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(gt_dir):
        try:
            render = Image.open(renders_dir / fname)
            gt = Image.open(gt_dir / fname)
            renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :])
            gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :])
            image_names.append(fname)
        except Exception as X:
            print(X)
    return renders, gts, image_names

def readMasks(pred_dir, gt_dir):
    preds = []
    gts = []
    image_names = []
    
    for fname in os.listdir(gt_dir):
        try:
            pred_mask = np.asarray(Image.open(pred_dir / fname))
            pred_mask_copy = pred_mask.copy()
            pred = (pred_mask_copy.mean(axis=-1) / 255).astype(bool)
            
            gt = np.asarray(Image.open(gt_dir / fname))
            
            preds.append(pred)
            gts.append(gt)
            image_names.append(fname)
        except Exception as X:
            print(X)
    return preds, gts, image_names

def evaluate(model_paths, no_psnr, benchmark_path):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")
    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            print("Benchmark:", benchmark_path)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"
            benchmark_dir = Path(benchmark_path)
            
            for method in os.listdir(test_dir):
                try:
                    print("Method:", method)

                    full_dict[scene_dir][method] = {}
                    per_view_dict[scene_dir][method] = {}
                    full_dict_polytopeonly[scene_dir][method] = {}
                    per_view_dict_polytopeonly[scene_dir][method] = {}

                    method_dir = test_dir / method
                    
                    gt_masks_dir = benchmark_dir / "gt_masks"
                    pred_masks_dir = method_dir / "pred_masks"
                    
                    pred_masks, gt_masks, image_names = readMasks(pred_masks_dir, gt_masks_dir)
                    accs = []
                    ious = []
                    
                    for idx in tqdm(range(len(pred_masks)), desc="Metric evaluation progress"):
                        pred_mask = pred_masks[idx]
                        gt_mask = gt_masks[idx]
                        accs.append(compute_acc(pred_mask, gt_mask))
                        ious.append(compute_iou(pred_mask, gt_mask))
                    print("  mIOU : {:>12.4f}".format(torch.tensor(ious).mean(), ".5"))
                    print("  mACC : {:>12.4f}".format(torch.tensor(accs).mean(), ".5"))
                    print("")
                    
                    if not no_psnr:
                        gt_dir = benchmark_dir / "gt_masks_object"
                        renders_dir = method_dir / "segment_objects"
                        renders, gts, image_names = readImages(renders_dir, gt_dir)

                        ssims = []
                        psnrs = []
                        lpipss = []

                        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                            renders[idx] = renders[idx].cuda()
                            gts[idx] = gts[idx].cuda()
                            ssims.append(ssim(renders[idx], gts[idx]))
                            psnrs.append(psnr(renders[idx], gts[idx]))
                            lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))
                            renders[idx] = renders[idx].cpu()
                            gts[idx] = gts[idx].cpu()
                        print("  SSIM : {:>12.4f}".format(torch.tensor(ssims).mean(), ".5"))
                        print("  PSNR : {:>12.4f}".format(torch.tensor(psnrs).mean(), ".5"))
                        print("  LPIPS: {:>12.4f}".format(torch.tensor(lpipss).mean(), ".5"))
                        print("")
                        
                        
                        full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                                "PSNR": torch.tensor(psnrs).mean().item(),
                                                                "LPIPS": torch.tensor(lpipss).mean().item(),
                                                                "mIOU": torch.tensor(ious).mean().item(),
                                                                "mACC": torch.tensor(accs).mean().item(),
                                                                }
                                                            )
                        per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                                    "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                                    "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                                    "IOU": {name: lp for lp, name in zip(torch.tensor(ious).tolist(), image_names)},
                                                                    "ACC": {name: lp for lp, name in zip(torch.tensor(accs).tolist(), image_names)},
                                                                    })
                    else:
                        full_dict[scene_dir][method].update({
                                                                "mIOU": torch.tensor(ious).mean().item(),
                                                                "mACC": torch.tensor(accs).mean().item(),
                                                                }
                                                            )
                        per_view_dict[scene_dir][method].update({
                                                                    "IOU": {name: lp for lp, name in zip(torch.tensor(ious).tolist(), image_names)},
                                                                    "ACC": {name: lp for lp, name in zip(torch.tensor(accs).tolist(), image_names)},
                                                                    })
                except Exception as X:
                    print(X)
                    print("Unable to compute metrics for", method)

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except:
            print("Unable to compute metrics for model", scene_dir)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--no_psnr', action="store_true")
    parser.add_argument('--benchmark_path', type=str)
    

    args = parser.parse_args()
    evaluate(args.model_paths, args.no_psnr, args.benchmark_path)
