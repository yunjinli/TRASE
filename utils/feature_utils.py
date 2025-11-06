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

import torch 

@torch.no_grad()
def get_sample_pixel_and_mask(sam_masks, num_sampled_pixels, num_sampled_masks):
    mask_sample_rate = num_sampled_masks / (sam_masks.shape[0])
    sampled_mask = torch.rand(sam_masks.shape[0]).cuda() < mask_sample_rate
    pixel_sample_rate = num_sampled_pixels / (sam_masks.shape[-1] * sam_masks.shape[-2])
    sampled_pixel = torch.rand(sam_masks.shape[-2], sam_masks.shape[-1]).cuda() < pixel_sample_rate
    non_mask_region = sam_masks.sum(dim = 0) == 0
    sampled_pixel = torch.logical_and(sampled_pixel, ~non_mask_region)
    
    return sampled_pixel, sampled_mask

@torch.no_grad()
def get_pixel_weights(sam_masks, sampled_pixel):
    per_pixel_mask_size = sam_masks * sam_masks.sum(-1).sum(-1)[:, None, None]
    per_pixel_mean_mask_size = per_pixel_mask_size.sum(dim = 0) / (sam_masks.sum(dim = 0) + 1e-9)
    per_pixel_mean_mask_size = per_pixel_mean_mask_size[sampled_pixel]
    pixel_to_pixel_mask_size = per_pixel_mean_mask_size.unsqueeze(0) * per_pixel_mean_mask_size.unsqueeze(1)
    ptp_max_size = pixel_to_pixel_mask_size.max()
    pixel_to_pixel_mask_size[pixel_to_pixel_mask_size == 0] = 1e10
    per_pixel_weight = torch.clamp(ptp_max_size / pixel_to_pixel_mask_size, 1.0, None)
    per_pixel_weight = (per_pixel_weight - per_pixel_weight.min()) / (per_pixel_weight.max() - per_pixel_weight.min()) * 9. + 1.
    return per_pixel_weight

@torch.no_grad()
def get_pixel_mask_correspondence_matrix(sam_masks, sampled_pixel, sampled_mask):
    sam_masks_sampled_pixel = sam_masks[:, sampled_pixel]
    ## Calculate the pixel-mask correspondence vector based on SAM masks
    pixel_mask_correspondence_vector = sam_masks_sampled_pixel[sampled_mask, :]
    mask_corr_matrix = torch.einsum('nh,nj->hj', pixel_mask_correspondence_vector.float(), pixel_mask_correspondence_vector.float())
    mask_corr_matrix[mask_corr_matrix != 0] = 1
                
    return mask_corr_matrix

def get_features_correspondence_matrix(rendered_features, sampled_pixel):
    sampled_rendered_features = rendered_features[:, sampled_pixel]
    sampled_rendered_features = sampled_rendered_features.permute([1, 0])
    sampled_rendered_features = torch.nn.functional.normalize(sampled_rendered_features, dim=-1, p=2)
    feature_corr_matrix = torch.einsum('hc,jc->hj', sampled_rendered_features, sampled_rendered_features)
    
    return feature_corr_matrix