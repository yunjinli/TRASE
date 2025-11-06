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


import torch
from scene import Scene, DeformModel
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from gaussian_renderer import GaussianModel
import numpy as np
from PIL import Image
import colorsys
import cv2
from sklearn.decomposition import PCA
import imageio
import concurrent.futures
from utils.general_utils import PILtoTorch
import hdbscan
import multiprocessing
from ext.grounded_sam import grouned_sam_output, load_model_hf, select_obj_ioa
from segment_anything import sam_model_registry, SamPredictor
import math
import pytorch3d.ops as ops

def generate_grid_index(depth):
    h, w = depth.shape
    grid = torch.meshgrid([torch.arange(h), torch.arange(w)])
    grid = torch.stack(grid, dim=-1)
    return grid

def feature3d_to_rgb(x, n_components=3):      
    X_center = x - torch.mean(x, axis=0)   # Center data
    q ,r = torch.linalg.qr(X_center)
    U, s, Vt = torch.linalg.svd(r, full_matrices=False)
    x_compress = torch.matmul(U[:, :n_components],torch.diag(s[:n_components]))
    pca_result = torch.matmul(q, x_compress)
    pca_normalized = (pca_result - pca_result.min()) / (pca_result.max() - pca_result.min())
    return pca_normalized

def multithread_write(image_list, path):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)
    def write_image(image, count, path):
        try:
            torchvision.utils.save_image(image, os.path.join(path, '{0:05d}'.format(count) + ".png"))
            return count, True
        except Exception as error1:
            try:
                Image.fromarray(image).save(os.path.join(path, '{0:05d}'.format(count) + ".png"))
            except Exception as error2:
                print("torchvision.utils.save_image failed:", error1)
                print(" Image.fromarray(image).save failed:", error2)
                return count, False
        
    tasks = []
    for index, image in enumerate(image_list):
        tasks.append(executor.submit(write_image, image, index, path))
    executor.shutdown()
    for index, status in enumerate(tasks):
        if status == False:
            write_image(image_list[index], index, path)
    
def feature_to_rgb(x, n_components=3):
    H, W = x.shape[1], x.shape[2]
    x = x.view(x.shape[0], -1).T
        
    X_center = x - torch.mean(x, axis=0)   # Center data
    q ,r = torch.linalg.qr(X_center)
    U, s, Vt=torch.linalg.svd(r, full_matrices=False)
    x_compress = torch.matmul(U[:, :n_components],torch.diag(s[:n_components]))
    pca_result = torch.matmul(q, x_compress)
    
    pca_result = pca_result.reshape(H, W, 3).permute(2, 0, 1)
    pca_normalized = (pca_result - pca_result.min()) / (pca_result.max() - pca_result.min())
    return pca_normalized

def postprocessing(features, query_feature, score_threshold=0.8):
    features /= features.norm(dim=-1, keepdim=True)
    query_feature /= query_feature.norm(dim=-1, keepdim=True)
    query_feature = query_feature.unsqueeze(-1)
    scores = features.half() @ query_feature.half()
    scores = scores[:, 0]
    mask = (scores >= score_threshold)
    return mask

to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)
@torch.no_grad()
def render_set(model_path, is_6dof, name, iteration, views, gaussians, pipeline, background, deform, load2gpu_on_the_fly, model_type, load_image_on_the_fly, segment_ids, text_prompt, threshold, white_background, score_threshold, multithread_save):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    render_feats_path = os.path.join(model_path, name, "ours_{}".format(iteration), "rendered_feats")
    canonical_path = os.path.join(model_path, name, "ours_{}".format(iteration), "canonical")
    point_path = os.path.join(model_path, name, "ours_{}".format(iteration), "pointcloud")
    gaussian_clusters_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gaussian_clusters")
    seg_path = os.path.join(model_path, name, "ours_{}".format(iteration), "segmentation")
    gaussian_feats_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gaussian_feats")
    segment_objects_path = os.path.join(model_path, name, "ours_{}".format(iteration), "segment_objects")
    text_prompt_objects_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"text_prompt_{text_prompt}_objects")
    pred_masks_path = os.path.join(model_path, name, "ours_{}".format(iteration), "pred_masks")
        
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(render_feats_path, exist_ok=True)
    makedirs(canonical_path, exist_ok=True)
    makedirs(point_path, exist_ok=True)
    makedirs(gaussian_clusters_path, exist_ok=True)
    makedirs(seg_path, exist_ok=True)
    makedirs(gaussian_feats_path, exist_ok=True)
    makedirs(segment_objects_path, exist_ok=True)
    makedirs(pred_masks_path, exist_ok=True)
    makedirs(text_prompt_objects_path, exist_ok=True)
    
    canonical_list = []
    
    render_images = []
    render_list = []
    
    gt_images = []
    gt_list = []
    
    rendered_feats_images = []
    rendered_feats_list = []
    
    pointcloud_images = []
    pointcloud_list = []
    
    gaussian_clusters_images = []
    gaussian_clusters_list = []
    
    gaussian_feats_images = []
    gaussian_feats_list = []
    
    seg_images = []
    seg_list = []
    
    segment_objects_images = []
    segment_objects_list = []
    
    text_prompt_objects_images = []
    text_prompt_objects_list = []
    
    pred_masks_images = []
    pred_masks_list = []
    
    try:
        cluster_ids_x = gaussians.get_clusters['id'].squeeze()
        cluster_point_colors = gaussians.get_clusters['rgb']
    except:
        cluster_ids_x = None
        cluster_point_colors = None

    if text_prompt != '':
        print("Text prompt detected: ", text_prompt)
        ## Language
        ckpt_repo_id = "ShilongLiu/GroundingDINO"
        ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
        ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
        groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)
        sam_checkpoint = 'dependency/sam_vit_h_4b8939.pth'
        sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        sam.to(device='cuda')
        sam_predictor = SamPredictor(sam)
    
    gaussians_feature_pca = feature3d_to_rgb(gaussians.get_gaussian_features.squeeze(1))
    torch.save(gaussians.get_gaussian_features.squeeze(1), os.path.join(render_feats_path, "gaussian_feats3d.pt"))
    
    try:
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            ## Deformation
            if load2gpu_on_the_fly:
                view.load2device()
            fid = view.fid
            xyz = gaussians.get_xyz
            time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
            d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input) if model_type == 'DeformNetwork' else deform.step(xyz.detach(), time_input, gaussians.get_gaussian_features.squeeze(1))
                
            results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof)
            torch.save(results['render_gaussian_features'], os.path.join(render_feats_path, '{0:05d}'.format(idx) + ".pt"))
            rendering = results["render"]
            
            if idx == 0:
                if text_prompt != '':
                    text_mask, annotated_frame_with_mask = grouned_sam_output(groundingdino_model, sam_predictor, text_prompt, to8b(rendering).transpose(1,2,0))
                    del sam_predictor
                    del groundingdino_model
                    Image.fromarray(annotated_frame_with_mask).save(os.path.join(render_path[:-8],'grounded-sam---' + text_prompt + '.png'))
                    Image.fromarray(text_mask.detach().cpu().numpy()).save(os.path.join(render_path[:-8],'binary-grounded-sam---' + text_prompt + '.png'))
                    depth = results["depth"]
                    depth = depth.squeeze()

                    grid_index = generate_grid_index(depth).cuda()
                    
                    z = view.zfar / (view.zfar - view.znear) * depth[text_mask] - view.zfar * view.znear / (view.zfar - view.znear)

                    uvz = torch.cat(((((grid_index[text_mask, :][:, 1] - 0.5) / view.image_width * 2 - 1) * depth[text_mask]).unsqueeze(-1),
                                    (((grid_index[text_mask, :][:, 0] - 0.5) / view.image_height * 2 - 1) * depth[text_mask]).unsqueeze(-1),
                                    z.unsqueeze(-1),
                                    depth[text_mask].unsqueeze(-1)), 1)
                    
                    text_masked_points_in_3D = uvz @ (torch.inverse(view.full_proj_transform))[:, :3]
                    
                    knn_obj = ops.knn_points(
                        text_masked_points_in_3D.unsqueeze(0),
                        (xyz + d_xyz).detach().unsqueeze(0),
                        K=1,
                    )
                    ijs = knn_obj.idx.squeeze(0).squeeze(-1)
                    
                    text_masked_points_cls = cluster_ids_x[ijs].int()
                    
                    text_masked_cls_id = torch.where(torch.bincount(text_masked_points_cls) > threshold, 1, 0).nonzero()
                    print("Text prompt cls id: ", text_masked_cls_id)
                
            render_images.append(to8b(rendering).transpose(1,2,0))
            if multithread_save:
                render_list.append(rendering.cpu())
            else:
                torchvision.utils.save_image(rendering.cpu(), os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

            rendered_feats = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof, override_color=gaussians_feature_pca)['render']
            rendered_feats_images.append(to8b(rendered_feats).transpose(1,2,0))
            if multithread_save:
                rendered_feats_list.append(rendered_feats.cpu())
            else:
                torchvision.utils.save_image(rendered_feats.cpu(), os.path.join(render_feats_path, '{0:05d}'.format(idx) + ".png"))
            
            cur_pts = torch.cat([xyz + d_xyz, torch.ones_like(xyz[..., :1])], dim=-1).cuda()
            
            cur_pts2d = cur_pts @ view.full_proj_transform.cuda()
            cur_pts2d = cur_pts2d[..., :2] / cur_pts2d[..., -1:]
            cur_pts2d = (cur_pts2d + 1) / 2 * torch.tensor([view.image_width, view.image_height]).cuda()

            buffer_image = torch.zeros(size=(3, view.image_height, view.image_width)).cuda() if not white_background else torch.ones(size=(3, view.image_height, view.image_width)).cuda()

            mask_1 = (cur_pts2d[:, 0] > 0) & (cur_pts2d[:, 0] < view.image_width)
            mask_2 = (cur_pts2d[:, 1] > 0) & (cur_pts2d[:, 1] < view.image_height)
            final_mask = mask_1 & mask_2
            buffer_image[0, (cur_pts2d[final_mask, 1]).type(torch.long), (cur_pts2d[final_mask, 0]).type(torch.long)] = 1 if not white_background else 0
            buffer_image[1, (cur_pts2d[final_mask, 1]).type(torch.long), (cur_pts2d[final_mask, 0]).type(torch.long)] = 1 if not white_background else 0
            buffer_image[2, (cur_pts2d[final_mask, 1]).type(torch.long), (cur_pts2d[final_mask, 0]).type(torch.long)] = 1 if not white_background else 0
            
            if multithread_save:
                pointcloud_list.append(buffer_image.detach().cpu())
            else:
                torchvision.utils.save_image(buffer_image.detach().cpu(), os.path.join(point_path, '{0:05d}'.format(idx) + ".png"))
                
            pointcloud_images.append(to8b(buffer_image).transpose(1,2,0))
            
            buffer_image = torch.zeros(size=(3, view.image_height, view.image_width)).cuda() if not white_background else torch.ones(size=(3, view.image_height, view.image_width)).cuda()
            try:
                buffer_image[0, (cur_pts2d[final_mask, 1]).type(torch.long), (cur_pts2d[final_mask, 0]).type(torch.long)] = cluster_point_colors[final_mask, 0]
                buffer_image[1, (cur_pts2d[final_mask, 1]).type(torch.long), (cur_pts2d[final_mask, 0]).type(torch.long)] = cluster_point_colors[final_mask, 1]
                buffer_image[2, (cur_pts2d[final_mask, 1]).type(torch.long), (cur_pts2d[final_mask, 0]).type(torch.long)] = cluster_point_colors[final_mask, 2]
                
                if multithread_save:
                    gaussian_clusters_list.append(buffer_image.detach().cpu())
                else:
                    torchvision.utils.save_image(buffer_image.detach().cpu(), os.path.join(gaussian_clusters_path, '{0:05d}'.format(idx) + ".png"))
                gaussian_clusters_images.append(to8b(buffer_image).transpose(1,2,0))
            except:
                print("[Warning] No clusters found...Gaussian clusters not rendered...")
            
            buffer_image = torch.zeros(size=(3, view.image_height, view.image_width)).cuda() if not white_background else torch.ones(size=(3, view.image_height, view.image_width)).cuda()
            
            buffer_image[0, (cur_pts2d[final_mask, 1]).type(torch.long), (cur_pts2d[final_mask, 0]).type(torch.long)] = gaussians_feature_pca[final_mask, 0]
            buffer_image[1, (cur_pts2d[final_mask, 1]).type(torch.long), (cur_pts2d[final_mask, 0]).type(torch.long)] = gaussians_feature_pca[final_mask, 1]
            buffer_image[2, (cur_pts2d[final_mask, 1]).type(torch.long), (cur_pts2d[final_mask, 0]).type(torch.long)] = gaussians_feature_pca[final_mask, 2]

            if multithread_save:
                gaussian_feats_list.append(buffer_image.detach().cpu())
            else:
                torchvision.utils.save_image(buffer_image.detach().cpu(), os.path.join(gaussian_feats_path, '{0:05d}'.format(idx) + ".png"))
                
            gaussian_feats_images.append(to8b(buffer_image).transpose(1,2,0))
            
            segmentation_mask = render(viewpoint_camera=view, pc=gaussians, pipe=pipeline, bg_color=background, d_xyz=d_xyz, d_rotation=d_rotation, d_scaling=d_scaling, is_6dof=is_6dof, override_color=cluster_point_colors.cuda().float())["render"]
            
            if multithread_save:            
                seg_list.append(segmentation_mask)
            else:
                torchvision.utils.save_image(segmentation_mask, os.path.join(seg_path, '{0:05d}'.format(idx) + ".png"))
                
            seg_images.append(to8b(segmentation_mask).transpose(1,2,0))
            
            if idx == 0: 
                results = render(view, gaussians, pipeline, background, 0.0, 0.0, 0.0, is_6dof)
                if multithread_save:
                    canonical_list.append(results["render"].cpu())
                else:
                    torchvision.utils.save_image(results["render"].cpu(), os.path.join(canonical_path, '{0:05d}'.format(idx) + ".png"))
                    
            if load_image_on_the_fly:
                with Image.open(view.image_path) as image_load:
                    im_data = np.array(image_load.convert("RGBA"))
                norm_data = im_data / 255.0
                arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + background.detach().cpu().numpy() * (1 - norm_data[:, :, 3:4])
                if norm_data[:, :, 3:4].min() < 1:
                    arr = np.concatenate([arr, norm_data[:, :, 3:4]], axis=2)
                    gt_image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGBA")
                else:
                    gt_image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
                gt = PILtoTorch(gt_image, (view.image_width, view.image_height))
            else:
                gt = view.original_image[0:3, :, :]
            if multithread_save:
                gt_list.append(gt.cpu())
            else:
                torchvision.utils.save_image(gt.cpu(), os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
                
            gt_images.append(to8b(gt).transpose(1,2,0))
            
            segmented_mask = None
            
            if segment_ids != -1:
                for id in segment_ids:
                    pre_mask = (cluster_ids_x == id)
                    filtered_mask = postprocessing(gaussians.get_gaussian_features.squeeze(1), gaussians.get_gaussian_features.squeeze(1)[pre_mask].mean(dim=0), score_threshold=score_threshold) ## Neu3D
                    post_mask = pre_mask & filtered_mask
                    if segmented_mask is None:
                        segmented_mask = post_mask
                    else:
                        segmented_mask |= post_mask
                
                buffer_image = render(view, gaussians, pipeline, torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"), d_xyz, d_rotation, d_scaling, is_6dof, mask=segmented_mask, override_color=torch.ones(cluster_point_colors.shape).cuda().float())['render']
                
                buffer_image[buffer_image < 0.5] = 0
                buffer_image[buffer_image != 0] = 1
                inlier_mask = buffer_image.mean(axis=0).bool()
                pred_masks_images.append(to8b(buffer_image).transpose(1,2,0))
                if multithread_save:
                    pred_masks_list.append(buffer_image.cpu())
                else:
                    torchvision.utils.save_image(buffer_image.cpu(), os.path.join(pred_masks_path, '{0:05d}'.format(idx) + ".png"))
                    
                buffer_image = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof, mask=segmented_mask)['render']
                
                if white_background:
                    buffer_image[:, ~inlier_mask] = 1
                else:
                    buffer_image[:, ~inlier_mask] = 0
                    
                segment_objects_images.append(to8b(buffer_image).transpose(1,2,0))
                if multithread_save:
                    segment_objects_list.append(buffer_image.cpu())
                else:
                    torchvision.utils.save_image(buffer_image.cpu(), os.path.join(segment_objects_path, '{0:05d}'.format(idx) + ".png"))
                    
            segmented_mask = None
            
            if text_prompt != '':
                for id in text_masked_cls_id:
                    pre_mask = (cluster_ids_x == id)
                    filtered_mask = postprocessing(gaussians.get_gaussian_features.squeeze(1), gaussians.get_gaussian_features.squeeze(1)[pre_mask].mean(dim=0), score_threshold=score_threshold)
                    post_mask = pre_mask & filtered_mask
                    if segmented_mask is None:
                        segmented_mask = post_mask
                    else:
                        segmented_mask |= post_mask
                        
                rendered_selected = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof, mask=segmented_mask, override_color=torch.ones(cluster_point_colors.shape).cuda().float())
                buffer_image = rendered_selected['render']
                
                buffer_image[buffer_image < 0.5] = 0
                buffer_image[buffer_image != 0] = 1
                inlier_mask = buffer_image.mean(axis=0).bool()
                buffer_image = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof, mask=segmented_mask)['render']
                if white_background:
                    buffer_image[:, ~inlier_mask] = 1
                else:
                    buffer_image[:, ~inlier_mask] = 0
                text_prompt_objects_images.append(to8b(buffer_image).transpose(1,2,0))
                if multithread_save:
                    text_prompt_objects_list.append(buffer_image.cpu())
                else:
                    torchvision.utils.save_image(buffer_image.cpu(), os.path.join(text_prompt_objects_path, '{0:05d}'.format(idx) + ".png"))
            if load2gpu_on_the_fly:
                view.load2device(data_device='cpu')
    except Exception as e:
        print(e)

    if multithread_save:
        multithread_write(render_list, render_path)
        multithread_write(gt_list, gts_path)
        multithread_write(rendered_feats_list, render_feats_path)
        multithread_write(canonical_list, canonical_path)
        multithread_write(pointcloud_list, point_path)
        multithread_write(gaussian_clusters_list, gaussian_clusters_path)
        multithread_write(seg_list, seg_path)
        multithread_write(gaussian_feats_list, gaussian_feats_path)
        multithread_write(segment_objects_list, segment_objects_path)
        multithread_write(pred_masks_list, pred_masks_path)
        multithread_write(text_prompt_objects_list, text_prompt_objects_path)

    del render_list
    del gt_list
    del rendered_feats_list
    del canonical_list
    del pointcloud_list
    del gaussian_clusters_list
    del seg_list
    del gaussian_feats_list
    del segment_objects_list
    del pred_masks_list
    del text_prompt_objects_list
    
    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_renders.mp4'), render_images[::2], fps=30, quality=8)
    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_gt.mp4'), gt_images[::2], fps=30, quality=8)
    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_rendered_feats.mp4'), rendered_feats_images[::2], fps=30, quality=8)
    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_pointcloud.mp4'), pointcloud_images[::2], fps=30, quality=8)
    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_gaussian_clusters.mp4'), gaussian_clusters_images[::2], fps=30, quality=8)
    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_segmentation.mp4'), seg_images[::2], fps=30, quality=8)
    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_gaussian_feats.mp4'), gaussian_feats_images[::2], fps=30, quality=8)
    
    if len(segment_objects_images) != 0:
        imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_segmented_objects.mp4'), segment_objects_images[::2], fps=30, quality=8)
        imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_pred_masks.mp4'), pred_masks_images[::2], fps=30, quality=8)
    if len(text_prompt_objects_images) != 0:
        imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), f'video_text_prompt_{text_prompt}_objects.mp4'), text_prompt_objects_images[::2], fps=30, quality=8)
    
    del render_images
    del gt_images
    del rendered_feats_images
    del pointcloud_images
    del gaussian_clusters_images
    del seg_images
    del gaussian_feats_images
    del segment_objects_images
    del pred_masks_images
    del text_prompt_objects_images
    
    out_path = os.path.join(render_path[:-8],'concat')
    makedirs(out_path,exist_ok=True)
    concat_list = []
    concat_images = []
    for idx, file_name in enumerate(tqdm(sorted(os.listdir(gts_path)))):
        if os.path.exists(os.path.join(segment_objects_path, file_name)):
            try:
                rgb = np.array(Image.open(os.path.join(render_path, file_name)))
                cls = np.array(Image.open(os.path.join(gaussian_clusters_path, file_name)))
                seg = np.array(Image.open(os.path.join(seg_path,file_name)))
                seg_obj = np.array(Image.open(os.path.join(segment_objects_path,file_name)))

                result = np.hstack([rgb, cls, seg, seg_obj])
                concat_images.append(result)
                if multithread_save:
                    concat_list.append(result.astype('uint8'))
                else:
                    # torchvision.utils.save_image(result.astype('uint8'), out_path, '{0:05d}'.format(idx) + ".png")
                    Image.fromarray(result.astype('uint8')).save(os.path.join(out_path, '{0:05d}'.format(idx) + ".png"))
                    
            except Exception as error:
                print("An exception occurred:", error) # An exception occurred: division by zero
        else:
            try:
                rgb = np.array(Image.open(os.path.join(render_path, file_name)))
                cls = np.array(Image.open(os.path.join(gaussian_clusters_path, file_name)))
                seg = np.array(Image.open(os.path.join(seg_path,file_name)))

                result = np.hstack([rgb, cls, seg])
                concat_images.append(result)
                if multithread_save:
                    concat_list.append(result.astype('uint8'))
                else:
                    # torchvision.utils.save_image(result.astype('uint8'), out_path, '{0:05d}'.format(idx) + ".png")
                    Image.fromarray(result.astype('uint8')).save(os.path.join(out_path, '{0:05d}'.format(idx) + ".png"))
                    
            except Exception as error:
                print("An exception occurred:", error) # An exception occurred: division by zero
    if multithread_save:
        multithread_write(concat_list, out_path)
        
    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_concat.mp4'), concat_images[::2], fps=30, quality=8)
    

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, model_type : str, segment_ids: list, text_prompt: str, threshold: int, kmeans: bool, score_threshold: float, multithread_save: bool):
    print("Deform type: ", model_type)
    print("Segment object IDs: ", segment_ids)
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        try:
            if not kmeans:
                print("Load from DBSCAN clusters")
                gaussians.load_clusters(path=os.path.join(dataset.model_path, f'point_cloud/iteration_{str(iteration)}/clusters.pt'))
            else:
                print("Load from K-Means clusters")
                gaussians.load_clusters(path=os.path.join(dataset.model_path, f'point_cloud/iteration_{str(iteration)}/clusters_kmeans.pt'))
        except:
            print("[WARNING] No cluster Ids found")
            
        deform = DeformModel(dataset.is_blender, dataset.is_6dof, model_type=model_type)
        deform.load_weights(dataset.model_path, iteration=iteration)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, dataset.is_6dof, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, deform, dataset.load2gpu_on_the_fly, model_type, dataset.load_image_on_the_fly, segment_ids, text_prompt, threshold, dataset.white_background, score_threshold, multithread_save)
                
        if (not skip_test) and (len(scene.getTestCameras()) > 0):
            render_set(dataset.model_path, dataset.is_6dof, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, deform, dataset.load2gpu_on_the_fly, model_type, dataset.load_image_on_the_fly, segment_ids, text_prompt, threshold, dataset.white_background, score_threshold, multithread_save)
                
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)

    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--segment_ids', type=int, nargs='+', default=-1)
    parser.add_argument('--text_prompt', type=str, default='')
    parser.add_argument("--threshold", default=500, type=int)
    parser.add_argument("--score_threshold", default=0.0, type=float)
    parser.add_argument('--kmeans', action="store_true")
    parser.add_argument('--multithread_save', action="store_true", default=False)
    
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, op.extract(args).deform_type, args.segment_ids, args.text_prompt, args.threshold, args.kmeans, args.score_threshold, args.multithread_save)