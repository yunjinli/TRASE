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
# Modified from codes in SC-GS: https://github.com/yihua7/SC-GS
#


import os
import time
import torch
from random import randint
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel, DeformModel
from utils.general_utils import safe_state, get_linear_noise_func
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
import math
from cam_utils import OrbitCamera
import numpy as np
import dearpygui.dearpygui as dpg
import imageio
import datetime
from PIL import Image
from scipy.spatial.transform import Rotation as R
from kmeans_pytorch import kmeans
import pytorch3d.ops as ops
from utils.rigid_utils import from_homogenous, to_homogenous
import torchvision
import hdbscan
import multiprocessing
import concurrent.futures
from utils.general_utils import PILtoTorch

from ext.grounded_sam import grouned_sam_output, load_model_hf, select_obj_ioa
from segment_anything import sam_model_registry, SamPredictor
import math
import pytorch3d.ops as ops
from os import makedirs
import time

def feature3d_to_rgb(x, n_components=3):      
    X_center = x - torch.mean(x, axis=0)   # Center data
    q ,r = torch.linalg.qr(X_center)
    U, s, Vt = torch.linalg.svd(r, full_matrices=False)
    x_compress = torch.matmul(U[:, :n_components],torch.diag(s[:n_components]))
    pca_result = torch.matmul(q, x_compress)
    pca_normalized = (pca_result - pca_result.min()) / (pca_result.max() - pca_result.min())
    return pca_normalized

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 1 / tanHalfFovX
    P[1, 1] = 1 / tanHalfFovY
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def landmark_interpolate(landmarks, steps, step, interpolation='log'):
    stage = (step >= np.array(steps)).sum()
    if stage == len(steps):
        return max(0, landmarks[-1])
    elif stage == 0:
        return 0
    else:
        ldm1, ldm2 = landmarks[stage-1], landmarks[stage]
        if ldm2 <= 0:
            return 0
        step1, step2 = steps[stage-1], steps[stage]
        ratio = (step - step1) / (step2 - step1)
        if interpolation == 'log':
            return np.exp(np.log(ldm1) * (1 - ratio) + np.log(ldm2) * ratio)
        elif interpolation == 'linear':
            return ldm1 * (1 - ratio) + ldm2 * ratio
        else:
            print(f'Unknown interpolation type: {interpolation}')
            raise NotImplementedError

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def generate_grid_index(depth):
    h, w = depth.shape
    grid = torch.meshgrid([torch.arange(h), torch.arange(w)])
    grid = torch.stack(grid, dim=-1)
    return grid

to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

class MiniCam:
    def __init__(self, c2w, width, height, fovy, fovx, znear, zfar, fid):
        # c2w (pose) should be in NeRF convention.

        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.fid = fid
        self.c2w = c2w

        w2c = np.linalg.inv(c2w)

        # rectify...
        w2c[1:3, :3] *= -1
        w2c[:3, 3] *= -1

        self.world_view_transform = torch.tensor(w2c).transpose(0, 1).cuda().float()
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .cuda().float()
        )
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = -torch.tensor(c2w[:3, 3]).cuda()

    def reset_extrinsic(self, R, T):
        self.world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


class GUI:
    def __init__(self, args, dataset, pipe, iteration, opt) -> None:
        self.dataset = dataset
        self.pipe = pipe
        self.iteration = iteration

        self.gaussians = GaussianModel(dataset.sh_degree)
        self.scene = Scene(dataset, self.gaussians, load_iteration=iteration, shuffle=False)
        
        self.deform_type = opt.deform_type
        self.deform = DeformModel(dataset.is_blender, dataset.is_6dof, self.deform_type)
        self.deform.load_weights(dataset.model_path, iteration=iteration)
        
        self.label_to_color = np.random.rand(1000, 3)
        self.num_clusters = 64
        
        if not self.gaussians.has_cluster_ids:
            try:
                self.gaussians.load_clusters(path=os.path.join(dataset.model_path, f'point_cloud/iteration_{str(iteration)}/clusters.pt'))
                self.cluster_ids_x = self.gaussians.get_clusters['id'].squeeze()
                self.cluster_point_colors = self.gaussians.get_clusters['rgb']
                self.num_clusters = str(len(np.unique(self.cluster_ids_x.detach().cpu().numpy())))
                
                print(f"[{__name__}] DBSCAN cluster loaded")
                
            except:
                self.cluster_point_colors = None
                self.cluster_ids_x = None
                print(f"[{__name__}][WARNING] No cluster_ids found, need to run [clustering]!!!")
        else:
            self.cluster_ids_x = self.gaussians.get_clusters['id'].int().squeeze(-1).detach().cpu().numpy()
            self.cluster_point_colors = torch.from_numpy(self.label_to_color[self.cluster_ids_x])
            print(self.cluster_ids_x)
            print(f"[{__name__}] Number of no-clustered points", (self.cluster_ids_x == -1).sum())
            print(self.cluster_ids_x.shape)
            print(f"[{__name__}] Number of clusters: ", len(np.unique(self.cluster_ids_x)))
            self.num_clusters = str(len(np.unique(self.cluster_ids_x)))
        
        self.gaussians_feature_pca = feature3d_to_rgb(self.gaussians.get_gaussian_features.squeeze(1))
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.viewpoint_stack = None
        
        self.is_6dof = dataset.is_6dof
        # For UI
        self.visualization_mode = 'RGB'

        self.W = args.W
        self.H = args.H
        self.cam = OrbitCamera(args.W, args.H, r=args.radius, fovy=args.fovy)
        self.vis_scale_const = None
        self.mode = "Render"
        self.seed = "random"
        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.video_speed = 1.
        
        # For Animation
        self.animation_time = 0.
        self.is_animation = False
        self.need_update_overlay = False
        self.buffer_overlay = None
        self.animate_tool = None
        self.showing_overlay = True
        self.traj_overlay = None
        self.vis_traj_realtime = False
        
        ## Segmentation
        self.selected_point_idcs = []
        self.mask_changed = False
        self.segmented_mask = None
        self.deformed_pcd_at_t = None
        self.motion_segmentation = False
        
        self.seg_score = None
        self.render_segmentation_mask = False
        
        ## Text Prompt backend
        ckpt_repo_id = "ShilongLiu/GroundingDINO"
        ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
        ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
        self.groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename).to(device='cpu')
        sam_checkpoint = 'dependency/sam_vit_h_4b8939.pth'
        sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        sam.to(device='cpu')
        self.sam_predictor = SamPredictor(sam)
        self.text_prompt = ''
        self.clustering_mode = 'DBSCAN'
        self.remove_selceted = False
        self.score_threshold = 0.1
        
        def kmean_clustering():
            # kmeans
            print(f"K-Means with K = {self.num_clusters}...")

            point_features = self.gaussians.get_gaussian_features.squeeze(1)
            normed_point_features = torch.nn.functional.normalize(point_features, dim = -1, p = 2)
            self.cluster_ids_x, _ = kmeans(
                    X=normed_point_features, num_clusters=self.num_clusters, distance='euclidean', device=torch.device('cuda:0')
                )
            self.cluster_ids_x = self.cluster_ids_x.cpu().numpy()
            self.cluster_point_colors = torch.from_numpy(self.label_to_color[self.cluster_ids_x]).float().cuda()
            # self.cluster_ids_x = self.seg_score.argmax(dim = -1).cpu().numpy()
            print(self.cluster_ids_x)
            print("Number of no-clustered points", (self.cluster_ids_x == -1).sum())
            print(self.cluster_ids_x.shape)
            print("Number of clusters: ", len(np.unique(self.cluster_ids_x)))
            
            self.cluseters = {"id": self.cluster_ids_x,
                            "rgb": self.cluster_point_colors}
            torch.save(self.cluseters, os.path.join(self.dataset.model_path, f"point_cloud/iteration_{self.iteration}/clusters_kmeans.pt"))
            
            self.gaussians.load_clusters(os.path.join(self.dataset.model_path, f"point_cloud/iteration_{self.iteration}/clusters_kmeans.pt"))
            self.cluster_ids_x = self.gaussians.get_clusters['id'].squeeze()
        def dbscan_clustering():
            # kmeans
            print("DBSCAN...")
            percent = 0.02
            point_features = self.gaussians.get_gaussian_features.squeeze(1)
            normed_point_features = torch.nn.functional.normalize(point_features, dim = -1, p = 2)
            sampled_point_features = point_features[torch.rand(point_features.shape[0]) > 1 - percent]
            normed_sampled_point_features = sampled_point_features / torch.norm(sampled_point_features, dim = -1, keepdim = True)
            
            clusterer = hdbscan.HDBSCAN(min_cluster_size=10, cluster_selection_epsilon=0.01, allow_single_cluster = False, core_dist_n_jobs=multiprocessing.cpu_count())
            
            cluster_labels = clusterer.fit_predict(normed_sampled_point_features.detach().cpu().numpy())

            cluster_centers = torch.zeros(len(np.unique(cluster_labels)), normed_sampled_point_features.shape[-1])
            for i in range(0, len(np.unique(cluster_labels))):
                cluster_centers[i] = torch.nn.functional.normalize(normed_sampled_point_features[cluster_labels == i-1].mean(dim = 0), dim = -1)

            self.seg_score = torch.einsum('nc,bc->bn', cluster_centers.cpu(), normed_point_features.cpu())
            self.cluster_point_colors = torch.from_numpy(self.label_to_color[self.seg_score.argmax(dim = -1).cpu().numpy()]).float().cuda()
            self.cluster_ids_x = self.seg_score.argmax(dim = -1).cpu().numpy()
            print(self.cluster_ids_x)
            print("Number of no-clustered points", (self.cluster_ids_x == -1).sum())
            print(self.cluster_ids_x.shape)
            print("Number of clusters: ", len(np.unique(cluster_labels)))
            dpg.set_value("_number_of_k", str(len(np.unique(cluster_labels))))
            self.cluseters = {"id": self.cluster_ids_x,
                            "rgb": self.cluster_point_colors}
            torch.save(self.cluseters, os.path.join(self.dataset.model_path, f"point_cloud/iteration_{self.iteration}/clusters.pt"))
            
            self.gaussians.load_clusters(os.path.join(self.dataset.model_path, f"point_cloud/iteration_{self.iteration}/clusters.pt"))
            self.cluster_ids_x = self.gaussians.get_clusters['id'].squeeze()
        self.clustering = {
            "K-Means": kmean_clustering,
            "DBSCAN": dbscan_clustering
        }
        def load_kmean():
            self.gaussians.load_clusters(path=os.path.join(dataset.model_path, f'point_cloud/iteration_{str(iteration)}/clusters_kmeans.pt'))
            self.cluster_ids_x = self.gaussians.get_clusters['id'].squeeze()
            self.cluster_point_colors = self.gaussians.get_clusters['rgb']
            
        def load_dbscan():
            self.gaussians.load_clusters(path=os.path.join(dataset.model_path, f'point_cloud/iteration_{str(iteration)}/clusters.pt'))
            self.cluster_ids_x = self.gaussians.get_clusters['id'].squeeze()
            self.cluster_point_colors = self.gaussians.get_clusters['rgb']
            
        self.load_cluster = {
            "K-Means": load_kmean,
            "DBSCAN": load_dbscan
        }
        dpg.create_context()
        self.register_dpg()
        self.test_step()
    
    @torch.no_grad()
    def render_set(self, model_path, is_6dof, name, iteration, views, gaussians, pipeline, background, deform, load2gpu_on_the_fly, load_image_on_the_fly, segmented_mask=None, white_background=False):
        render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
        gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
        segment_objects_path = os.path.join(model_path, name, "ours_{}".format(iteration), "segment_objects")
        pred_masks_path = os.path.join(model_path, name, "ours_{}".format(iteration), "pred_masks")
        
        if self.remove_selceted:
            remove_objects_path = os.path.join(model_path, name, "ours_{}".format(iteration), "remove_objects")
            makedirs(remove_objects_path, exist_ok=True)
        
        makedirs(render_path, exist_ok=True)
        makedirs(gts_path, exist_ok=True)
        makedirs(segment_objects_path, exist_ok=True)
        makedirs(pred_masks_path, exist_ok=True)
        
        render_images = []
        render_list = []
        gt_list = []
        gt_images = []
        obj_images = []
        obj_list = []
        canonical_list = []
        pointcloud_list = []
        pointcloud_images = []
        point_cls_list = []
        point_cls_images = []
        point_feats_list = []
        point_feats_images = []
        seg_list = []
        seg_images = []
        segment_objects_list = []
        segment_objects_images = []
        text_prompt_objects_list = []
        text_prompt_objects_images = []
        pred_masks_list = []
        pred_masks_images = []
        
        remove_objects_list = []
        remove_objects_images = []
        
        cluster_point_colors = gaussians.get_clusters['rgb']
        
        for idx, view in tqdm(enumerate(tqdm(views, desc="Rendering progress"))):
            ## Deformation
            if load2gpu_on_the_fly:
                view.load2device()
            fid = view.fid
            xyz = gaussians.get_xyz
            time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
            d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
                
            results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof)
            
            rendering = results["render"]
            
            render_images.append(to8b(rendering).transpose(1,2,0))
            torchvision.utils.save_image(rendering.cpu(), os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            
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
            torchvision.utils.save_image(gt.cpu(), os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            gt_images.append(to8b(gt).transpose(1,2,0))
            buffer_image = render(view, gaussians, pipeline, torch.tensor([0, 0, 0]).float().cuda(), d_xyz, d_rotation, d_scaling, is_6dof, mask=segmented_mask, override_color=torch.ones(cluster_point_colors.shape).cuda().float())['render']
            
            buffer_image[buffer_image < 0.5] = 0
            buffer_image[buffer_image != 0] = 1
            inlier_mask = buffer_image.mean(axis=0).bool()
            pred_masks_images.append(to8b(buffer_image).transpose(1,2,0))
            torchvision.utils.save_image(buffer_image.cpu(), os.path.join(pred_masks_path, '{0:05d}'.format(idx) + ".png"))
            
            buffer_image = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof, mask=segmented_mask)['render']
            if white_background:
                buffer_image[:, ~inlier_mask] = 1
            else:
                buffer_image[:, ~inlier_mask] = 0
            segment_objects_images.append(to8b(buffer_image).transpose(1,2,0))
            torchvision.utils.save_image(buffer_image.cpu(), os.path.join(segment_objects_path, '{0:05d}'.format(idx) + ".png"))
            
            if self.remove_selceted:
                buffer_image = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof, mask=~segmented_mask)['render']
                remove_objects_images.append(to8b(buffer_image).transpose(1,2,0))
                torchvision.utils.save_image(buffer_image.cpu(), os.path.join(remove_objects_path, '{0:05d}'.format(idx) + ".png"))
                
            if load2gpu_on_the_fly:
                view.load2device(data_device='cpu')
                
        del render_list
        del gt_list
        del obj_list
        del canonical_list
        del pointcloud_list
        del point_cls_list
        del seg_list
        del point_feats_list
        del segment_objects_list
        del pred_masks_list
        del text_prompt_objects_list
        del remove_objects_list
        
        imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_renders.mp4'), render_images[::2], fps=30, quality=8)
        imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_gt.mp4'), gt_images[::2], fps=30, quality=8)
        if len(segment_objects_images) != 0:
            imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_segmented_objects.mp4'), segment_objects_images[::2], fps=30, quality=8)
            imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_pred_masks.mp4'), pred_masks_images[::2], fps=30, quality=8)
        if len(remove_objects_images) != 0:
            imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_remove_objects.mp4'), remove_objects_images[::2], fps=30, quality=8)
              
        del render_images
        del gt_images
        del obj_images
        del pointcloud_images
        del point_cls_images
        del seg_images
        del point_feats_images
        del segment_objects_images
        del pred_masks_images
        del text_prompt_objects_images
        del remove_objects_images
        
       
    @torch.no_grad()
    def postprocessing(self, features, query_feature, score_threshold=0.8):
        features /= features.norm(dim=-1, keepdim=True)
        query_feature /= query_feature.norm(dim=-1, keepdim=True)
        query_feature = query_feature.unsqueeze(-1)
        scores = features.half() @ query_feature.half()
        scores = scores[:, 0]
        mask = (scores >= score_threshold)
        return mask
    
    def __del__(self):
        dpg.destroy_context()

    def register_dpg(self):
        ### register texture
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.W,
                self.H,
                self.buffer_image,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture",
            )

        ### register window
        # the rendered image, as the primary window
        with dpg.window(
            tag="_primary_window",
            width=self.W,
            height=self.H,
            pos=[0, 0],
            no_move=True,
            no_title_bar=True,
            no_scrollbar=True,
        ):
            # add the texture
            dpg.add_image("_texture")

        # control window
        with dpg.window(
            label="Control",
            tag="_control_window",
            width=600,
            height=self.H,
            pos=[self.W, 0],
            no_move=True,
            no_title_bar=True,
        ):
            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # timer stuff
            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")

            def callback_setattr(sender, app_data, user_data):
                setattr(self, user_data, app_data)
                
            with dpg.collapsing_header(label="Segmentation", default_open=True):
                with dpg.group(horizontal=True):
                    def callback_change_clustering_mode(sender, app_data):
                        self.clustering_mode = app_data
                        if self.clustering_mode == "DBSCAN":
                            dpg.configure_item("_number_of_k", enabled=False)
                        else:
                            dpg.configure_item("_number_of_k", enabled=True)
                            
                    dpg.add_combo(
                        ("K-Means", "DBSCAN"),
                        default_value=self.clustering_mode,
                        callback=callback_change_clustering_mode,
                        width=100
                    )
                    dpg.add_input_text(label="", tag="_number_of_k", width=100)
                    dpg.configure_item("_number_of_k", enabled=False)
                    dpg.set_value('_number_of_k', str(self.num_clusters))
                    def callback_clustering(sender, app_data):
                        # self.clustering()
                        self.num_clusters = (int)(dpg.get_value('_number_of_k'))
                        print("Clustering Mode: ", self.clustering_mode)
                        
                        self.clustering[self.clustering_mode]()
                    dpg.add_button(
                        label="Clustering",
                        tag="_button_clustering",
                        callback=callback_clustering,
                    )
                
                    dpg.bind_item_theme("_button_clustering", theme_button)
                    
                    def callback_load_cluster(sender, app_data):
                        print(f"Load saved {self.clustering_mode} clusters")
                        self.load_cluster[self.clustering_mode]()
                    dpg.add_button(
                        label="Load",
                        tag="_button_load_clustering",
                        callback=callback_load_cluster,
                    )
                
                    dpg.bind_item_theme("_button_load_clustering", theme_button)
                
                def callback_render_segmentation_mask(sender, app_data):
                    self.render_segmentation_mask = ~self.render_segmentation_mask
                    
                dpg.add_button(
                    label="Render mask",
                    tag="_button_render_segmentation_mask",
                    callback=callback_render_segmentation_mask,
                )
                
                dpg.bind_item_theme("_button_render_segmentation_mask", theme_button)
            with dpg.collapsing_header(label="Scene Editing", default_open=True):
                # with dpg.collapsing_header(label="Text Prompt", default_open=True):
                with dpg.group(horizontal=True):
                    dpg.add_text("Text Prompt ")
                    dpg.add_input_text(label="", tag="_input_text_prompt")
                    def callback_input_text_prompt(sender, app_data):
                        self.text_prompt = dpg.get_value('_input_text_prompt')
                        print("Got text prompt: ", self.text_prompt)
                    dpg.add_button(label="Enter", callback=callback_input_text_prompt)
                dpg.add_slider_int(label="Text Prompt Threshold", default_value=5000,
                                min_value=0, max_value=10000, tag="_text_prompt_threshold")
                def callback_score_threshold(sender, app_data):
                    self.score_threshold = dpg.get_value('_click_prompt_threshold')
                    print(f"Change score threshold to {self.score_threshold}")
                    # self.callback_select_point()
                    self.mask_changed = True
                    if self.mask_changed:
                        print("Compute object mask...")
                        self.selected_clusters = []
                        if len(self.selected_point_idcs) == 0:
                            self.segmented_mask = None
                        else:
                            # self.segmented_mask = (self.cluster_ids_x in self.cluster_ids_x[np.array(self.selected_point_idcs)])
                            self.segmented_mask = None
                            for selected_point_i in self.selected_point_idcs:
                                ## With post-processing
                                pre_mask = (self.cluster_ids_x == self.cluster_ids_x[selected_point_i])
                                
                                filtered_mask = self.postprocessing(self.gaussians.get_gaussian_features.squeeze(1), self.gaussians.get_gaussian_features.squeeze(1)[pre_mask].mean(dim=0), score_threshold=self.score_threshold) 
                                post_mask = pre_mask & filtered_mask
                                if self.segmented_mask is None:
                                    self.segmented_mask = post_mask
                                else:
                                    self.segmented_mask |= post_mask
                    
                                if self.cluster_ids_x[selected_point_i] not in self.selected_clusters:
                                    self.selected_clusters.append(self.cluster_ids_x[selected_point_i])
                        print("Selected cluster ID: ", [print_id.int().detach().cpu().numpy()[0] for print_id in self.selected_clusters])
                                
                        self.mask_changed = False
                dpg.add_slider_float(label="Score Threshold", default_value=self.score_threshold, callback=callback_score_threshold,
                                min_value=0, max_value=1, tag="_click_prompt_threshold")
            
                def callback_remove_object(sender, app_data):
                    self.remove_selceted = ~self.remove_selceted
                dpg.add_button(
                    label="Remove Object",
                    tag="_button_remove_object",
                    callback=callback_remove_object,
                )
                
                dpg.bind_item_theme("_button_remove_object", theme_button)
                    
                def callback_save_object(sender, app_data):
                    # self.render_segmentation_mask = ~self.render_segmentation_mask
                    if self.remove_selceted:
                        self.gaussians.save_ply(path=os.path.join(self.dataset.model_path, f'point_cloud/iteration_{str(self.iteration)}/point_cloud_object.ply'), mask=~self.segmented_mask)
                    else:
                        self.gaussians.save_ply(path=os.path.join(self.dataset.model_path, f'point_cloud/iteration_{str(self.iteration)}/point_cloud_object.ply'), mask=self.segmented_mask)
                    
                dpg.add_button(
                    label="Save Object",
                    tag="_button_save_object",
                    callback=callback_save_object,
                )
                
                dpg.bind_item_theme("_button_save_object", theme_button)
                
                def callback_render_object(sender, app_data):
                    self.render_set(self.dataset.model_path, self.dataset.is_6dof, "test", self.scene.loaded_iter, self.scene.getTestCameras(), self.gaussians, self.pipe, self.background, self.deform, self.dataset.load2gpu_on_the_fly, self.dataset.load_image_on_the_fly, self.segmented_mask)
                    
                dpg.add_button(
                    label="Render Object",
                    tag="_button_render_object",
                    callback=callback_render_object,
                )
                
                dpg.bind_item_theme("_button_render_object", theme_button)
                
                def callback_vis_traj_realtime():
                    self.vis_traj_realtime = not self.vis_traj_realtime
                    if not self.vis_traj_realtime:
                        self.traj_coor = None
                    print('Visualize trajectory: ', self.vis_traj_realtime)
                dpg.add_button(
                    label="Traj",
                    tag="_button_vis_traj",
                    callback=callback_vis_traj_realtime,
                )
                dpg.bind_item_theme("_button_vis_traj", theme_button)

            # rendering options
            with dpg.collapsing_header(label="Rendering", default_open=True):
                # mode combo
                def callback_change_mode(sender, app_data):
                    self.mode = app_data
                    self.need_update = True

                dpg.add_combo(
                    ("Render", "Rendered Features", "Gaussian Features", "Gaussian Clusters", "Segmentation", "Point Cloud", "Depth"),
                    label="mode",
                    default_value=self.mode,
                    callback=callback_change_mode,
                )

                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.cam.fovy = np.deg2rad(app_data)
                    self.need_update = True

                dpg.add_slider_int(
                    label="FoV (vertical)",
                    min_value=1,
                    max_value=120,
                    format="%d deg",
                    default_value=np.rad2deg(self.cam.fovy),
                    callback=callback_set_fovy,
                )
            
            # animation options
            with dpg.collapsing_header(label="Time Editing", default_open=True):
                # save current model
                with dpg.group(horizontal=True):
                    dpg.add_text("Freeze Time: ")
                    def callback_animation_time(sender):
                        self.animation_time = dpg.get_value(sender)
                        self.is_animation = True
                        self.need_update = True
                        # self.animation_initialize()
                    dpg.add_slider_float(
                        label="",
                        default_value=0.,
                        max_value=1.,
                        min_value=0.,
                        callback=callback_animation_time,
                    )
                    def callback_animation_mode(sender, app_data):
                        with torch.no_grad():
                            self.is_animation = not self.is_animation
                            if self.is_animation:
                                if not hasattr(self, 'animate_tool') or self.animate_tool is None:
                                    self.animation_initialize()
                    dpg.add_button(
                        label="Play",
                        tag="_button_vis_animation",
                        callback=callback_animation_mode,
                        user_data='Animation',
                    )
                    dpg.bind_item_theme("_button_vis_animation", theme_button)

        def callback_set_mouse_loc(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return
            self.mouse_loc = np.array(app_data)

        def callback_keypoint_drag(sender, app_data):
            if not self.is_animation:
                print("Please switch to animation mode!")
                return
            if not dpg.is_item_focused("_primary_window"):
                return
            if len(self.deform_keypoints.get_kpt()) == 0:
                return
            if self.animate_tool is None:
                self.animation_initialize()
            # 2D to 3D delta
            dx = app_data[1]
            dy = app_data[2]
            if dpg.is_key_down(dpg.mvKey_R):
                side = self.cam.rot.as_matrix()[:3, 0]
                up = self.cam.rot.as_matrix()[:3, 1]
                forward = self.cam.rot.as_matrix()[:3, 2]
                rotvec_z = forward * np.radians(-0.05 * dx)
                rot_mat = (R.from_rotvec(rotvec_z)).as_matrix()
                self.deform_keypoints.set_rotation_delta(rot_mat)
            else:
                delta = 0.00010 * self.cam.rot.as_matrix()[:3, :3] @ np.array([dx, -dy, 0])
                self.deform_keypoints.update_delta(delta)
                self.need_update_overlay = True

        def callback_select_point(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return
            ##### select keypoints by shift + click
            
            if dpg.is_key_down(dpg.mvKey_S) or dpg.is_key_down(dpg.mvKey_D) or dpg.is_key_down(dpg.mvKey_F) or dpg.is_key_down(dpg.mvKey_A) or dpg.is_key_down(dpg.mvKey_Q):
                if not self.is_animation:
                    print("Please switch to animation mode!")
                    return
                # Rendering the image with node gaussians to select nodes as keypoints
                fid = torch.tensor(self.animation_time).cuda().float()
                cur_cam = MiniCam(
                    self.cam.pose,
                    self.W,
                    self.H,
                    self.cam.fovy,
                    self.cam.fovx,
                    self.cam.near,
                    self.cam.far,
                    fid = fid
                )
                with torch.no_grad():
                    gaussians = self.gaussians
                    xyz = gaussians.get_xyz
                    N = self.gaussians.get_xyz.shape[0]
                    time_input = fid.unsqueeze(0).expand(N, -1)
                    
                    d_xyz, d_rotation, d_scaling = self.deform.step(xyz.detach(), time_input) if self.deform_type == 'DeformNetwork' else self.deform.step(xyz.detach(), time_input, gaussians.get_gaussian_features.squeeze(1))

                    out = render(viewpoint_camera=cur_cam, pc=gaussians, pipe=self.pipe, bg_color=self.background, d_xyz=d_xyz, d_rotation=d_rotation, d_scaling=d_scaling, is_6dof=self.is_6dof)

                    # Project mouse_loc to points_3d
                    pw, ph = int(self.mouse_loc[0]), int(self.mouse_loc[1])
                    d = out['depth'][0][ph, pw]
                    z = cur_cam.zfar / (cur_cam.zfar - cur_cam.znear) * d - cur_cam.zfar * cur_cam.znear / (cur_cam.zfar - cur_cam.znear)
                    uvz = torch.tensor([((pw-.5)/self.W * 2 - 1) * d, ((ph-.5)/self.H*2-1) * d, z, d]).cuda().float().view(1, 4)
                    p3d = (uvz @ torch.inverse(cur_cam.full_proj_transform))[0, :3]

                    if self.is_6dof:
                        self.deformed_pcd_at_t = from_homogenous(
                            torch.bmm(d_xyz, to_homogenous(xyz).unsqueeze(-1)).squeeze(-1))
                    else:
                        self.deformed_pcd_at_t = xyz + d_xyz
                    keypoint_idxs = torch.tensor([(p3d - self.deformed_pcd_at_t).norm(dim=-1).argmin()]).cuda()
                    
                if dpg.is_key_down(dpg.mvKey_A):
                    self.selected_point_idcs.append(keypoint_idxs.detach().cpu().numpy())
                    self.mask_changed = True
                    print(f"Select point ID: {keypoint_idxs.detach().cpu().numpy()} from cluster {self.cluster_ids_x[keypoint_idxs.detach()]}")
                
                if dpg.is_key_down(dpg.mvKey_D):
                    dmax = 1000000
                    for pid in self.selected_point_idcs:
                        d = (self.deformed_pcd_at_t[pid].squeeze(0) - p3d).norm()
                        if d < dmax:
                            dmax = d 
                            remove_idx = pid
                    self.selected_point_idcs.remove(remove_idx)
                    print(f"Delete point ID: {keypoint_idxs.detach().cpu().numpy()}")
                    self.mask_changed = True
                print(f"Current selected point IDs: {self.selected_point_idcs}")
                
                self.need_update_overlay = True
                if self.mask_changed:
                    print("Compute object mask...")
                    self.selected_clusters = []
                    if len(self.selected_point_idcs) == 0:
                        self.segmented_mask = None
                    else:
                        self.segmented_mask = None
                        for selected_point_i in self.selected_point_idcs:
                            ## With post-processing
                            pre_mask = (self.cluster_ids_x == self.cluster_ids_x[selected_point_i])
                            
                            filtered_mask = self.postprocessing(self.gaussians.get_gaussian_features.squeeze(1), self.gaussians.get_gaussian_features.squeeze(1)[pre_mask].mean(dim=0), score_threshold=self.score_threshold) 
                            post_mask = pre_mask & filtered_mask
                            if self.segmented_mask is None:
                                self.segmented_mask = post_mask
                            else:
                                self.segmented_mask |= post_mask
                
                            if self.cluster_ids_x[selected_point_i] not in self.selected_clusters:
                                self.selected_clusters.append(self.cluster_ids_x[selected_point_i])
                    print("Selected cluster ID: ", [print_id.int().detach().cpu().numpy()[0] for print_id in self.selected_clusters])
                            
                    self.mask_changed = False
        self.callback_select_point = callback_select_point
        self.callback_keypoint_drag = callback_keypoint_drag

        ### register camera handler

        def callback_camera_drag_rotate_or_draw_mask(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.orbit(dx, dy)
            self.need_update = True

        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

        def callback_camera_drag_pan(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True
                
        with dpg.handler_registry():
            # for camera moving
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left,
                callback=callback_camera_drag_rotate_or_draw_mask,
            )
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan
            )

            dpg.add_mouse_move_handler(callback=callback_set_mouse_loc)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Right, callback=callback_keypoint_drag)
            dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Left, callback=callback_select_point)

        dpg.create_viewport(
            title="TRASE: Tracking-free 4D Segmentation and Editing",
            width=self.W + 600,
            height=self.H + (45 if os.name == "nt" else 0),
            resizable=False,
        )

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core
                )

        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        if os.path.exists("LXGWWenKai-Regular.ttf"):
            with dpg.font_registry():
                with dpg.font("LXGWWenKai-Regular.ttf", 18) as default_font:
                    dpg.bind_font(default_font)

        dpg.show_viewport()

    # gui mode
    def render(self):
        while dpg.is_dearpygui_running():
            self.test_step()
            dpg.render_dearpygui_frame()

    @torch.no_grad()
    def test_step(self, specified_cam=None):

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        if not hasattr(self, 't0'):
            self.t0 = time.time()
            self.fps_of_fid = 10
            
        if self.is_animation:
            if not self.showing_overlay:
                self.buffer_overlay = None
            else:
                self.update_control_point_overlay()
            fid = torch.tensor(self.animation_time).cuda().float()
        else:
            fid = torch.remainder(torch.tensor((time.time()-self.t0) * self.fps_of_fid).float().cuda() / len(self.scene.getTestCameras()) * self.video_speed, 1.)

        cur_cam = MiniCam(
            self.cam.pose,
            self.W,
            self.H,
            self.cam.fovy,
            self.cam.fovx,
            self.cam.near,
            self.cam.far,
            fid = fid
        )
        
        fid = cur_cam.fid
        
        gaussians = self.gaussians
        xyz = gaussians.get_xyz
        N = self.gaussians.get_xyz.shape[0]
        time_input = fid.unsqueeze(0).expand(N, -1)
    
        d_xyz, d_rotation, d_scaling = self.deform.step(xyz.detach(), time_input) if self.deform_type == 'DeformNetwork' else self.deform.step(xyz.detach(), time_input, gaussians.get_gaussian_features.squeeze(1))
        
        if self.vis_traj_realtime:
            if self.is_6dof:
                self.update_trajectory_overlay(gs_xyz=from_homogenous(torch.bmm(d_xyz, to_homogenous(xyz).unsqueeze(-1)).squeeze(-1)), camera=cur_cam, segmentation_mask=self.segmented_mask)
            else:    
                self.update_trajectory_overlay(gs_xyz=gaussians.get_xyz + d_xyz, camera=cur_cam, segmentation_mask=self.segmented_mask)
        
        motion_segmentation_mask = None
    
        if self.mode in ['Point Cloud', 'Gaussian Features', 'Gaussian Clusters']:
            if not self.motion_segmentation:
                if self.is_6dof:
                    cur_pts = torch.cat([from_homogenous(
                                torch.bmm(d_xyz, to_homogenous(xyz).unsqueeze(-1)).squeeze(-1)), torch.ones_like(xyz[..., :1])], dim=-1).cuda()
                else:
                    if self.segmented_mask is not None:
                        cur_pts = torch.cat([xyz + d_xyz, torch.ones_like(xyz[..., :1])], dim=-1)[self.segmented_mask].cuda()
                    else:
                        cur_pts = torch.cat([xyz + d_xyz, torch.ones_like(xyz[..., :1])], dim=-1).cuda()
            else:
                if self.is_6dof:
                    cur_pts = torch.cat([from_homogenous(
                                torch.bmm(d_xyz[motion_segmentation_mask], to_homogenous(xyz[motion_segmentation_mask]).unsqueeze(-1)).squeeze(-1)), torch.ones_like(xyz[motion_segmentation_mask, :1])], dim=-1).cuda()
                else:
                    cur_pts = torch.cat([xyz[motion_segmentation_mask] + d_xyz[motion_segmentation_mask], torch.ones_like(xyz[motion_segmentation_mask, :1])], dim=-1).cuda()
            cur_pts2d = cur_pts @ cur_cam.full_proj_transform.cuda()
            cur_pts2d = cur_pts2d[..., :2] / cur_pts2d[..., -1:]
            cur_pts2d = (cur_pts2d + 1) / 2 * torch.tensor([cur_cam.image_height, cur_cam.image_width]).cuda()

            buffer_image = torch.zeros(size=(3, cur_cam.image_height, cur_cam.image_width)).cuda() if not self.dataset.white_background else torch.ones(size=(3, cur_cam.image_height, cur_cam.image_width)).cuda()

            mask_1 = (cur_pts2d[:, 0] > 0) & (cur_pts2d[:, 0] < cur_cam.image_width)
            mask_2 = (cur_pts2d[:, 1] > 0) & (cur_pts2d[:, 1] < cur_cam.image_height)
            final_mask = mask_1 & mask_2
            
            if self.mode != 'Gaussian Clusters':
                if self.segmented_mask is not None:
                    gaussians_feature_pca = self.gaussians_feature_pca[self.segmented_mask]
                else:
                    gaussians_feature_pca = self.gaussians_feature_pca
                    
                if not self.dataset.white_background:
                    buffer_image[0, (cur_pts2d[final_mask, 1]).type(torch.long), (cur_pts2d[final_mask, 0]).type(torch.long)] = 1 if self.mode == 'Point Cloud' else gaussians_feature_pca[final_mask, 0]
                    buffer_image[1, (cur_pts2d[final_mask, 1]).type(torch.long), (cur_pts2d[final_mask, 0]).type(torch.long)] = 1 if self.mode == 'Point Cloud' else gaussians_feature_pca[final_mask, 1]
                    buffer_image[2, (cur_pts2d[final_mask, 1]).type(torch.long), (cur_pts2d[final_mask, 0]).type(torch.long)] = 1 if self.mode == 'Point Cloud' else gaussians_feature_pca[final_mask, 2]
                else:
                    buffer_image[0, (cur_pts2d[final_mask, 1]).type(torch.long), (cur_pts2d[final_mask, 0]).type(torch.long)] = 0 if self.mode == 'Point Cloud' else gaussians_feature_pca[final_mask, 0]
                    buffer_image[1, (cur_pts2d[final_mask, 1]).type(torch.long), (cur_pts2d[final_mask, 0]).type(torch.long)] = 0 if self.mode == 'Point Cloud' else gaussians_feature_pca[final_mask, 1]
                    buffer_image[2, (cur_pts2d[final_mask, 1]).type(torch.long), (cur_pts2d[final_mask, 0]).type(torch.long)] = 0 if self.mode == 'Point Cloud' else gaussians_feature_pca[final_mask, 2]
            else:
                if self.segmented_mask is not None:
                    cls_colors = self.cluster_point_colors[self.segmented_mask]
                else:
                    cls_colors = self.cluster_point_colors
        
                buffer_image[0, (cur_pts2d[final_mask, 1]).type(torch.long), (cur_pts2d[final_mask, 0]).type(torch.long)] = cls_colors[final_mask, 0]
                buffer_image[1, (cur_pts2d[final_mask, 1]).type(torch.long), (cur_pts2d[final_mask, 0]).type(torch.long)] = cls_colors[final_mask, 1]
                buffer_image[2, (cur_pts2d[final_mask, 1]).type(torch.long), (cur_pts2d[final_mask, 0]).type(torch.long)] = cls_colors[final_mask, 2]
        
        elif self.mode == 'Segmentation':
            if self.cluster_point_colors is not None:
                if self.segmented_mask is not None:
                    buffer_image = render(viewpoint_camera=cur_cam, pc=gaussians, pipe=self.pipe, bg_color=self.background, d_xyz=d_xyz, d_rotation=d_rotation, d_scaling=d_scaling, is_6dof=self.is_6dof, override_color=self.cluster_point_colors.cuda().float(), mask=self.segmented_mask)["render"]
                else:
                    buffer_image = render(viewpoint_camera=cur_cam, pc=gaussians, pipe=self.pipe, bg_color=self.background, d_xyz=d_xyz, d_rotation=d_rotation, d_scaling=d_scaling, is_6dof=self.is_6dof, override_color=self.cluster_point_colors.cuda().float())["render"]
        else:
            if self.text_prompt != '':
                out = render(viewpoint_camera=cur_cam, pc=gaussians, pipe=self.pipe, bg_color=self.background, d_xyz=d_xyz, d_rotation=d_rotation, d_scaling=d_scaling, is_6dof=self.is_6dof)
                self.groundingdino_model = self.groundingdino_model.to('cuda')
                self.sam_predictor.model = self.sam_predictor.model.to('cuda')
                text_mask, _ = grouned_sam_output(self.groundingdino_model, self.sam_predictor, self.text_prompt, to8b(out['render']).transpose(1,2,0))
                self.groundingdino_model = self.groundingdino_model.to('cpu')
                self.sam_predictor.model = self.sam_predictor.model.to('cpu')
                depth = out["depth"]
                depth = depth.squeeze()
                grid_index = generate_grid_index(depth).cuda()
                z = cur_cam.zfar / (cur_cam.zfar - cur_cam.znear) * depth[text_mask] - cur_cam.zfar * cur_cam.znear / (cur_cam.zfar - cur_cam.znear)
                uvz = torch.cat(((((grid_index[text_mask, :][:, 1] - 0.5) / cur_cam.image_width * 2 - 1) * depth[text_mask]).unsqueeze(-1),
                                (((grid_index[text_mask, :][:, 0] - 0.5) / cur_cam.image_height * 2 - 1) * depth[text_mask]).unsqueeze(-1),
                                z.unsqueeze(-1),
                                depth[text_mask].unsqueeze(-1)), 1)
                text_masked_points_in_3D = uvz @ (torch.inverse(cur_cam.full_proj_transform))[:, :3]
                knn_obj = ops.knn_points(
                    text_masked_points_in_3D.unsqueeze(0),
                    (xyz + d_xyz).detach().unsqueeze(0),
                    K=1,
                )
                ijs = knn_obj.idx.squeeze(0).squeeze(-1)
                threshold = dpg.get_value('_text_prompt_threshold')
                text_masked_points_cls = self.cluster_ids_x[ijs].int()
                text_masked_cls_id = torch.where(torch.bincount(text_masked_points_cls) > threshold, 1, 0).nonzero()
                print("Text prompt cls id: ", text_masked_cls_id)
                self.segmented_mask = None
                for id in text_masked_cls_id:
                    if self.segmented_mask is None:
                        self.segmented_mask = (self.cluster_ids_x == id)
                    else:
                        self.segmented_mask |= (self.cluster_ids_x == id)
                self.text_prompt = ''
                        
            if self.segmented_mask is None:
                out = render(viewpoint_camera=cur_cam, pc=gaussians, pipe=self.pipe, bg_color=self.background, d_xyz=d_xyz, d_rotation=d_rotation, d_scaling=d_scaling, is_6dof=self.is_6dof)
                self.rendered_cluster = None if self.cluster_point_colors is None else render(viewpoint_camera=cur_cam, pc=gaussians, pipe=self.pipe, bg_color=self.background, d_xyz=d_xyz, d_rotation=d_rotation, d_scaling=d_scaling, is_6dof=self.is_6dof, override_color=self.cluster_point_colors.cuda().float())["render"].permute(1, 2, 0)  
            else:
                out = render(viewpoint_camera=cur_cam, pc=gaussians, pipe=self.pipe, bg_color=self.background, d_xyz=d_xyz, d_rotation=d_rotation, d_scaling=d_scaling, is_6dof=self.is_6dof, mask=self.segmented_mask) if not self.remove_selceted else render(viewpoint_camera=cur_cam, pc=gaussians, pipe=self.pipe, bg_color=self.background, d_xyz=d_xyz, d_rotation=d_rotation, d_scaling=d_scaling, is_6dof=self.is_6dof, mask=~self.segmented_mask)
                
            buffer_image = out['render']
            if self.mode == 'Render':
                buffer_image = out['render']  # [3, H, W]
            elif self.mode == 'Rendered Features':
                if self.segmented_mask is None:
                    buffer_image = render(cur_cam, gaussians, pipeline, self.background, d_xyz, d_rotation, d_scaling, self.is_6dof, override_color=self.gaussians_feature_pca)['render']
                else:
                    buffer_image = render(viewpoint_camera=cur_cam, pc=gaussians, pipe=self.pipe, bg_color=self.background, d_xyz=d_xyz, d_rotation=d_rotation, d_scaling=d_scaling, is_6dof=self.is_6dof, override_color=self.gaussians_feature_pca, mask=self.segmented_mask)['render'] if not self.remove_selceted else render(viewpoint_camera=cur_cam, pc=gaussians, pipe=self.pipe, bg_color=self.background, d_xyz=d_xyz, d_rotation=d_rotation, d_scaling=d_scaling, is_6dof=self.is_6dof, override_color=self.gaussians_feature_pca, mask=~self.segmented_mask)['render']
            elif self.mode == 'Depth':
                buffer_image = out['depth']
                buffer_image = buffer_image.repeat(3, 1, 1)
                buffer_image = (buffer_image - buffer_image.min()) / (buffer_image.max() - buffer_image.min() + 1e-20)
        
        buffer_image = torch.nn.functional.interpolate(
            buffer_image.unsqueeze(0),
            size=(self.H, self.W),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        self.buffer_image = (
            buffer_image.permute(1, 2, 0)
            .contiguous()
            .clamp(0, 1)
            .contiguous()
            .detach()
            .cpu()
            .numpy()
        )

        self.need_update = True

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        if self.is_animation and self.buffer_overlay is not None:
            overlay_mask = self.buffer_overlay.sum(axis=-1, keepdims=True) == 0
            try:
                buffer_image = self.buffer_image * overlay_mask + self.buffer_overlay
            except:
                buffer_image = self.buffer_image
        else:
            buffer_image = self.buffer_image

        if self.vis_traj_realtime:
            buffer_image = buffer_image * (1 - self.traj_overlay[..., 3:]) + self.traj_overlay[..., :3] * self.traj_overlay[..., 3:]

        if self.mode == 'Render':    
            if self.rendered_cluster is not None and self.segmented_mask is None and self.render_segmentation_mask:
                buffer_image += 0.3 * self.rendered_cluster.cpu().numpy()
        
        dpg.set_value("_log_infer_time", f"{t:.4f}ms ({int(1000/t)} FPS FID: {fid.item()})")
        dpg.set_value(
            "_texture", buffer_image
        )  # buffer must be contiguous, else seg fault!
        return buffer_image
    
    def update_control_point_overlay(self):
        from skimage.draw import line_aa
        if self.need_update_overlay:
            try:
                buffer_overlay = np.zeros_like(self.buffer_image)
                mv = self.cam.view # [4, 4]
                proj = self.cam.perspective # [4, 4]
                mvp = proj @ mv
                source_points = self.deformed_pcd_at_t.detach().cpu().numpy()[np.array(self.selected_point_idcs).flatten()]

                source_points_clip = np.matmul(np.pad(source_points, ((0, 0), (0, 1)), constant_values=1.0), mvp.T)  # [N, 4]
                source_points_clip[:, :3] /= source_points_clip[:, 3:] # perspective division

                source_points_2d = (((source_points_clip[:, :2] + 1) / 2) * np.array([self.H, self.W])).round().astype(np.int32)

                radius = int((self.H + self.W) / 2 * 0.005)
                for i in range(len(source_points_clip)):
                    # draw source point
                    if source_points_2d[i, 0] >= radius and source_points_2d[i, 0] < self.W - radius and source_points_2d[i, 1] >= radius and source_points_2d[i, 1] < self.H - radius:
                        buffer_overlay[source_points_2d[i, 1]-radius:source_points_2d[i, 1]+radius, source_points_2d[i, 0]-radius:source_points_2d[i, 0]+radius] += np.array([1,0,0])
                self.buffer_overlay = buffer_overlay
            except:
                self.buffer_overlay = None
                
    def update_trajectory_overlay(self, gs_xyz, camera, samp_num=32, gs_num=512, thickness=1, segmentation_mask=None):
        if not hasattr(self, 'traj_coor') or self.traj_coor is None:
            from utils.time_utils import farthest_point_sample
            self.traj_coor = torch.zeros([0, gs_num, 4], dtype=torch.float32).cuda()
            if segmentation_mask is None:
                opacity_mask = self.gaussians.get_opacity[..., 0] > .1 if self.gaussians.get_xyz.shape[0] == gs_xyz.shape[0] else torch.ones_like(gs_xyz[:, 0], dtype=torch.bool)
                masked_idx = torch.arange(0, opacity_mask.shape[0], device=opacity_mask.device)[opacity_mask]
                self.traj_idx = masked_idx[farthest_point_sample(gs_xyz[None, opacity_mask], gs_num)[0]]
            else:
                print("Visualize Trajectory with mask")
                opacity_mask = self.gaussians.get_opacity[segmentation_mask, 0] > .1 if self.gaussians.get_xyz[segmentation_mask].shape[0] == gs_xyz[segmentation_mask].shape[0] else torch.ones_like(gs_xyz[segmentation_mask, 0], dtype=torch.bool)
                masked_idx = torch.arange(0, opacity_mask.shape[0], device=opacity_mask.device)[opacity_mask]
                self.traj_idx = masked_idx[farthest_point_sample(gs_xyz[segmentation_mask][opacity_mask].unsqueeze(0), gs_num)[0]]  
            from matplotlib import cm
            self.traj_color_map = cm.get_cmap("jet")
        if segmentation_mask is None:
            pts = gs_xyz[None, self.traj_idx]
        else:
            pts = gs_xyz[segmentation_mask][None, self.traj_idx]
        pts = torch.cat([pts, torch.ones_like(pts[..., :1])], dim=-1)
        self.traj_coor = torch.cat([self.traj_coor, pts], axis=0)
        if self.traj_coor.shape[0] > samp_num:
            self.traj_coor = self.traj_coor[-samp_num:]
        traj_uv = self.traj_coor @ camera.full_proj_transform
        traj_uv = traj_uv[..., :2] / traj_uv[..., -1:]
        traj_uv = (traj_uv + 1) / 2 * torch.tensor([camera.image_height, camera.image_width]).cuda()
        traj_uv = traj_uv.detach().cpu().numpy()

        import cv2
        colors = np.array([np.array(self.traj_color_map(i/max(1, float(gs_num - 1)))[:3]) * 255 for i in range(gs_num)], dtype=np.int32)
        alpha_img = np.zeros([camera.image_height, camera.image_width, 3], dtype=np.float32)
        traj_img = np.zeros([camera.image_height, camera.image_width, 3], dtype=np.float32)
        for i in range(gs_num):            
            alpha_img = cv2.polylines(img=alpha_img, pts=[traj_uv[:, i].astype(np.int32)], isClosed=False, color=[1, 1, 1], thickness=thickness)
            color = colors[i] / 255
            traj_img = cv2.polylines(img=traj_img, pts=[traj_uv[:, i].astype(np.int32)], isClosed=False, color=[float(color[0]), float(color[1]), float(color[2])], thickness=thickness)
        traj_img = np.concatenate([traj_img, alpha_img[..., :1]], axis=-1)
        self.traj_overlay = traj_img


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    
    parser.add_argument("--iteration", default=-1, type=int)
    # GUI
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=800, help="GUI width")
    parser.add_argument('--H', type=int, default=800, help="GUI height")
    parser.add_argument('--elevation', type=float, default=0, help="default GUI camera elevation")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")    
    
    parser.add_argument("--quiet", action="store_true")
    
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    safe_state(args.quiet)

    gui = GUI(args=args, dataset=model.extract(args), pipe=pipeline.extract(args), iteration=args.iteration, opt=op.extract(args))

    gui.render()
    
    # All done
    print("\nTraining complete.")
