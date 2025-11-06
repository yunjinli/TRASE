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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.deform_model import DeformModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], load_object=None):
        """
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            print("[Warning] Assuming loaing from pretrained 3DGS on Mip-NeRF 360 scene...")
            scene_info = sceneLoadTypeCallbacks["Colmap"](path=args.source_path, 
                                                          images=args.images, 
                                                          eval=args.eval, 
                                                          load_image_on_the_fly=args.load_image_on_the_fly, 
                                                          load_mask_on_the_fly=args.load_mask_on_the_fly)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Multi-View data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](path=args.source_path, 
                                                           white_background=args.white_background, 
                                                           eval=args.eval, 
                                                           load_image_on_the_fly=args.load_image_on_the_fly, 
                                                           load_mask_on_the_fly=args.load_mask_on_the_fly, 
                                                           end_frame=args.end_frame
                                                           )
        elif os.path.exists(os.path.join(args.source_path, "dataset.json")):
            print("Found dataset.json file, assuming Nerfies data set!")
            scene_info = sceneLoadTypeCallbacks["nerfies"](path=args.source_path, 
                                                           eval=args.eval, 
                                                           load_image_on_the_fly=args.load_image_on_the_fly, 
                                                           load_mask_on_the_fly=args.load_mask_on_the_fly)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            if isinstance(self.loaded_iter,str):
                print("edit load path", self.loaded_iter)
                if load_object:
                    self.gaussians.load_ply(os.path.join(self.model_path,
                                                            "point_cloud"+self.loaded_iter,
                                                            f"point_cloud_{load_object}.ply"), spatial_lr_scale=self.cameras_extent)
                else:
                    self.gaussians.load_ply(os.path.join(self.model_path,
                                                                "point_cloud"+self.loaded_iter,
                                                                "point_cloud.ply"), spatial_lr_scale=self.cameras_extent)
            else:
                if load_object:
                    self.gaussians.load_ply(os.path.join(self.model_path,
                                                            "point_cloud",
                                                            "iteration_" + str(self.loaded_iter),
                                                            f"point_cloud_{load_object}.ply"), spatial_lr_scale=self.cameras_extent)
                else:
                    self.gaussians.load_ply(os.path.join(self.model_path,
                                                                "point_cloud",
                                                                "iteration_" + str(self.loaded_iter),
                                                                "point_cloud.ply"), spatial_lr_scale=self.cameras_extent)
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration, is_smooth_gaussian_features=False, smooth_K=16):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"), is_smooth_gaussian_features=is_smooth_gaussian_features, smooth_K=smooth_K)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]