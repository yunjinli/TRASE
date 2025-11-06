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
from random import randint
from utils.loss_utils import l1_loss, ssim, loss_nnfm_style
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel, DeformModel
from utils.general_utils import safe_state, get_linear_noise_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import pytorch3d.ops as ops
from kmeans_pytorch import kmeans
import math
from utils.time_utils import farthest_point_sample
from style_transfer.fx import VGG16FeatureExtractor
from PIL import Image
from utils.general_utils import PILtoTorch

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
from enum import Enum

class OPT_STATE_NAME(Enum):
    GAUSSIAN = 1
    FEATURE = 2

class OPT_STATE:
    def __init__(self, max_iterations):
        self.state = OPT_STATE_NAME.GAUSSIAN.name
        self.iterations = 0
        self.max_iterations = max_iterations
        
    def step(self):
        self.iterations += 1
        
    def switch(self):
        is_switch = False
        if self.iterations > self.max_iterations:
            if self.state == OPT_STATE_NAME.GAUSSIAN.name:
                self.state = OPT_STATE_NAME.FEATURE.name
            else:
                self.state = OPT_STATE_NAME.GAUSSIAN.name
            self.iterations = 0
            is_switch = True
        return is_switch
    
    
def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, load_iteration, segment_ids, reference_img_path):
    print("===== SADG: Style Transfer =====")
    
    opt_state = OPT_STATE(max_iterations=opt.iterative_opt_interval)
    
    if load_iteration == -1:
        print("[ERROR] Please load a pretrained scene!!!")
        return
    else:
        print("Load from: ", load_iteration)
        first_iter = load_iteration
        tb_writer = prepare_output_and_logger(dataset)
        ## Gaussian model
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=load_iteration)
        
        gaussians.load_clusters(os.path.join(dataset.model_path, f"point_cloud/iteration_{load_iteration}/clusters.pt"))
        gaussians.set_style_transfer_mode()
        gaussians.set_style_object_mask(segment_ids)
        gaussians.training_setup(opt)
        gaussians.change_optimization_target(opt_state=opt_state.state)
        ## Deformation model
        deform = DeformModel(dataset.is_blender, dataset.is_6dof)
        deform.load_weights(dataset.model_path, iteration=load_iteration)
        deform.train_setting(opt)
        fx_keys = ['conv4_1']
        vgg_ext = VGG16FeatureExtractor(fx_keys).cuda()
        ref_style_pil = Image.open(reference_img_path)
        ref_style_img = PILtoTorch(ref_style_pil, ref_style_pil.size)
        
        ref_style_img_norm = vgg_ext.normalize(ref_style_img).cuda()
        print(ref_style_img_norm.shape)
        
        
        
        
    smooth_weights = None
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    
    ## From deformation
    best_psnr = 0.0
    best_iteration = 0
    
    cluster_ids_x = gaussians.get_clusters['id']
    segmented_mask = None
    if segment_ids != -1:
        for id in segment_ids:
            if segmented_mask is None:
                segmented_mask = (cluster_ids_x == id)
            else:
                segmented_mask |= (cluster_ids_x == id)
    segmented_mask = segmented_mask.squeeze(-1)          
    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()
        
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if iteration < opt.warm_up_3d_features:
            ## Optimization should on only for the gaussians
            pass
        else:
            if(opt_state.switch()):
                gaussians.change_optimization_target(opt_state=opt_state.state)
                deform.change_optimization_target(opt_state=opt_state.state)
                print(f"Change to mode {opt_state.state}, reset camera stack...")
                viewpoint_stack = scene.getTrainCameras().copy()
                
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        
        # Setting time variable for deformation field
        total_frame = len(viewpoint_stack)
        time_interval = 1 / total_frame
        
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()        
         
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        
        ## Get deformation dxyz from the network
        fid = viewpoint_cam.fid
        if iteration < opt.warm_up:
            d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
        else:
            N = gaussians.get_xyz.shape[0]
            time_input = fid.unsqueeze(0).expand(N, -1)

            with torch.no_grad():
                d_xyz, d_rotation, d_scaling = deform.step(gaussians.get_xyz.detach(), time_input)
            

        Ll1 = None
        
                        
        if opt_state.state == OPT_STATE_NAME.GAUSSIAN.name:
            render_pkg = render(viewpoint_camera=viewpoint_cam, pc=gaussians, pipe=pipe, bg_color=background, d_xyz=d_xyz, d_rotation=d_rotation, d_scaling=d_scaling, is_6dof=dataset.is_6dof)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            if pipe.debug:
                opencvImage = cv2.cvtColor(image.permute(1, 2, 0).detach().cpu().numpy(), cv2.COLOR_RGB2BGR)
                cv2.imshow('Training View', opencvImage)
                if cv2.waitKey(1) == ord('q'):
                    break
            image_norm = vgg_ext.normalize(image)
            
            rendered_vgg_feats = vgg_ext(image_norm)[fx_keys[0]].squeeze(0)
            ref_vgg_feats = vgg_ext(ref_style_img_norm)[fx_keys[0]].squeeze(0)
            
            
            loss = loss_nnfm_style(rendered_vgg_feats.view(rendered_vgg_feats.shape[0], -1),
                                   ref_vgg_feats.view(ref_vgg_feats.shape[0], -1))
            

        if not torch.isnan(loss):
            loss.backward()
            gaussians.set_background_zero_grad(segmented_mask=segmented_mask)
        else:
            print("NaN loss detected!!!")
            
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            if not torch.isnan(loss):
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            else:
                ema_loss_for_log = ema_loss_for_log
            if iteration % 10 == 0:
                show_dict = {
                    "Loss": f"{ema_loss_for_log:.{3}f}", 
                    "State": f'{opt_state.state}',
                    "Points": f'{gaussians.get_xyz.shape[0]}'
                }
                
                if opt.monitor_mem:
                    show_dict["CUDA"] = f'{(torch.cuda.max_memory_allocated(device=None) / (1024 * 1024 * 1024)):.1f} GB'
                    
                    
                progress_bar.set_postfix(show_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            cur_psnr = training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                                       testing_iterations, scene, render, None, (pipe, background), deform,
                                       dataset.load2gpu_on_the_fly, dataset.is_6dof)
            if iteration in testing_iterations:
                if cur_psnr.item() > best_psnr:
                    best_psnr = cur_psnr.item()
                    best_iteration = iteration
                    
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, smooth_weights = torch.softmax(smooth_weights, dim = -1) if smooth_weights is not None else None, smooth_type = 'traditional', smooth_K = opt.smooth_K)
                
                deform.save_weights(args.model_path, iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                # gaussians.set_d_scaling(d_scaling=d_scaling)
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                ## Optimizer s1tep
                if not torch.isnan(loss):
                    gaussians.optimizer[opt_state.state].step()
                ## Zero the grads of the optimizer
                gaussians.optimizer[OPT_STATE_NAME.GAUSSIAN.name].zero_grad(set_to_none = True)
                gaussians.optimizer[OPT_STATE_NAME.FEATURE.name].zero_grad(set_to_none = True)
                gaussians.update_learning_rate(iteration, opt_state.state)
            if dataset.load2gpu_on_the_fly:
                viewpoint_cam.load2device('cpu')
             
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
    print("Best PSNR = {} in Iteration {}".format(best_psnr, best_iteration))
    
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc, loss_obj_3d,
                    renderArgs, deform, load2gpu_on_the_fly, is_6dof=False):
    if tb_writer:
        if Ll1:
            tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        if loss_obj_3d:
            tb_writer.add_scalar('train_loss_patches/loss_obj_3d', loss_obj_3d.item(), iteration)

    test_psnr = 0.0
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                images = torch.tensor([], device="cuda")
                gts = torch.tensor([], device="cuda")
                for idx, viewpoint in enumerate(config['cameras']):
                    if load2gpu_on_the_fly:
                        viewpoint.load2device()
                    fid = viewpoint.fid
                    xyz = scene.gaussians.get_xyz
                    time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
                    d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs, d_xyz, d_rotation, d_scaling, is_6dof)["render"],
                        0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    images = torch.cat((images, image.unsqueeze(0)), dim=0)
                    gts = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)

                    if load2gpu_on_the_fly:
                        viewpoint.load2device('cpu')
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)

                l1_test = l1_loss(images, gts)
                psnr_test = psnr(images, gts).mean()
                if config['name'] == 'test' or len(validation_configs[0]['cameras']) == 0:
                    test_psnr = psnr_test
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

    return test_psnr

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1_000, 7_000, 30_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1_000, 7_000, 30_000, 60_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument('--load_iteration', type=int, default=-1)
    
    parser.add_argument('--segment_ids', type=int, nargs='+', default=-1)
    parser.add_argument("--reference_img_path", type=str, default=None)
    
    

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    
    print("Optimizing " + args.model_path)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.load_iteration, args.segment_ids, args.reference_img_path)

    # All done
    print("\nTraining complete.")
