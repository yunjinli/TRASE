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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.rigid_utils import from_homogenous, to_homogenous

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack((w, x, y, z), dim=-1)

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, 
           d_xyz, d_rotation, d_scaling, is_6dof=False, ## for deformation
           scaling_modifier = 1.0, override_color = None, 
           mask = None, norm_gaussian_features = True, is_smooth_gaussian_features = False, smooth_K = 16):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    if is_6dof:
        if torch.is_tensor(d_xyz) is False:
            means3D = pc.get_xyz
        else:
            means3D = from_homogenous(
                torch.bmm(d_xyz, to_homogenous(pc.get_xyz).unsqueeze(-1)).squeeze(-1))
    else:
        means3D = pc.get_xyz + d_xyz
            
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling + d_scaling
        rotations = pc.get_rotation + d_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
            # sh_objs = pc.get_point_features
    else:
        colors_precomp = override_color

    if not is_smooth_gaussian_features:
        sh_objs = pc.get_gaussian_features
    else:
        sh_objs = pc.get_smoothed_gaussian_features(K=smooth_K, dropout=0.5)
    
    if norm_gaussian_features:
        sh_objs = sh_objs / (sh_objs.norm(dim=2, keepdim=True) + 1e-9)
    
    if mask is not None:
        means3D = means3D[mask]
        means2D = means2D[mask]
        opacity = opacity[mask]
        if colors_precomp is not None:
            colors_precomp = colors_precomp[mask]
        else:
            shs = shs[mask]
        sh_objs = sh_objs[mask]
        scales = scales[mask]
        rotations = rotations[mask]
        if cov3D_precomp is not None:
            cov3D_precomp = cov3D_precomp[mask]
        
    rendered_image, radii, rendered_feats, depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        sh_objs = sh_objs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "render_gaussian_features": rendered_feats,
            "depth": depth}


def rotmat2qvec(R):
    ## from https://github.com/guanjunwu/sa4d
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flatten()
    K = torch.tensor([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]).to(R) / 3.0
    eigvals, eigvecs = torch.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], torch.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def rx(theta):
    ## from https://github.com/guanjunwu/sa4d
    return torch.tensor([[1, 0, 0],
                        [0, torch.cos(theta), -torch.sin(theta)],
                        [0, torch.sin(theta), torch.cos(theta)]])

def ry(theta):
    ## from https://github.com/guanjunwu/sa4d
    return torch.tensor([[torch.cos(theta), 0, torch.sin(theta)],
                        [0, 1, 0],
                        [-torch.sin(theta), 0, torch.cos(theta)]])

def rz(theta):
    ## from https://github.com/guanjunwu/sa4d
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                        [torch.sin(theta), torch.cos(theta), 0],
                        [0, 0, 1]])
    
def rescale(means3d, scales, scale_factor: float):
    ## from https://github.com/guanjunwu/sa4d
    means3d = means3d * scale_factor
    scales = scales * scale_factor
    return means3d, scales
        
def rotate_by_euler_angles(means3d, rotations, rotation_angles):
    ## from https://github.com/guanjunwu/sa4d
    """
    rotate in z-y-x order, radians as unit
    """
    x, y, z = rotation_angles

    if x == 0. and y == 0. and z == 0.:
        return means3d, rotations

    rotation_matrix = torch.tensor(rx(x) @ ry(y) @ rz(z), dtype=torch.float32).to(rotations)

    return rotate_by_matrix(means3d, rotations, rotation_matrix)
    
def rotate_by_matrix(means3d, rotations, rotation_matrix, keep_sh_degree: bool = True):
    ## from https://github.com/guanjunwu/sa4d
    # rotate xyz
    means3d = torch.tensor(torch.matmul(means3d, rotation_matrix.T))

    # rotate gaussian
    # rotate via quaternions
    def quat_multiply(quaternion0, quaternion1):
        w0, x0, y0, z0 = torch.split(quaternion0, 1, dim=-1)
        w1, x1, y1, z1 = torch.split(quaternion1, 1, dim=-1)
        return torch.concatenate((
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
        ), dim=-1)

    quaternions = rotmat2qvec(rotation_matrix)[None, ...]
    rotations_from_quats = quat_multiply(rotations, quaternions)
    rotations = rotations_from_quats / torch.linalg.norm(rotations_from_quats, dim=-1, keepdims=True)

    if keep_sh_degree is False:
        print("set sh_degree=0 when rotation transform enabled")
        sh_degrees = 0
        
    return means3d, rotations

def translation(means3d, offsets):
    ## from https://github.com/guanjunwu/sa4d
    means3d += offsets 
    
    return means3d
    
def transform(means3d, rotations, scales, scale_factor, offsets, rotation_angles):
    ## from https://github.com/guanjunwu/sa4d
    means3d, scales = rescale(means3d, scales, scale_factor)
    means3d, rotations = rotate_by_euler_angles(means3d, rotations, rotation_angles)
    means3d = translation(means3d, offsets)
    
    return means3d, rotations, scales

def render_composite(viewpoint_camera, background_gaussian : GaussianModel, 
                     dynamic_gaussian : GaussianModel, 
                     d_xyz, d_rotation, d_scaling,
                     bg_color : torch.Tensor, 
                     scales_bias, motion_bias, rotation_bias,
                    scaling_modifier = 1.0, mask = None):
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points_bg = torch.zeros_like(background_gaussian.get_xyz, dtype=background_gaussian.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points_dynamic = torch.zeros_like(dynamic_gaussian.get_xyz, dtype=dynamic_gaussian.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points = torch.cat([screenspace_points_bg, screenspace_points_dynamic], dim=0)
    
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=background_gaussian.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means3D_final, scales_final, rotations_final, opacity_final, shs_final, shs_obj_final = background_gaussian.get_xyz, background_gaussian.get_scaling, background_gaussian.get_rotation, background_gaussian.get_opacity, background_gaussian.get_features, background_gaussian.get_gaussian_features

    means3D_dynamic = dynamic_gaussian.get_xyz + d_xyz
    scales_dynamic = dynamic_gaussian.get_scaling + d_scaling
    rotations_dynamic = dynamic_gaussian.get_rotation + d_rotation
    shs_dynamic = dynamic_gaussian.get_features
    opacity_dynamic = dynamic_gaussian.get_opacity
    sh_objs_dynamic = dynamic_gaussian.get_gaussian_features
    
    # cov3D_precomp
    if mask is not None:
        means3D_dynamic = means3D_dynamic[mask]
        opacity_dynamic = opacity_dynamic[mask]
        shs_dynamic = shs_dynamic[mask]
        sh_objs_dynamic = sh_objs_dynamic[mask]
        scales_dynamic = scales_dynamic[mask]
        rotations_dynamic = rotations_dynamic[mask]
            
    means3D_dynamic, rotations_dynamic, scales_dynamic = transform(means3D_dynamic, rotations_dynamic, scales_dynamic, scales_bias, motion_bias, rotation_bias)
    
    means3D_final = torch.cat([means3D_final, means3D_dynamic], dim=0)
    scales_final = torch.cat([scales_final, scales_dynamic], dim=0)
    rotations_final = torch.cat([rotations_final, rotations_dynamic], dim=0)
    opacity_final = torch.cat([opacity_final, opacity_dynamic], dim=0)
    shs_final = torch.cat([shs_final, shs_dynamic], dim=0)
    shs_obj_final = torch.cat([shs_obj_final, sh_objs_dynamic], dim=0)
    
    colors_precomp = None
            
    means2D = screenspace_points
    colors_precomp = None
    cov3D_precomp = None
    
    rendered_image, _, _, _ = rasterizer(
        means3D = means3D_final,
        means2D = means2D,
        shs = shs_final,
        sh_objs = shs_obj_final,
        colors_precomp = colors_precomp,
        opacities = opacity_final,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = cov3D_precomp)

    return {"render": rendered_image}

