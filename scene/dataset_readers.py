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
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from utils.camera_utils import camera_nerfies_from_JSON
import torch
from glob import glob
from typing import Union
from tqdm import tqdm
from multiprocessing.pool import ThreadPool

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    masks: Union[torch.tensor, dict]
    mask_path: str
    fid: float
    
class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info, apply=False):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal
    cam_centers = []
    if apply:
        c2ws = []
    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        if apply:
            c2ws.append(C2W)
        cam_centers.append(C2W[:3, 3:4])
    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal
    translate = -center
    if apply:
        c2ws = np.stack(c2ws, axis=0)
        c2ws[:, :3, -1] += translate
        c2ws[:, :3, -1] /= radius
        w2cs = np.linalg.inv(c2ws)
        for i in range(len(cam_info)):
            cam = cam_info[i]
            cam_info[i] = cam._replace(R=w2cs[i, :3, :3].T, T=w2cs[i, :3, 3])
        apply_translate = translate
        apply_radius = radius
        translate = 0
        radius = 1.
        return {"translate": translate, "radius": radius, "apply_translate": apply_translate, "apply_radius": apply_radius}
    else:
        return {"translate": translate, "radius": radius}

def translate_cam_info(cam_info, translate):
    for i in range(len(cam_info)):
        cam = cam_info[i]
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        C2W[:3, 3] += translate
        W2C = np.linalg.inv(C2W)
        cam_info[i] = cam._replace(R=W2C[:3, :3].T, T=W2C[:3, 3])
        
def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, source_path=None, load_image_on_the_fly=False):
    cam_infos = []
    num_frames = len(cam_extrinsics)
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE" or intr.model == "OPENCV" or intr.model == "SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path) if os.path.exists(image_path) else None
        width, height = image.size[0], image.size[1]
        
        if load_image_on_the_fly:
            image = None
            
        try:
            masks_path = os.path.join(source_path, 'masks', image_name + '.pt')
            masks = torch.load(masks_path).to('cpu')
        except:
            print("[Warning] Mask path not specified")
            masks_path = None
            masks = None
        try:
            fid = int(image_name) / (num_frames - 1)
        except:
            print("[Warning] Assume static scene")
            fid = 0.0
            
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, masks=masks, fid=fid, objects=None, mask_path=None)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapStaticSceneInfo(path, images, eval, llffhold=8, load_image_on_the_fly=False, load_mask_on_the_fly=False):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    print(os.path.join(path, reading_dir))
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), load_image_on_the_fly=load_image_on_the_fly)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", load_image_on_the_fly=False, load_mask_on_the_fly=False, end_frame=None):
    cam_infos = []
    
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
    if "camera_angle_x" in contents:
        print("This is Blender dataset")
        dataset_type = 'blender'
        fovx = contents["camera_angle_x"]
        time_duration = None
    elif 'fl_x' in contents and 'fl_y' in contents and 'cx' in contents and 'cy' in contents:
        print("This is Neu3D dataset")
        dataset_type = 'neu3d'
        time_duration = 10.0
    elif 'technicolor' in path:
        print("This is Technicolor dataset")
        dataset_type = 'technicolor'
        ## We downsample the dataset during preprocessing already
        time_duration = 10.0 / 6.0
    else:
        print("This is Google Immersive dataset")
        dataset_type = 'immersive'
        # time_duration = 10.0 / 6.0
        time_duration = 10.0
        
    frames = contents["frames"]
    tbar = tqdm(range(len(frames)))
    def frame_read_fn(idx_frame):
        idx = idx_frame[0]
        frame = idx_frame[1]
        
        fid = int(frame['file_path'].split('/')[-1][-4:])
        frame_time = frame['time']
        # if time_duration:
        #     frame_time /= time_duration
        if time_duration:
            if end_frame != -1:
                frame_time /= (end_frame / 300.0) * 10.0
                if fid > end_frame:
                    return None
            else:
                frame_time /= time_duration
                
        cam_name = os.path.join(path, frame["file_path"] + extension)

        if dataset_type == 'immersive' or dataset_type == 'technicolor':
            w2c = np.array(frame["transform_matrix"])
        else:
            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1
            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        image_path = os.path.join(path, cam_name) # .replace('hdImgs_unditorted', 'hdImgs_unditorted_rgba').replace('.jpg', '.png')
        image_name = Path(cam_name).stem
        
        if not load_image_on_the_fly:
            with Image.open(image_path) as image_load:
                im_data = np.array(image_load.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            if norm_data[:, :, 3:4].min() < 1:
                arr = np.concatenate([arr, norm_data[:, :, 3:4]], axis=2)
                image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGBA")
            else:
                image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            width, height = image.size[0], image.size[1]
        else:
            image = None
            try:
                width = frame['w']
                height = frame['h']
            except:
                width = contents['w']
                height = contents['h']
        
        tbar.update(1)
        
        if dataset_type == 'neu3d':
            focal_length_x = contents['fl_x']
            focal_length_y = contents['fl_y']
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        
            masks_path = os.path.join(path, 'masks', frame["file_path"].split('/')[-1] + '.pt')
            
            if load_mask_on_the_fly:
                masks = None
            else:
                masks = torch.load(masks_path) if os.path.exists(masks_path) else None
                if torch.is_tensor(masks):
                    masks = masks.to('cpu')
                
            return CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=width, height=height, fid=frame_time, masks=masks, mask_path=masks_path)
        elif dataset_type == 'blender':
            ## Blender
            fovy = focal2fov(fov2focal(fovx, width), height)
            FovY = fovy
            FovX = fovx
            
            masks_path = os.path.join(path, frame["file_path"].split('/')[-2], 'masks', frame["file_path"].split('/')[-1] + '.pt')
            if load_mask_on_the_fly:
                masks = None
            else:
                masks = torch.load(masks_path) if os.path.exists(masks_path) else None
                if torch.is_tensor(masks):
                    masks = masks.to('cpu')
                
            return CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=width, height=height, fid=frame_time, masks=masks, mask_path=masks_path)
            
        elif dataset_type == 'immersive' or dataset_type == 'technicolor':
            focal_length_x = frame['fl_x']
            focal_length_y = frame['fl_y']
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
            
            masks_path = os.path.join(path, 'masks', frame["file_path"].split('/')[-1] + '.pt')
            
            if load_mask_on_the_fly:
                masks = None
            else:
                masks = torch.load(masks_path) if os.path.exists(masks_path) else None
                if torch.is_tensor(masks):
                    masks = masks.to('cpu')

            return CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=width, height=height, fid=frame_time, masks=masks, mask_path=masks_path)
        else:
            raise NotImplementedError()
             
    with ThreadPool() as pool:
        cam_infos = pool.map(frame_read_fn, zip(list(range(len(frames))), frames))
        pool.close()
        pool.join()
        
    cam_infos = [cam_info for cam_info in cam_infos if cam_info is not None]
    
    print(f"[INFO] {len(cam_infos)} images loaded.")

    return cam_infos

def readMultiViewInfo(path, white_background, eval, extension=".png", load_image_on_the_fly=False, load_mask_on_the_fly=False, end_frame=None):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension, load_image_on_the_fly, load_mask_on_the_fly, end_frame)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension, load_image_on_the_fly, load_mask_on_the_fly, end_frame)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readNerfiesCameras(path, load_image_on_the_fly=False, load_mask_on_the_fly=False):
    with open(f'{path}/scene.json', 'r') as f:
        scene_json = json.load(f)
    with open(f'{path}/metadata.json', 'r') as f:
        meta_json = json.load(f)
    with open(f'{path}/dataset.json', 'r') as f:
        dataset_json = json.load(f)

    coord_scale = scene_json['scale']
    scene_center = scene_json['center']

    name = path.split('/')[-2]
    if name.startswith('vrig'):
        print("vrig dataset")
        train_img = dataset_json['train_ids']
        val_img = dataset_json['val_ids']
        all_img = train_img + val_img
        ratio = 0.25
    elif name.startswith('NeRF'):
        print("It's NeRF-DS dataset")
        train_img = dataset_json['train_ids']
        val_img = dataset_json['val_ids']
        all_img = train_img + val_img
        ratio = 0.5
    elif name.startswith('interp'):
        all_id = dataset_json['ids']
        train_img = all_id[::4]
        val_img = all_id[2::4]
        all_img = train_img + val_img
        ratio = 0.5
    else:  # for hypernerf
        print("It's HyperNeRF misc dataset")
        all_id = dataset_json['ids']
        train_img = all_id[::4]
        val_img = all_id[2::4]
        all_img = train_img + val_img
        ratio = 0.5

    train_num = len(train_img)

    all_cam = [meta_json[i]['camera_id'] for i in all_img]
    all_time = [meta_json[i]['time_id'] for i in all_img]
    max_time = max(all_time)
    all_time = [meta_json[i]['time_id'] / max_time for i in all_img]
    selected_time = set(all_time)

    # all poses
    all_cam_params = []
    for im in all_img:
        camera = camera_nerfies_from_JSON(f'{path}/camera/{im}.json', ratio)
        camera['position'] = camera['position'] - scene_center
        camera['position'] = camera['position'] * coord_scale
        all_cam_params.append(camera)

    all_img = [f'{path}/rgb/{int(1 / ratio)}x/{i}.png' for i in all_img]

    cam_infos = []
        
    for idx in range(len(all_img)):
        image_path = all_img[idx]
        image_name = Path(image_path).stem
        
        image = np.array(Image.open(image_path))
        image = Image.fromarray((image).astype(np.uint8))
        
        width = image.size[0]
        height = image.size[1]
        
        masks_path = os.path.join(path, 'masks', image_name + '.pt')
        
        if not load_mask_on_the_fly:
            masks = torch.load(masks_path) if os.path.exists(masks_path) else None
            if torch.is_tensor(masks):
                masks = masks.to('cpu')
        else:
            masks = None 
            
        if load_image_on_the_fly:
            image = None
            
        orientation = all_cam_params[idx]['orientation'].T
        position = -all_cam_params[idx]['position'] @ orientation
        focal = all_cam_params[idx]['focal_length']
        fid = all_time[idx]
        T = position
        R = orientation

        FovY = focal2fov(focal, height)
        FovX = focal2fov(focal, width)
        
        cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height,
                              fid=fid, masks=masks, mask_path=masks_path)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos, train_num, scene_center, coord_scale

def readNerfiesColmapCameras(path):
    with open(f'{path}/metadata.json', 'r') as f:
        meta_json = json.load(f)
    with open(f'{path}/dataset.json', 'r') as f:
        dataset_json = json.load(f)

    name = path.split('/')[-2]
    if name.startswith('vrig'):
        train_img = dataset_json['train_ids']
        val_img = dataset_json['val_ids']
        all_img = train_img + val_img
        ratio = 0.25
    elif name.startswith('NeRF'):
        train_img = dataset_json['train_ids']
        val_img = dataset_json['val_ids']
        all_img = train_img + val_img
        ratio = 1.0
    elif name.startswith('interp'):
        all_id = dataset_json['ids']
        train_img = all_id[::4]
        val_img = all_id[2::4]
        all_img = train_img + val_img
        ratio = 0.5
    else:  # for hypernerf
        train_img = dataset_json['ids'][::4]
        all_img = train_img
        ratio = 0.5
        # ratio = 1

    train_num = len(train_img)

    all_cam = [meta_json[i]['camera_id'] for i in all_img]
    all_time = [meta_json[i]['time_id'] for i in all_img]
    max_time = max(all_time)
    all_time = [meta_json[i]['time_id'] / max_time for i in all_img]
    selected_time = set(all_time)
    
    sparse_name = "sparse" if os.path.exists(os.path.join(path, 'colmap', "sparse")) else "colmap_sparse"
    cameras_extrinsic_file = os.path.join(path, 'colmap', f"{sparse_name}/0", "images.bin")
    cameras_intrinsic_file = os.path.join(path, 'colmap', f"{sparse_name}/0", "cameras.bin")
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    
    img_path = f'{path}/rgb/{int(1 / ratio)}x/'
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=img_path, source_path=path)
    name2idx = {cam.image_name: idx for idx, cam in enumerate(cam_infos_unsorted)}

    cam_infos = []
    for idx in range(len(all_img)):
        cam_infos.append(cam_infos_unsorted[name2idx[all_img[idx]]])
        cam_infos[-1]._replace(fid=all_time[idx])
    return cam_infos, train_num, 0, 1

def readNerfiesInfo(path, eval, load_image_on_the_fly=False, load_mask_on_the_fly=False):
    print("Reading Nerfies Info")
    # cam_infos, train_num, scene_center, scene_scale = readNerfiesCameras(path, objects_folder=None)
    if os.path.exists(os.path.join(path, 'colmap')):
        cam_infos, train_num, scene_center, scene_center = readNerfiesColmapCameras(path)
        recenter_by_pcl = apply_cam_norm = True
    else:
        cam_infos, train_num, scene_center, scene_scale = readNerfiesCameras(path, load_image_on_the_fly=load_image_on_the_fly, load_mask_on_the_fly=load_mask_on_the_fly)
        recenter_by_pcl = apply_cam_norm = False
        
    if eval:
        train_cam_infos = cam_infos[:train_num]
        test_cam_infos = cam_infos[train_num:]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    # nerf_normalization = getNerfppNorm(train_cam_infos)
    nerf_normalization = getNerfppNorm(train_cam_infos, apply=apply_cam_norm)
    
    if os.path.exists(os.path.join(path, 'colmap')):
        print('Using COLMAP for Nerfies!')
        sparse_name = "sparse" if os.path.exists(os.path.join(path, 'colmap', "sparse")) else "colmap_sparse"
        if recenter_by_pcl:
            ply_path = os.path.join(path, f"colmap/{sparse_name}/0/points3d_recentered.ply")
        elif apply_cam_norm:
            ply_path = os.path.join(path, f"colmap/{sparse_name}/0/points3d_normalized.ply")
        else:
            ply_path = os.path.join(path, f"colmap/{sparse_name}/0/points3d.ply")
        bin_path = os.path.join(path, f"colmap/{sparse_name}/0/points3D.bin")
        txt_path = os.path.join(path, f"colmap/{sparse_name}/0/points3D.txt")
        adj_path = os.path.join(path, f"colmap/{sparse_name}/0/camera_adjustment")
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            if apply_cam_norm:
                xyz += nerf_normalization["apply_translate"]
                xyz /= nerf_normalization["apply_radius"]
            if recenter_by_pcl:
                pcl_center = xyz.mean(axis=0)
                translate_cam_info(train_cam_infos, - pcl_center)
                translate_cam_info(test_cam_infos, - pcl_center)
                xyz -= pcl_center
                np.savez(adj_path, translate=-pcl_center)
            storePly(ply_path, xyz, rgb)
        else:
            translate = np.load(adj_path + '.npz')['translate']
            translate_cam_info(train_cam_infos, translate=translate)
            translate_cam_info(test_cam_infos, translate=translate)
    else:
        ply_path = os.path.join(path, "points3d.ply")
        if not os.path.exists(ply_path):
            print(f"Generating point cloud from nerfies...")

            xyz = np.load(os.path.join(path, "points.npy"))
            xyz = (xyz - scene_center) * scene_scale
            num_pts = xyz.shape[0]
            shs = np.random.random((num_pts, 3)) / 255.0
            pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
                shs), normals=np.zeros((num_pts, 3)))

            storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapStaticSceneInfo, # For Mip-NeRF 360 dataset
    "Blender" : readMultiViewInfo, # For multi-view dataset (Neu3D, technicolor, immersive)
    "nerfies": readNerfiesInfo,  # NeRF-DS & HyperNeRF dataset proposed by [https://github.com/google/hypernerf/releases/tag/v0.1]
}