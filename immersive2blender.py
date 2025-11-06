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
# Inspired by the preprocessing script from:
# SpacetimeGaussians: https://github.com/oppo-us-research/SpacetimeGaussians
# 4d-gaussian-splatting: https://github.com/fudan-zvg/4d-gaussian-splatting
#

import os
import argparse
import glob

import numpy as np
import json
import sys
import math
import shutil
import sqlite3
from scipy.spatial.transform import Rotation
from copy import deepcopy
import cv2
from tqdm import tqdm
from pathlib import Path
    
IS_PYTHON3 = sys.version_info[0] >= 3
MAX_IMAGE_ID = 2**31 - 1

CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL)"""

CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    prior_qw REAL,
    prior_qx REAL,
    prior_qy REAL,
    prior_qz REAL,
    prior_tx REAL,
    prior_ty REAL,
    prior_tz REAL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))
""".format(MAX_IMAGE_ID)

CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB,
    qvec BLOB,
    tvec BLOB)
"""

CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
"""

CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB)"""

CREATE_NAME_INDEX = \
    "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"

CREATE_ALL = "; ".join([
    CREATE_CAMERAS_TABLE,
    CREATE_IMAGES_TABLE,
    CREATE_KEYPOINTS_TABLE,
    CREATE_DESCRIPTORS_TABLE,
    CREATE_MATCHES_TABLE,
    CREATE_TWO_VIEW_GEOMETRIES_TABLE,
    CREATE_NAME_INDEX
])


def array_to_blob(array):
    if IS_PYTHON3:
        return array.tostring()
    else:
        return np.getbuffer(array)

def blob_to_array(blob, dtype, shape=(-1,)):
    if IS_PYTHON3:
        return np.fromstring(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)

class COLMAPDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)

        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table = \
            lambda: self.executescript(CREATE_CAMERAS_TABLE)
        self.create_descriptors_table = \
            lambda: self.executescript(CREATE_DESCRIPTORS_TABLE)
        self.create_images_table = \
            lambda: self.executescript(CREATE_IMAGES_TABLE)
        self.create_two_view_geometries_table = \
            lambda: self.executescript(CREATE_TWO_VIEW_GEOMETRIES_TABLE)
        self.create_keypoints_table = \
            lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
        self.create_matches_table = \
            lambda: self.executescript(CREATE_MATCHES_TABLE)
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

    def update_camera(self, model, width, height, params, camera_id):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "UPDATE cameras SET model=?, width=?, height=?, params=?, prior_focal_length=1 WHERE camera_id=?",
            (model, width, height, array_to_blob(params),camera_id))
        return cursor.lastrowid

def camTodatabase(txtfile, database_path):
    import os
    import argparse

    camModelDict = {'SIMPLE_PINHOLE': 0,
                    'PINHOLE': 1,
                    'SIMPLE_RADIAL': 2,
                    'RADIAL': 3,
                    'OPENCV': 4,
                    'FULL_OPENCV': 5,
                    'SIMPLE_RADIAL_FISHEYE': 6,
                    'RADIAL_FISHEYE': 7,
                    'OPENCV_FISHEYE': 8,
                    'FOV': 9,
                    'THIN_PRISM_FISHEYE': 10}

    if os.path.exists(database_path)==False:
        print("ERROR: database path dosen't exist -- please check database.db.")
        return
    # Open the database.
    db = COLMAPDatabase.connect(database_path)

    idList=list()
    modelList=list()
    widthList=list()
    heightList=list()
    paramsList=list()
    # Update real cameras from .txt
    with open(txtfile, "r") as cam:
        lines = cam.readlines()
        for i in range(0,len(lines),1):
            if lines[i][0]!='#':
                strLists = lines[i].split()
                print(strLists)
                cameraId=int(strLists[0])
                cameraModel=camModelDict[strLists[1]] #SelectCameraModel
                width=int(strLists[2])
                height=int(strLists[3])
                paramstr=np.array(strLists[4:12])
                params = paramstr.astype(np.float64)
                idList.append(cameraId)
                modelList.append(cameraModel)
                widthList.append(width)
                heightList.append(height)
                paramsList.append(params)
                camera_id = db.update_camera(cameraModel, width, height, params, cameraId)

    # Commit the data to the file.
    db.commit()
    # Read and check cameras.
    rows = db.execute("SELECT * FROM cameras")
    for i in range(0,len(idList),1):
        camera_id, model, width, height, params, prior = next(rows)
        params = blob_to_array(params, np.float64)
        assert camera_id == idList[i]
        assert model == modelList[i] and width == widthList[i] and height == heightList[i]
        assert np.allclose(params, paramsList[i])

    # Close database.db.
    db.close()

def do_system(arg):
    print(f"==== running: {arg}")
    err = os.system(arg)
    if err:
        print("FATAL: command failed")
        sys.exit(err)

def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

def compute_undistort_intrinsic(K, height, width, distortion_params):
    ## Refer to https://github.com/scannetpp/scannetpp/blob/136c416baa915738c27db1aad198429be8fba68d/dslr/undistort.py
    assert len(distortion_params.shape) == 1
    assert distortion_params.shape[0] == 4  # OPENCV_FISHEYE has k1, k2, k3, k4

    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K,
        distortion_params,
        (width, height),
        np.eye(3),
        balance=0.0,
    )
    # Make the cx and cy to be the center of the image
    new_K[0, 2] = width / 2.0
    new_K[1, 2] = height / 2.0
    return new_K

def undistort_image(video, frames, down_scale=1.0):
    with open(os.path.join(video, "models.json"), "r") as f:
        meta = json.load(f)

    for idx , camera in enumerate(meta):
        folder = camera['name']
        output_img_folder_name = 'images'
        if down_scale != 1:
            folder += f'_{down_scale}x'
            output_img_folder_name += f'_{down_scale}x'
        view = camera
        intrinsics = np.array([[view['focal_length'] / down_scale, 0.0, view['principal_point'][0] / down_scale],
                            [0.0, view['focal_length'] / down_scale, view['principal_point'][1] / down_scale],
                            [0.0, 0.0, 1.0]])
        dis_cef = np.zeros((4))
        w, h = int(camera['width'] // down_scale), int(camera['height'] // down_scale)
        dis_cef[:2] = np.array(view['radial_distortion'])[:2]
        new_K = compute_undistort_intrinsic(intrinsics, h, w, dis_cef)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            intrinsics, dis_cef, np.eye(3), new_K, (w, h), cv2.CV_32FC1
        )
        assert new_K[0, 0] == new_K[1, 1]
        assert meta[idx]['name'] == camera['name'], f"{meta[idx]['name']} is not {camera['name']}"
        meta[idx]['focal_length'] = new_K[0, 0]
        meta[idx]['principal_point'][0] = new_K[0, 2]
        meta[idx]['principal_point'][1] = new_K[1, 2]
        meta[idx]['height'] = h
        meta[idx]['width'] = w
        print(f"{camera['name']} new K: {new_K}")
        for frame in frames:
            videofolder = os.path.join(video, folder)
            imagepath = os.path.join(videofolder, str(frame) + ".png")
            
            imagesavepath = os.path.join(video, output_img_folder_name, f"{camera['name']}_{frame:04d}.png")
            output_img_folder = os.path.join(video, output_img_folder_name)
            
            if not os.path.exists(output_img_folder):
                os.makedirs(output_img_folder)   
            if not os.path.exists(imagesavepath):
                
                assert os.path.exists(imagepath), f"{imagepath} not exist"
                
                try:
                    image = cv2.imread(imagepath).astype(np.float32) #/ 255.0
                except:
                    print("failed to read image", imagepath)
                    quit()
                    
                undistorted_image = cv2.remap(
                    image,
                    map1,
                    map2,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101,
                )
                undistorted_image = undistorted_image.clip(0,255.0).astype(np.uint8)

                cv2.imwrite(imagesavepath, undistorted_image)
            else:
                print("already exists")

    with open(os.path.join(video, "models_new.json"), "w") as f:
        json.dump(meta, f, indent=4)
            
def extractframes(videopath_str, startframe=0, endframe=300, downscale=1):
    videopath = Path(videopath_str)
    output_dir = videopath.with_suffix('')
    # print(output_dir)
    if downscale != 1:
        output_dir = str(output_dir) + f'_{downscale}x'
    output_dir = Path(output_dir)
    if all((output_dir / f"{i}.png").exists() for i in range(startframe, endframe)):
        print(f"Already extracted all the frames in {output_dir}")
        return

    cam = cv2.VideoCapture(str(videopath))
    cam.set(cv2.CAP_PROP_POS_FRAMES, startframe)

    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(startframe, endframe):
        success, frame = cam.read()
        if not success:
            print(f"Error reading frame {i}")
            break

        if downscale > 1:
            new_width, new_height = int(frame.shape[1] / downscale), int(frame.shape[0] / downscale)
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

        cv2.imwrite(str(output_dir / f"{i}.png"), frame)

    cam.release()
    
def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])
    
def update_transform(path, train_frames, test_frames):
    new_cam_intrinsics = {}
    with open(os.path.join(path, 'tmp/created/sparse/cameras.txt'), "r") as cam:
        lines = cam.readlines()
        for i in range(0,len(lines),1):
            if lines[i][0] != '#':
                strLists = lines[i].split()
                cameraId=int(strLists[0])
                width=int(strLists[2])
                height=int(strLists[3])
                paramstr=np.array(strLists[4:12])
                params = paramstr.astype(np.float64)
                new_cam_intrinsics[cameraId] = {'w': width,
                                                'h': height,
                                                'fl_x': params[0],
                                                'fl_y': params[1],
                                                'cx': params[2],
                                                'cy': params[3],
                                                }
    new_cam_poses_intrinsics = {}
    with open(os.path.join(path, 'tmp/created/sparse/images.txt'), "r") as cam:
        lines = cam.readlines()
        for i in range(0,len(lines),1):
            if lines[i][0] != '#':
                if '.png' in lines[i]:
                    strLists = lines[i].split()
                    cam_name = strLists[-1][:11]
                    cam_id = int(strLists[-2])
                    new_cam_poses_intrinsics[cam_name] = new_cam_intrinsics[cam_id]
                    poses = np.array(strLists[1:8]).astype(np.float64)
                    w2c = np.eye(4)
                    w2c[:3, :3] = qvec2rotmat(poses[:4])
                    w2c[:3, 3] = poses[4:]
                    new_cam_poses_intrinsics[cam_name]['transform_matrix'] = w2c.tolist()
    
    new_train_frames = []
    new_test_frames = []
    for frame in train_frames:
        cam_name = frame['file_path'].split('/')[-1][:11]
        if cam_name in new_cam_poses_intrinsics.keys():
            # print(new_cam_poses_intrinsics[cam_name])
            frame['transform_matrix'] = new_cam_poses_intrinsics[cam_name]['transform_matrix']
            frame['w'] = new_cam_poses_intrinsics[cam_name]['w']
            frame['h'] = new_cam_poses_intrinsics[cam_name]['h']
            frame['fl_x'] = new_cam_poses_intrinsics[cam_name]['fl_x']
            frame['fl_y'] = new_cam_poses_intrinsics[cam_name]['fl_y']
            frame['cx'] = new_cam_poses_intrinsics[cam_name]['cx']
            frame['cy'] = new_cam_poses_intrinsics[cam_name]['cy']
            # print(frame)
            new_train_frames.append(frame)
        
    for frame in test_frames:
        cam_name = frame['file_path'].split('/')[-1][:11]
        if cam_name in new_cam_poses_intrinsics.keys():
            frame['transform_matrix'] = new_cam_poses_intrinsics[cam_name]['transform_matrix']
            frame['w'] = new_cam_poses_intrinsics[cam_name]['w']
            frame['h'] = new_cam_poses_intrinsics[cam_name]['h']
            frame['fl_x'] = new_cam_poses_intrinsics[cam_name]['fl_x']
            frame['fl_y'] = new_cam_poses_intrinsics[cam_name]['fl_y']
            frame['cx'] = new_cam_poses_intrinsics[cam_name]['cx']
            frame['cy'] = new_cam_poses_intrinsics[cam_name]['cy']
            new_test_frames.append(frame)
            
    train_transforms = {
        'frames': new_train_frames,
    }
    test_transforms = {
        'frames': new_test_frames,
    }
    train_output_path = os.path.join(path, 'transforms_train.json')
    test_output_path = os.path.join(path, 'transforms_test.json')
    print(f'[INFO] write to {train_output_path} and {test_output_path}')
    
    with open(train_output_path, 'w') as f:
        json.dump(train_transforms, f, indent=2)
    with open(test_output_path, 'w') as f:
        json.dump(test_transforms, f, indent=2)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser() # TODO: refine it.
    parser.add_argument("--path", default="", help="input path to the video")
    parser.add_argument("--scale", type=int, default=1, help="scale of the image")
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=300, type=int)
    parser.add_argument("--extract_image_only", action='store_true', default=False)
    args = parser.parse_args()

    # path must end with / to make sure image path is relative
    if args.path[-1] != '/':
        args.path += '/'
    scene_name = args.path.split("/")[-2]
    print(f"Processing {scene_name}")
    if scene_name in ["04_Truck", "03_Dog", "06_Goats"]:
        if args.end > 150:
            args.end = 150 
    videos_list = glob.glob(args.path + "*.mp4")
    
    # extract images
    for v in tqdm(videos_list):
        extractframes(videopath_str=v, startframe=args.start, endframe=args.end, downscale=args.scale)
    
    undistort_image(args.path, frames=[i for i in range(args.start, args.end)], down_scale=args.scale)
    
    if not args.extract_image_only:
        with open(os.path.join(args.path, "models.json"), "r") as f:
            meta = json.load(f)
        
        if args.scale != 1:
            images = [f[len(args.path):] for f in sorted(glob.glob(os.path.join(args.path, f"images_{args.scale}x/", "*"))) if f.lower().endswith('png') or f.lower().endswith('jpg') or f.lower().endswith('jpeg')]
            cams = sorted(set([im[10:21] for im in images]))
        else:
            images = [f[len(args.path):] for f in sorted(glob.glob(os.path.join(args.path, "images/", "*"))) if f.lower().endswith('png') or f.lower().endswith('jpg') or f.lower().endswith('jpeg')]
            cams = sorted(set([im[7:18] for im in images]))

        N = len(meta)
        print(f'[INFO] loaded {len(images)} images from {len(cams)} videos, {N} camera poses')

        assert N == len(cams)
        
        H = meta[0]['height']
        W = meta[0]['width']
        
        print(f'[INFO] Original: H = {H}, W = {W}')

        with open(os.path.join(args.path, "models_new.json"), "r") as f:
            meta = json.load(f)
            
        H = meta[0]['height']
        W = meta[0]['width']
        
        print(f'[INFO] H = {H}, W = {W}')
        
        train_frames = []
        test_frames = []
        poses = []
        cam_infos = {}
        for idx , camera in enumerate(meta):
            cameraname = camera['name']
            
            view = camera
            
            focal_length = camera['focal_length']
            width, height = W, H
            principal_point = [0, 0]
            
            principal_point[0] = view['principal_point'][0]
            principal_point[1] = view['principal_point'][1]
        
            
            R = Rotation.from_rotvec(view['orientation']).as_matrix()
            t = np.array(view['position'])[:, np.newaxis]
            w2c = np.concatenate((R, -np.dot(R, t)), axis=1)
            poses.append(np.concatenate((w2c, np.array([0, 0, 0, 1]).reshape(1, -1)), axis=0))

            K = np.array([[focal_length, 0, principal_point[0]], 
                        [0, focal_length, principal_point[1]], 
                        [0, 0, 1]])


            newfocalx = K[0, 0]
            newfocaly = K[1, 1]
            newcx = K[0, 2]
            newcy = K[1, 2]
            
            cam_infos[cameraname] = {
                "w": width,
                "h": height,
                "fl_x": newfocalx,
                "fl_y": newfocaly,
                "cx": newcx,
                "cy": newcy,
            }
        
        for i in range(N):
            cam_frames = [{'file_path': im.lstrip("/").split('.')[0], 
                        'transform_matrix': poses[i].tolist(),
                        'w': cam_infos[cams[i]]['w'],
                        'h': cam_infos[cams[i]]['h'],
                        'fl_x': cam_infos[cams[i]]['fl_x'],
                        'fl_y': cam_infos[cams[i]]['fl_y'],
                        'cx': cam_infos[cams[i]]['cx'],
                        'cy': cam_infos[cams[i]]['cy'],
                        'time': int(im.lstrip("/").split('.')[0][-4:]) / 30.} for im in images if cams[i] in im]
            if i == 0:
                test_frames += cam_frames
            else:
                train_frames += cam_frames

        train_transforms = {
            'frames': train_frames,
        }
        test_transforms = {
            'frames': test_frames,
        }

        train_output_path = os.path.join(args.path, 'transforms_train.json')
        test_output_path = os.path.join(args.path, 'transforms_test.json')
        print(f'[INFO] write to {train_output_path} and {test_output_path}')
        
        with open(train_output_path, 'w') as f:
            json.dump(train_transforms, f, indent=2)
        with open(test_output_path, 'w') as f:
            json.dump(test_transforms, f, indent=2)
        
        colmap_workspace = os.path.join(args.path, 'tmp')
        os.makedirs(os.path.join(colmap_workspace, 'created', 'sparse'), exist_ok=True)
        
        fname2pose = {}
        fname2camid= {}
        
        counter = 0
        with open(os.path.join(colmap_workspace, 'created/sparse/cameras.txt'), 'w') as f:
            for frame in train_frames:
                if frame['time'] == 0:
                    f.write(f'{counter + 1} PINHOLE {W} {H} {frame["fl_x"]} {frame["fl_y"]} {frame["cx"]} {frame["cy"]}\n')
                    fname = frame['file_path'].split('/')[-1] + '.png'
                    pose = np.array(frame['transform_matrix'])
                    fname2pose.update({fname: pose})
                    fname2camid.update({fname: counter + 1})
                    counter += 1
            for frame in test_frames:
                if frame['time'] == 0:
                    f.write(f'{counter + 1} PINHOLE {W} {H} {frame["fl_x"]} {frame["fl_y"]} {frame["cx"]} {frame["cy"]}\n')
                    fname = frame['file_path'].split('/')[-1] + '.png'
                    pose = np.array(frame['transform_matrix'])
                    fname2pose.update({fname: pose})
                    fname2camid.update({fname: counter + 1})
                    counter += 1
        assert counter == N
        
        if args.scale != 1:
            images_path = os.path.join(args.path, f"images_{args.scale}x/")
        else:
            images_path = os.path.join(args.path, "images/")
                
        os.makedirs(os.path.join(colmap_workspace, 'images'), exist_ok=True)
        for fname in fname2pose.keys():
            os.symlink(os.path.abspath(os.path.join(images_path, fname)), os.path.join(colmap_workspace, 'images', fname))
                    
        with open(os.path.join(colmap_workspace, 'created/sparse/images.txt'), 'w') as f:
            idx = 1
            for fname in fname2pose.keys():
                w2c = fname2pose[fname]
                colmapR = w2c[:3, :3]
                colmapQ = rotmat2qvec(colmapR)
                T = w2c[:3, 3]

                f.write(f'{idx} {colmapQ[0]} {colmapQ[1]} {colmapQ[2]} {colmapQ[3]} {T[0]} {T[1]} {T[2]} {fname2camid[fname]} {fname}\n\n')
                idx += 1
        
        with open(os.path.join(colmap_workspace, 'created/sparse/points3D.txt'), 'w') as f:
            f.write('')
        
        db_path = os.path.join(colmap_workspace, 'database.db')
        
        do_system(f"colmap feature_extractor \
                    --database_path {db_path} \
                    --image_path {os.path.join(colmap_workspace, 'images')}")
        
        camTodatabase(os.path.join(colmap_workspace, 'created/sparse/cameras.txt'), db_path)
        
        do_system(f"colmap exhaustive_matcher  \
                    --database_path {db_path}")
        
        os.makedirs(os.path.join(colmap_workspace, 'mapper', 'sparse'), exist_ok=True)
        
        do_system(f"colmap mapper   \
                    --database_path {db_path} \
                    --image_path {os.path.join(colmap_workspace, 'images')} \
                    --output_path  {os.path.join(colmap_workspace, 'mapper/sparse')}")
        do_system(f"colmap bundle_adjuster \
                    --input_path {os.path.join(colmap_workspace, 'mapper/sparse/0')} \
                    --output_path {os.path.join(colmap_workspace, 'mapper/sparse')}")
        
        do_system(f"colmap model_converter \
                    --input_path  {os.path.join(colmap_workspace, 'mapper/sparse')} \
                    --output_path  {os.path.join(colmap_workspace, 'created/sparse')} \
                    --output_type TXT")
        
        update_transform(path=args.path, train_frames=train_frames, test_frames=test_frames)
        
        os.makedirs(os.path.join(colmap_workspace, 'dense'), exist_ok=True)
        
        do_system(f"colmap image_undistorter  \
                    --image_path  {os.path.join(colmap_workspace, 'images')} \
                    --input_path  {os.path.join(colmap_workspace, 'created/sparse')} \
                    --output_path  {os.path.join(colmap_workspace, 'dense')}")
        
        do_system(f"colmap patch_match_stereo   \
                    --workspace_path   {os.path.join(colmap_workspace, 'dense')}")
        
        do_system(f"colmap stereo_fusion    \
                    --workspace_path {os.path.join(colmap_workspace, 'dense')} \
                    --output_path {os.path.join(args.path, 'points3d.ply')}")
        
        shutil.rmtree(colmap_workspace)
        os.remove(os.path.join(args.path, 'points3d.ply.vis'))
        
        print(f"[INFO] Initial point cloud is saved in {os.path.join(args.path, 'points3d.ply')}.")
    else:
        print(f"Extracted train / test images from video. Please make sure that you have points3d.ply, transforms_test.json, and transforms_train.json saved in the data folder for {scene_name}. Otherwise, you have to set --extract_image_only to False...")