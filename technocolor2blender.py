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
import csv

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
         
def extractframes(input_path, startframe=0, endframe=300, downscale=1):
    if downscale != 1:
        output_dir = os.path.join(input_path, f'images_{downscale}x')
    else:
        output_dir = os.path.join(input_path, 'images')
        
    os.makedirs(output_dir, exist_ok=True)
    # name = input_path.split("/")[-2]
    for i in range(startframe, endframe):
        img_list = glob.glob(input_path + "*_undist_" + str(i).zfill(5)+"_*.png")
        # print(img_list)
        for img_path in img_list:
            image_output_path = os.path.join(output_dir, f"camera_00{img_path.split('.')[-2][-2:]}_{str(i).zfill(4)}.png")
            # if not os.path.exists(image_output_path):
            frame = cv2.imread(img_path)
            if downscale > 1:
                new_width, new_height = int(frame.shape[1] / downscale), int(frame.shape[0] / downscale)
                frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            cv2.imwrite(image_output_path, frame)
            # else:
            #     print("already exists")
    
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
    
def fixbroken(imagepath, refimagepath):
    try:
        img = Image.open(imagepath) # open the image file
        print("start verifying", imagepath)
        img.verify() # if we already fixed it. 
        print("already fixed", imagepath)
    except :
        print('Bad file:', imagepath)
        import cv2
        from PIL import Image, ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img = Image.open(imagepath)
        
        img.load()
        img.save("tmp.png")

        savedimage = cv2.imread("tmp.png")
        mask = savedimage == 0
        refimage = cv2.imread(refimagepath)
        composed = savedimage * (1-mask) + refimage * (mask)
        cv2.imwrite(imagepath, composed)
        print("fixing done", imagepath)
        os.remove("tmp.png")

SCENE_FRAMES = {
    'Birthday': [151, 201],
    'Fabien': [51, 101],
    'Painter': [100, 150],
    'Theater': [51, 101],
    'Train': [151, 201]
}
if __name__ == '__main__':
    parser = argparse.ArgumentParser() # TODO: refine it.
    parser.add_argument("--path", default="", help="input path to the video")
    parser.add_argument("--scale", type=int, default=1, help="scale of the image")
    parser.add_argument("--extract_image_only", action='store_true', default=False)

    args = parser.parse_args()

    # path must end with / to make sure image path is relative
    if args.path[-1] != '/':
        args.path += '/'
    scene_name = args.path.split("/")[-2]
    
    print(f"Processing {scene_name}")
    try:
        scene_frames = SCENE_FRAMES[scene_name]
    except:
        raise NotImplementedError(f"Scene {scene_name} is not used in our experiment")
    
    if scene_name == "Birthday":
        print("check broken")
        fixbroken(args.path + "Birthday_undist_00173_09.png", args.path + "Birthday_undist_00172_09.png")

    extractframes(args.path, startframe=scene_frames[0], endframe=scene_frames[1], downscale=args.scale)
    
    if args.scale != 1:
        images = [f[len(args.path):] for f in sorted(glob.glob(os.path.join(args.path, f"images_{args.scale}x/", "*"))) if f.lower().endswith('png') or f.lower().endswith('jpg') or f.lower().endswith('jpeg')]
        cams = sorted(set([im[10:21] for im in images]))
    else:
        images = [f[len(args.path):] for f in sorted(glob.glob(os.path.join(args.path, "images/", "*"))) if f.lower().endswith('png') or f.lower().endswith('jpg') or f.lower().endswith('jpeg')]
        cams = sorted(set([im[7:18] for im in images]))

    print(f'[INFO] loaded {len(images)} images from {len(cams)} videos')
    N = len(cams)
    
    H = 1088
    W = 2048
    
    print(f'[INFO] Original: H = {H}, W = {W}')
    
    print(f'[INFO] H = {H // args.scale}, W = {W // args.scale}')
    
    train_frames = []
    test_frames = []
    poses = []
    cam_infos = {}
    
    with open(os.path.join(args.path, "cameras_parameters.txt"), "r") as f:
        meta = csv.reader(f, delimiter=" ")
        for idx, row in enumerate(meta):
            if idx == 0:
                continue
            idx = idx - 1
            cameraname = f'camera_{str(idx).zfill(4)}'
            
            row = [float(c) for c in row if c.strip() != '']
            
            width = W // args.scale
            height = H // args.scale
            
            fx = row[0] / args.scale
            fy = row[0] / args.scale

            cx = row[1] / args.scale
            cy = row[2] / args.scale
            
            
            colmapQ = [row[5], row[6], row[7], row[8]] 
            colmapT = [row[9], row[10], row[11]]
            
            w2c = np.eye(4)
            w2c[:3, :3] = qvec2rotmat(np.array(colmapQ))
            w2c[:3, 3] = np.array(colmapT)

            poses.append(w2c)
            
            cam_infos[cameraname] = {
                "w": width,
                "h": height,
                "fl_x": fx,
                "fl_y": fy,
                "cx": cx,
                "cy": cy,
            }
    # print(cam_infos)
    for i in range(N):
        cam_frames = [{'file_path': im.lstrip("/").split('.')[0], 
                    'transform_matrix': poses[i].tolist(),
                    'w': cam_infos[cams[i]]['w'],
                    'h': cam_infos[cams[i]]['h'],
                    'fl_x': cam_infos[cams[i]]['fl_x'],
                    'fl_y': cam_infos[cams[i]]['fl_y'],
                    'cx': cam_infos[cams[i]]['cx'],
                    'cy': cam_infos[cams[i]]['cy'],
                    'time': (int(im.lstrip("/").split('.')[0][-4:]) - scene_frames[0]) / 30.} for im in images if cams[i] in im]
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
    for frame in train_frames:
        original_image = cv2.imread(os.path.join(args.path, frame['file_path'] + '.png'))
        w = frame['w'] 
        h = frame['h'] 
        c_x = frame['cx']
        c_y = frame['cy']
        new_c_x, new_c_y = w / 2, h / 2 

        # Compute the translation matrix
        translation_matrix = np.array([
            [1, 0, new_c_x - c_x],
            [0, 1, new_c_y - c_y]
        ], dtype=np.float32)
        
        # Apply the translation to the image
        translated_image = cv2.warpAffine(original_image, translation_matrix, (w, h))
        cv2.imwrite(os.path.join(args.path, frame['file_path'] + '.png'), translated_image)
        frame['cx'] = new_c_x
        frame['cy'] = new_c_y
    for frame in test_frames:
        original_image = cv2.imread(os.path.join(args.path, frame['file_path'] + '.png'))
        w = frame['w'] 
        h = frame['h'] 
        c_x = frame['cx']
        c_y = frame['cy']
        new_c_x, new_c_y = w / 2, h / 2 

        # Compute the translation matrix
        translation_matrix = np.array([
            [1, 0, new_c_x - c_x],
            [0, 1, new_c_y - c_y]
        ], dtype=np.float32)

        # Apply the translation to the image
        translated_image = cv2.warpAffine(original_image, translation_matrix, (w, h))
        cv2.imwrite(os.path.join(args.path, frame['file_path'] + '.png'), translated_image)
        frame['cx'] = new_c_x
        frame['cy'] = new_c_y
        
    if not args.extract_image_only:            
        train_output_path = os.path.join(args.path, 'transforms_train.json')
        test_output_path = os.path.join(args.path, 'transforms_test.json')
        print(f'[INFO] write to {train_output_path} and {test_output_path}')
        
        with open(train_output_path, 'w') as f:
            json.dump(train_transforms, f, indent=2)
        with open(test_output_path, 'w') as f:
            json.dump(test_transforms, f, indent=2)
        if os.path.exists(os.path.join(args.path, 'tmp')):
            shutil.rmtree(os.path.join(args.path, 'tmp'))
        colmap_workspace = os.path.join(args.path, 'tmp')

        os.makedirs(os.path.join(colmap_workspace, 'created', 'sparse'), exist_ok=True)
        
        fname2pose = {}
        fname2camid= {}
        
        counter = 0
        with open(os.path.join(colmap_workspace, 'created/sparse/cameras.txt'), 'w') as f:
            for frame in train_frames:
                if frame['time'] == 0:
                    f.write(f'{counter + 1} PINHOLE {W // args.scale} {H // args.scale} {frame["fl_x"]} {frame["fl_y"]} {frame["cx"]} {frame["cy"]}\n')
                    fname = frame['file_path'].split('/')[-1] + '.png'
                    pose = np.array(frame['transform_matrix'])
                    fname2pose.update({fname: pose})
                    fname2camid.update({fname: counter + 1})
                    counter += 1
        assert counter == N - 1
        
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
            
        os.makedirs(os.path.join(colmap_workspace, 'triangulated', 'sparse'), exist_ok=True)
        
        do_system(f"colmap point_triangulator   \
                    --database_path {db_path} \
                    --image_path {os.path.join(colmap_workspace, 'images')} \
                    --input_path  {os.path.join(colmap_workspace, 'created/sparse')} \
                    --output_path  {os.path.join(colmap_workspace, 'triangulated/sparse')}")
        
        do_system(f"colmap model_converter \
                    --input_path  {os.path.join(colmap_workspace, 'triangulated/sparse')} \
                    --output_path  {os.path.join(colmap_workspace, 'created/sparse')} \
                    --output_type TXT")
        
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
        