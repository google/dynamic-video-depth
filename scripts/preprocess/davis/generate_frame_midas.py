# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from os.path import join
from os import makedirs
from glob import glob
from skimage.transform import resize as imresize
from tqdm import tqdm
from PIL import Image
from scipy.ndimage import map_coordinates
import trimesh
import sys
sys.path.insert(0, '')
from configs import midas_pretrain_path
from third_party.MiDaS import MidasNet

model = MidasNet(midas_pretrain_path, non_negative=True, resize=[256, 512], normalize_input=True)

model = model.eval().cuda()


data_list_root = "./datafiles/DAVIS/JPEGImages/1080p"
camera_path = "./datafiles/DAVIS/triangulation"
mask_path = './datafiles/DAVIS/Annotations/1080p'
outpath = './datafiles/davis_processed/frames_midas'

track_names = ['train', 'dog']
track_ids = [0, 1]


for track_id in track_ids:
    print(track_names[track_id])
    frames = sorted(glob(join(data_list_root, f'{track_names[track_id]}', '*.jpg')))
    mask_paths = sorted(glob(join(mask_path, f'{track_names[track_id]}', '*.png')))
    makedirs(join(outpath, f'{track_names[track_id]}'), exist_ok=True)
    intrinsics_path = join(camera_path, f'{track_names[track_id]}.intrinsics.txt')
    extrinsics_path = join(camera_path, f'{track_names[track_id]}.matrices.txt')
    obj_path = join(camera_path, f'{track_names[track_id]}.obj')
    fx, fy, cx, cy = np.loadtxt(intrinsics_path)[0][1:]
    extrinsics = np.loadtxt(extrinsics_path)
    extrinsics = extrinsics[:, 1:]
    extrinsics = np.asarray([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])[None, ...]@np.linalg.inv(np.reshape(extrinsics, [-1, 4, 4]))
    mesh = trimesh.load(obj_path)
    points_3d = mesh.vertices
    h_pt = np.ones([points_3d.shape[0], 4])
    h_pt[:, :3] = points_3d
    h_pt = h_pt.T
    intrinsics = np.zeros([3, 3])
    intrinsics[[0, 0, 1, 1, 2], [0, 2, 1, 2, 2]] = [fx, cx, fy, cy, 1]

    print('calculating NN_depth')
    full_pred_depths = []
    pts_list = []

    mvs_depths = []
    pred_depths = []
    masks = []
    for x in tqdm(range(len(frames))):
        img = np.asarray(Image.open(frames[x])).astype(np.float32) / 255

        img_batch = torch.from_numpy(img).permute(2, 0, 1)[None, ...].float().cuda()

        with torch.no_grad():
            pred_d = model(img_batch)
            pred_d = pred_d.squeeze().cpu().numpy()
            full_pred_depths.append(pred_d)

        out = extrinsics[x, :]@h_pt
        im_pt = intrinsics @ out[:3, :]
        depth = im_pt[2, :].copy()
        im_pt = im_pt / im_pt[2:, :]

        mask = np.asarray(Image.open(mask_paths[x]).convert('RGB')).astype(np.float32)[:, :, 0] / 255
        masks.append(mask)
        H, W, _ = img.shape
        select_idx = np.where((im_pt[0, :] >= 0) * (im_pt[0, :] < W) * (im_pt[1, :] >= 0) * (im_pt[1, :] < H))[0]
        pts = im_pt[:, select_idx]
        depth = depth[select_idx]
        out = map_coordinates(mask, [pts[1, :], pts[0, :]])
        select_idx = np.where(out < 0.1)[0]
        pts = pts[:, select_idx]
        depth = depth[select_idx]
        select_idx = np.where(depth > 1e-3)[0]
        pts = pts[:, select_idx]
        depth = depth[select_idx]

        pred_depth = map_coordinates(pred_d, [pts[1, :], pts[0, :]])
        mvs_depths.append(depth)
        pred_depths.append(pred_depth)
        pts_list.append(pts)
    print(img.shape)

    print('calculating scale')
    scales = []
    for x in tqdm(range(len(frames))):
        nn_depth = pred_depths[x]
        mvs_depth = mvs_depths[x]
        scales.append(np.median(nn_depth / mvs_depth))
    s = np.mean(scales)

    print('saving per frame output')

    for idf, frame_path in tqdm(enumerate(frames)):
        img_orig = np.asarray(Image.open(frames[idf])).astype(np.float32) / 255
        max_W = 384
        multiple = 64
        H, W, _ = img_orig.shape
        if W > max_W:
            sc = max_W / W
            target_W = max_W
        else:
            target_W = W
        target_H = int(np.round((H * sc) / multiple) * multiple)

        img = imresize(img_orig, ([target_H, target_W]), preserve_range=True).astype(np.float32)

        T_G_1 = extrinsics[idf, ...]  # world2cam
        T_G_1[:3, 3] *= s
        T_G_1 = np.linalg.inv(T_G_1)  # cam2world
        T_G_1 = T_G_1.astype(np.float32)
        depth_mvs = imresize(full_pred_depths[idf].astype(np.float32), ([target_H, target_W]), preserve_range=True).astype(np.float32)
        in_1 = intrinsics.copy()
        in_1[0, 0] /= W / target_W
        in_1[1, 1] /= H / target_H
        in_1[0, 2] = (target_W - 1) / 2
        in_1[1, 2] = (target_H - 1) / 2
        in_1 = in_1.astype(np.float32)
        depth = full_pred_depths[idf].astype(np.float32)
        depth = imresize(depth, ([target_H, target_W]), preserve_range=True).astype(np.float32)
        resized_mask = imresize(masks[idf], [target_H, target_W], preserve_range=True)
        resized_mask = np.where(resized_mask > 1e-3, 1, 0)

        np.savez(join(outpath, track_names[track_id], 'frame_%05d.npz' % idf), img=img, pose_c2w=T_G_1,
                 depth_mvs=depth_mvs, intrinsics=in_1, depth_pred=depth, img_orig=img_orig, motion_seg=resized_mask)
