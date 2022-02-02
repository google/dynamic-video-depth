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


import sys
import numpy as np
import h5py
import torch
from os.path import join, basename, dirname
from os import makedirs
from glob import glob
from skimage.transform import resize as imresize
from tqdm import tqdm
sys.path.insert(0, '')
from third_party.MiDaS import MidasNet
from configs import midas_pretrain_path
from PIL import Image

model = MidasNet(midas_pretrain_path, non_negative=True, resize=None, normalize_input=True)
model = model.eval().cuda()


data_list_root = "./datafiles/shutterstock/triangulation"
image_list_root = './datafiles/shutterstock/images'
outpath = './datafiles/shutterstock/frames_midas'
TRIM_BAD_FRAMES = True

track_paths = sorted(glob(join(data_list_root, '*')))

track_names = [basename(x) for x in track_paths]
print('Track names: ', track_names)
track_ids = np.arange(len(track_names))
# %% filter out valid sequences
track_lut = {}
track_ts = {}
tracks_grad = {}
for tr in track_ids:
    file_list = []
    all_files = sorted(glob(join(track_paths[tr], '*.h5')))
    ts = []
    for f in all_files:
        ts_str = f.split('/')[-1].split('_')[-1].split('.')[0]
        ts.append(int(ts_str))

    idx = np.argsort(ts)
    sorted_path = [all_files[x] for x in idx]
    track_lut[tr] = sorted_path
    track_ts[tr] = sorted(ts)

    sorted_ts = np.array(sorted(ts))
    grad = sorted_ts[1:] - sorted_ts[:-1]
    tracks_grad[tr] = grad

for tr in track_ids:
    valid_tracks = []
    th = 40000
    g = tracks_grad[tr]
    idx = np.where(g > th)[0]
    print('Valid indices: ', idx)

if TRIM_BAD_FRAMES:
  valid_track_lut = {}
  for tr in track_ids:
      valid_tracks = []
      if tr == 0:
          valid_tracks = track_lut[tr][14:]
      elif tr == 3:
          valid_tracks = track_lut[tr][:134]
      else:
          valid_tracks = track_lut[tr]
      valid_track_lut[tr] = valid_tracks
else:
  valid_track_lut = track_lut


def get_im_size(im, dim_max=384, multiple=32):
    H, W, _ = im.shape
    if W > H:
        if W > dim_max:
            sc = dim_max / W
            target_W = dim_max
        else:
            target_W = np.floor(W / multiple) * multiple
            sc = target_W / W
        target_H = int(np.round((H * sc) / multiple) * multiple)
        return [target_H, target_W]
    else:
        if H > dim_max:
            sc = dim_max / H
            target_H = dim_max
        else:
            target_H = np.floor(H / multiple) * multiple
            sc = target_H / H
        target_W = int(np.round((W * sc) / multiple) * multiple)
        return [target_H, target_W]


for track_id in track_ids:
    frames = valid_track_lut[track_id]

    hdf5_file_handles = []

    for idf, f in enumerate(frames):
        hdf5_file_handles.append(h5py.File(f, 'r'))
    test_in = np.array(hdf5_file_handles[0]['prediction/K'])
    if len(test_in) < 3:
        for f in hdf5_file_handles:
            test_f = np.array(hdf5_file_handles[0]['prediction/K'])
            if np.any(np.isnan(test_f)):
                continue
            else:
                print('found!!')
        print('corrupted!')
        continue
    makedirs(join(outpath, f'{track_names[track_id]}'), exist_ok=True)

    print(track_names[track_id])

    print('calculating NN_depth')
    depths = []
    conf = []
    mvs_depths = []
    for x in tqdm(range(len(hdf5_file_handles))):
        img = hdf5_file_handles[x]['prediction/img']
        if not img:
            img = Image.open(hdf5_file_handles[x]['prediction'].attrs['image_path'])
            stored_shape = hdf5_file_handles[x]['prediction'].attrs['image_shape']
            if not np.all(img.shape == stored_shape):
                img = imresize(np.asarray(img), stored_shape[:2], preserve_range=True)
            img = np.asarray(img).astype(float) / 255
        else:
            img = np.asarray(img) # Already a float32 array in range 0,1

        img_batch = torch.from_numpy(img).permute(2, 0, 1)[None, ...].float().cuda()
        with torch.no_grad():
            pred_d = model(img_batch)
            depths.append(pred_d.squeeze().cpu().numpy())
        mvs_depth = np.array(hdf5_file_handles[x]['prediction/mvs_depth'])
        mvs_depths.append(mvs_depth)
    print(img.shape)

    print('calculating scale')
    scales = []
    for x in tqdm(range(len(hdf5_file_handles))):
        nn_depth = depths[x]
        mvs_depth = mvs_depths[x]
        idx, idy = np.where(mvs_depth > 1e-3)
        scales.append(np.median(nn_depth[idx, idy] / mvs_depth[idx, idy]))
    s = np.mean(scales)
    print(s)

    print('saving per frame output')

    for idf, h5file in tqdm(enumerate(hdf5_file_handles)):
        img_orig = h5file['prediction/img']
        if not img_orig:
            img_orig = Image.open(h5file['prediction'].attrs['image_path'])
            img_orig = np.asarray(img_orig).astype(float) / 255
        else:
            img_orig = np.asarray(img_orig) # Already a float32 array in range 0,1

        max_dim = 384
        multiple = 32
        H, W, _ = img_orig.shape
        target_H, target_W = get_im_size(img_orig)

        img = imresize(img_orig, ((target_H, target_W)), preserve_range=True).astype(np.float32)

        T_G_1 = np.array(h5file['prediction/T_1_G'])
        T_G_1[:3, 3] *= s
        T_G_1 = np.linalg.inv(T_G_1)
        T_G_1 = T_G_1.astype(np.float32)
        depth_mvs = mvs_depths[idf] * s
        depth_mvs = depth_mvs.astype(np.float32)
        depth_mvs = imresize(depth_mvs, ([target_H, target_W]), preserve_range=True).astype(np.float32)
        in_1 = np.array(h5file['prediction/K'])
        in_1[0, 0] /= W / target_W
        in_1[1, 1] /= H / target_H
        in_1[0, 2] = (target_W - 1) / 2
        in_1[1, 2] = (target_H - 1) / 2
        in_1 = in_1.astype(np.float32)
        depth = depths[idf].astype(np.float32)
        depth = imresize(depth, ([target_H, target_W]), preserve_range=True).astype(np.float32)
        np.savez(join(outpath, track_names[track_id], 'frame_%05d.npz' % idf), img=img, pose_c2w=T_G_1,
                 depth_mvs=depth_mvs, intrinsics=in_1, depth_pred=depth, img_orig=img_orig, motion_seg=None)


# %%
