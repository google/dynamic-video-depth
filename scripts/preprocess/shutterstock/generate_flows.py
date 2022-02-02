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

import torch
import sys
import os
from os.path import join, basename
sys.path.append('./third_party/RAFT')
sys.path.append('./third_party/RAFT/core')
from raft import RAFT
import numpy as np
import torch.nn.functional as F
from functools import lru_cache
from glob import glob
import argparse
from tqdm import tqdm
import subprocess

try:
    import cv2
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'opencv-python'])
finally:
    import cv2

from skimage.transform import resize as imresize


data_list_root = "./datafiles/shutterstock/frames_midas"
outpath = './datafiles/shutterstock/flow_pairs'


def resize_flow(flow, size):
    resized_width, resized_height = size
    H, W = flow.shape[:2]
    scale = np.array((resized_width / float(W), resized_height / float(H))).reshape(
        1, 1, -1
    )
    resized = cv2.resize(
        flow, dsize=(resized_width, resized_height), interpolation=cv2.INTER_CUBIC
    )
    resized *= scale
    return resized


def get_oob_mask(flow_1_2):
    H, W, _ = flow_1_2.shape
    hh, ww = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float())
    coord = torch.zeros([H, W, 2])
    coord[..., 0] = ww
    coord[..., 1] = hh
    target_range = coord + flow_1_2
    m1 = (target_range[..., 0] < 0) + (target_range[..., 0] > W - 1)
    m2 = (target_range[..., 1] < 0) + (target_range[..., 1] > H - 1)
    return (m1 + m2).float().numpy()


def backward_flow_warp(im2, flow_1_2):
    H, W, _ = im2.shape
    hh, ww = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float())
    coord = torch.zeros([1, H, W, 2])
    coord[0, ..., 0] = ww
    coord[0, ..., 1] = hh
    sample_grids = coord + flow_1_2[None, ...]
    sample_grids[..., 0] /= (W - 1) / 2
    sample_grids[..., 1] /= (H - 1) / 2
    sample_grids -= 1
    im = torch.from_numpy(im2).float().permute(2, 0, 1)[None, ...]
    out = F.grid_sample(im, sample_grids, align_corners=True)
    o = out[0, ...].permute(1, 2, 0).numpy()
    return o


def get_L2_error_map(v1, v2):
    return np.linalg.norm(v1 - v2, axis=-1)


def load_RAFT():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args(['--model', './third_party/RAFT/models/raft-sintel.pth', '--path', './'])
    net = torch.nn.DataParallel(RAFT(args).cuda())
    net.load_state_dict(torch.load(args.model))
    return net


@lru_cache(maxsize=200)
def read_frame_data(key, frame_id):
    data = np.load(join(data_list_root, key, 'frame_%05d.npz' % frame_id), allow_pickle=True)
    data_dict = {}
    for k in data.keys():
        data_dict[k] = data[k]
    return data_dict

net = load_RAFT()

def generate_pair_data(key, frame_id_1, frame_id_2_list):
    im1_data = read_frame_data(key, frame_id_1)
    im2_data_list = [read_frame_data(key, f) for f in frame_id_2_list]

    im1 = im1_data['img_orig'] * 255
    im2_list = [i['img_orig'] * 255 for i in im2_data_list]
    im1 = imresize(im1, (288, 512), anti_aliasing=True)
    im2_list = [imresize(i, (288, 512), anti_aliasing=True) for i in im2_list]

    im1_list = [im1] * len(im2_list)
    
    def image_list_to_cuda_batch(im_list):
        reorder = np.array(im_list).transpose(0, 3, 1, 2)
        to_cuda = torch.from_numpy(reorder.astype(np.float32)).cuda()
        return to_cuda

    def cuda_batch_to_numpy_batch(cuda_batch):
        return cuda_batch.permute(0, 2, 3, 1).cpu().numpy()

    im1_batch = image_list_to_cuda_batch(im1_list)
    im2_batch = image_list_to_cuda_batch(im2_list)

    with torch.no_grad():
        flow_low, flow_up = net(image1=im1_batch, image2=im2_batch, iters=20, test_mode=True)
        flow_1_2_batch = cuda_batch_to_numpy_batch(flow_up)

        flow_low, flow_up = net(image1=im2_batch, image2=im1_batch, iters=20, test_mode=True)
        flow_2_1_batch = cuda_batch_to_numpy_batch(flow_up)

    H, W, _ = im1_data['img'].shape
    for j, frame_id_2 in enumerate(frame_id_2_list):
        flow_1_2 = resize_flow(flow_1_2_batch[j,...], [W, H])
        flow_2_1 = resize_flow(flow_2_1_batch[j,...], [W, H])

        warp_flow_1_2 = backward_flow_warp(flow_1_2, flow_2_1)  # using latter to sample former
        err_1 = np.linalg.norm(warp_flow_1_2 + flow_2_1, axis=-1)
        mask_1 = np.where(err_1 > 1, 1, 0)
        oob_mask_1 = get_oob_mask(flow_2_1)
        mask_1 = np.clip(mask_1 + oob_mask_1, a_min=0, a_max=1)
        warp_flow_2_1 = backward_flow_warp(flow_2_1, flow_1_2)
        err_2 = np.linalg.norm(warp_flow_2_1 + flow_1_2, axis=-1)
        mask_2 = np.where(err_2 > 1, 1, 0)
        oob_mask_2 = get_oob_mask(flow_1_2)
        mask_2 = np.clip(mask_2 + oob_mask_2, a_min=0, a_max=1)
        save_dict = {}
        save_dict['flow_1_2'] = flow_1_2.astype(np.float32)
        save_dict['flow_2_1'] = flow_2_1.astype(np.float32)
        save_dict['mask_1'] = mask_1.astype(np.uint8)
        save_dict['mask_2'] = mask_2.astype(np.uint8)
        save_dict['frame_id_1'] = frame_id_1
        save_dict['frame_id_2'] = frame_id_2
        np.savez(join(outpath, key, f'flowpair_{frame_id_1:05d}_{frame_id_2:05d}.npz'), **save_dict)

# %%


track_names = sorted(glob(join(data_list_root, '*')))
track_names = [basename(x) for x in track_names]
track_ids = np.arange(len(track_names))

# %%
for track_id in tqdm(track_ids):
    key = track_names[track_id]
    print(key)
    l = len(sorted(glob(join(data_list_root, key, 'frame_*.npz'))))
    os.makedirs(join(outpath, track_names[track_id]), exist_ok=True)
    MAX_GAP = 8
    for k in tqdm(range(l-1)):
        gaps = range(min(l-k-1, MAX_GAP))
        end_frames = [k + g + 1 for g in gaps]
        generate_pair_data(key, k, end_frames)
