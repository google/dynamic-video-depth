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
from os.path import join
from os import makedirs
import numpy as np
from functools import lru_cache
from glob import glob
from tqdm import tqdm


data_list_root = "./datafiles/davis_processed/frames_midas/"

flow_path = './datafiles/davis_processed/flow_pairs/'

save_path_root = './datafiles/davis_processed/sequences_select_pairs_midas/'


@lru_cache(maxsize=1024)
def read_frame_data(key, frame_id):
    data = np.load(join(data_list_root, key, 'frame_%05d.npz' % frame_id))
    data_dict = {}
    for k in data.keys():
        data_dict[k] = data[k]
    return data_dict


@lru_cache(maxsize=1024)
def read_flow_data(key, frame_id_1, frame_id_2):
    data = np.load(join(flow_path, key, f'flowpair_{frame_id_1:05d}_{frame_id_2:05d}.npz'), allow_pickle=True)
    data_dict = {}
    for k in data.keys():
        data_dict[k] = data[k]
    return data_dict


def prepare_pose_dict_one_way(im1_data, im2_data):
    # return R_1 R_2 t_1 t_2
    cam_pose_c2w_1 = im1_data['pose_c2w']
    R_1 = cam_pose_c2w_1[:3, :3]
    t_1 = cam_pose_c2w_1[:3, 3]

    cam_pose_c2w_2 = im2_data['pose_c2w']
    R_2 = cam_pose_c2w_2[:3, :3]
    t_2 = cam_pose_c2w_2[:3, 3]
    K = im1_data['intrinsics']

    # for network use:
    R_1_tensor = torch.zeros([1, 1, 1, 3, 3])
    R_1_T_tensor = torch.zeros([1, 1, 1, 3, 3])
    R_2_tensor = torch.zeros([1, 1, 1, 3, 3])
    R_2_T_tensor = torch.zeros([1, 1, 1, 3, 3])
    t_1_tensor = torch.zeros([1, 1, 1, 1, 3])
    t_2_tensor = torch.zeros([1, 1, 1, 1, 3])
    K_tensor = torch.zeros([1, 1, 1, 3, 3])
    K_inv_tensor = torch.zeros([1, 1, 1, 3, 3])
    R_1_tensor[0, ..., :, :] = torch.from_numpy(R_1.T)
    R_2_tensor[0, ..., :, :] = torch.from_numpy(R_2.T)
    R_1_T_tensor[0, ..., :, :] = torch.from_numpy(R_1)
    R_2_T_tensor[0, ..., :, :] = torch.from_numpy(R_2)
    t_1_tensor[0, ..., :] = torch.from_numpy(t_1)
    t_2_tensor[0, ..., :] = torch.from_numpy(t_2)
    K_tensor[..., :, :] = torch.from_numpy(K.T)
    K_inv_tensor[..., :, :] = torch.from_numpy(np.linalg.inv(K).T)

    pose_dict = {}
    pose_dict['R_1'] = R_1_tensor
    pose_dict['R_2'] = R_2_tensor
    pose_dict['R_1_T'] = R_1_T_tensor
    pose_dict['R_2_T'] = R_2_T_tensor
    pose_dict['t_1'] = t_1_tensor
    pose_dict['t_2'] = t_2_tensor
    pose_dict['K'] = K_tensor
    pose_dict['K_inv'] = K_inv_tensor
    return pose_dict


def collate_sequence_fix_gap(key, seq_list, gap=1):
    sequential_pairs_start = seq_list
    sequential_pairs_end = seq_list + gap
    list_of_pairs_select = [(x, y) for x, y in zip(sequential_pairs_start, sequential_pairs_end)]
    sequential_data = collate_pairs(key, list_of_pairs_select)
    flow_1_2_batch = sequential_data['flow_1_2']
    flow_1_2_batch = flow_1_2_batch.permute([0, 3, 1, 2])

    return sequential_data


def collate_pairs(key, list_of_pairs):

    dict_of_list = {}
    for idp, pair in enumerate(list_of_pairs):

        dd = datadict_from_pair(key, pair)
        for k, v in dd.items():
            if k not in dict_of_list.keys():
                dict_of_list[k] = []
            dict_of_list[k].append(v)

    for k in dict_of_list.keys():
        dict_of_list[k] = torch.cat(dict_of_list[k], dim=0)
    return dict_of_list


def datadict_from_pair(key, pair):
    frame_id_1, frame_id_2 = pair
    im1_data = read_frame_data(key, pair[0])
    im2_data = read_frame_data(key, pair[1])
    fid_1, fid_2 = sorted([frame_id_1, frame_id_2])
    flow_data_dict = read_flow_data(key, fid_1, fid_2)
    if fid_1 == frame_id_1:
        flow_1_2 = flow_data_dict['flow_1_2']
        flow_2_1 = flow_data_dict['flow_2_1']
        mask_1 = flow_data_dict['mask_1']
        mask_2 = flow_data_dict['mask_2']
    else:
        flow_1_2 = flow_data_dict['flow_2_1']
        flow_2_1 = flow_data_dict['flow_1_2']
        mask_1 = flow_data_dict['mask_1']
        mask_2 = flow_data_dict['mask_2']
    pose_dict = prepare_pose_dict_one_way(im1_data, im2_data)
    gt_depth_1 = torch.from_numpy(im1_data['depth_mvs']).float()
    pred_depth_1 = torch.from_numpy(im1_data['depth_pred']).float()
    H, W = gt_depth_1.shape
    depth_1_tensor = torch.zeros([1, 1, H, W])
    depth_1_tensor[0, 0, ...] = gt_depth_1
    depth_1_tensor_p = torch.zeros([1, 1, H, W])
    depth_1_tensor_p[0, 0, ...] = pred_depth_1

    flow_1_2 = torch.from_numpy(flow_1_2).float()[None, ...]
    flow_2_1 = torch.from_numpy(flow_2_1).float()[None, ...]
    mask_1 = torch.from_numpy(mask_1).float()
    mask_2 = torch.from_numpy(mask_2).float()
    mask_1 = 1 - torch.ceil(mask_1)[None, ..., None, None]
    mask_2 = 1 - torch.ceil(mask_2)[None, ..., None, None]
    img_1 = torch.from_numpy(im1_data['img']).float()[None, ...]
    img_2 = torch.from_numpy(im2_data['img']).float()[None, ...]
    fid_1 = pair[0]
    fid_2 = pair[1]
    if 'motion_seg' in im1_data.keys():
        motion_seg = torch.from_numpy(im1_data['motion_seg'])[None, ..., None, None].float()
    else:
        motion_seg = mask_2
    samples = {}
    for k in pose_dict:
        samples[k] = pose_dict[k]
    samples['img_1'] = img_1
    samples['img_2'] = img_2
    samples['depth_1'] = depth_1_tensor
    samples['flow_1_2'] = flow_1_2
    samples['flow_2_1'] = flow_2_1
    samples['mask_1'] = mask_1
    samples['mask_2'] = mask_2
    samples['motion_seg_1'] = motion_seg
    samples['depth_pred_1'] = depth_1_tensor_p
    samples['fid_1'] = torch.FloatTensor([fid_1])
    samples['fid_2'] = torch.FloatTensor([fid_2])
    return samples


if __name__ == '__main__':
    track_names = sorted(glob(join(data_list_root, '*')))
    track_names = ['train', 'dog']
    for key in track_names:
        all_frames = sorted(glob(join(data_list_root, key, '*.npz')))
        gaps = [1, 2, 3, 4, 5, 6, 7, 8]
        bs = 1
        save_path = join(save_path_root, key, '001')
        print(key)
        makedirs(save_path, exist_ok=True)

        print('saving...')

        for gap in tqdm(gaps):
            fids = np.arange(len(all_frames) - bs - gap)
            cnt = 0
            for f in fids:
                seq_list_forward = np.arange(f, f + bs)
                sequence = collate_sequence_fix_gap(key, seq_list_forward, gap=gap,)
                torch.save(sequence, join(save_path, f'shuffle_False_gap_{gap:02d}_sequence_{cnt:05d}.pt'))
                cnt += 1
