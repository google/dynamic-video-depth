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
from .base_dataset import Dataset as base_dataset
from glob import glob
from os.path import join
import torch


class Dataset(base_dataset):

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument('--cache', action='store_true', help='cache the data into ram')
        parser.add_argument('--subsample', action='store_true', help='subsample the video in time')
        parser.add_argument('--track_id', default='train', type=str, help='the track id to load')
        parser.add_argument('--overfit', action='store_true', help='overfit and see if things works')
        parser.add_argument('--gaps', type=str, default='1,2,3,4', help='gaps for sequences')
        parser.add_argument('--repeat', type=int, default=1, help='number of repeatition')
        parser.add_argument('--select', action='store_true', help='pred')
        return parser, set()

    def __init__(self, opt, mode='train', model=None):
        super().__init__(opt, mode, model)
        self.mode = mode
        assert mode in ('train', 'vali')

        data_root = './datafiles/davis_processed'
        # tracks = sorted(glob(join(data_root, 'frames_midas', '*')))
        # tracks = [x.split('/')[-1] for x in tracks]
        track_name = opt.track_id  # tracks[opt.track_id]
        if model is None:
            self.required = ['img', 'flow']
            self.preproc = None
        elif mode == 'train':
            self.required = model.requires
            self.preproc = model.preprocess
        else:
            self.required = ['img']
            self.preproc = model.preprocess

        frame_prefix = 'frames_midas'
        seq_prefix = 'sequences_select_pairs_midas'

        if mode == 'train':

            if self.opt.subsample:
                data_path = join(data_root, seq_prefix, track_name, 'subsample')
            else:
                data_path = join(data_root, seq_prefix, track_name, '%03d' % 1)

            gaps = opt.gaps.split(',')
            gaps = [int(x) for x in gaps]
            self.file_list = []
            for g in gaps:

                file_list = sorted(glob(join(data_path, f'shuffle_False_gap_{g:02d}_*.pt')))
                self.file_list += file_list

            frame_data_path = join(data_root, frame_prefix, track_name)
            self.n_frames = len(sorted(glob(join(frame_data_path, '*.npz')))) + 0.0

        else:
            data_path = join(data_root, frame_prefix, track_name)
            self.file_list = sorted(glob(join(data_path, '*.npz')))
            self.n_frames = len(self.file_list) + 0.0

    def __len__(self):
        if self.mode != 'train':
            return len(self.file_list)
        else:
            return len(self.file_list) * self.opt.repeat

    def __getitem__(self, idx):
        sample_loaded = {}
        if self.opt.overfit:
            idx = idx % self.opt.capat
        else:
            idx = idx % len(self.file_list)

        if self.opt.subsample:
            unit = 2.0
        else:
            unit = 1.0

        if self.mode == 'train':

            dataset = torch.load(self.file_list[idx])

            _, H, W, _ = dataset['img_1'].shape
            dataset['img_1'] = dataset['img_1'].permute([0, 3, 1, 2])
            dataset['img_2'] = dataset['img_2'].permute([0, 3, 1, 2])
            ts = dataset['fid_1'].reshape([-1, 1, 1, 1]).expand(-1, -1, H, W) / self.n_frames
            ts2 = dataset['fid_2'].reshape([-1, 1, 1, 1]).expand(-1, -1, H, W) / self.n_frames
            for k in dataset:
                if type(dataset[k]) == list:
                    continue
                sample_loaded[k] = dataset[k].float()
            sample_loaded['time_step'] = unit / self.n_frames
            sample_loaded['time_stamp_1'] = ts.float()
            sample_loaded['time_stamp_2'] = ts2.float()
            sample_loaded['frame_id_1'] = np.asarray(dataset['fid_1'])
            sample_loaded['frame_id_2'] = np.asarray(dataset['fid_2'])

        else:
            dataset = np.load(self.file_list[idx])
            H, W, _ = dataset['img'].shape
            sample_loaded['time_stamp_1'] = np.ones([1, H, W]) * idx / self.n_frames
            sample_loaded['img'] = np.transpose(dataset['img'], [2, 0, 1])
            sample_loaded['frame_id_1'] = idx

            sample_loaded['time_step'] = unit / self.n_frames
            sample_loaded['depth_pred'] = dataset['depth_pred'][None, ...]
            sample_loaded['cam_c2w'] = dataset['pose_c2w']
            sample_loaded['K'] = dataset['intrinsics']
            sample_loaded['depth_mvs'] = dataset['depth_mvs'][None, ...]
            # add decomposed cam mat
            cam_pose_c2w_1 = dataset['pose_c2w']
            R_1 = cam_pose_c2w_1[:3, :3]
            t_1 = cam_pose_c2w_1[:3, 3]
            K = dataset['intrinsics']

            # for network use:
            R_1_tensor = np.zeros([1, 1, 3, 3])
            R_1_T_tensor = np.zeros([1, 1, 3, 3])
            t_1_tensor = np.zeros([1, 1, 1, 3])
            K_tensor = np.zeros([1, 1, 3, 3])
            K_inv_tensor = np.zeros([1, 1, 3, 3])
            R_1_tensor[..., :, :] = R_1.T
            R_1_T_tensor[..., :, :] = R_1
            t_1_tensor[..., :] = t_1
            K_tensor[..., :, :] = K.T
            K_inv_tensor[..., :, :] = np.linalg.inv(K).T

            sample_loaded['R_1'] = R_1_tensor
            sample_loaded['R_1_T'] = R_1_T_tensor
            sample_loaded['t_1'] = t_1_tensor
            sample_loaded['K'] = K_tensor
            sample_loaded['K_inv'] = K_inv_tensor
        sample_loaded['pair_path'] = self.file_list[idx]
        self.convert_to_float32(sample_loaded)
        return sample_loaded
