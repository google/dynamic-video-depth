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
import torch.utils.data as data
import torch


class Dataset(data.Dataset):
    @classmethod
    def add_arguments(cls, parser):
        return parser, set()

    def __init__(self, opt, mode='train', model=None):
        super().__init__()
        self.opt = opt
        self.mode = mode

    @staticmethod
    def convert_to_float32(sample_loaded):
        for k, v in sample_loaded.items():
            if isinstance(v, np.ndarray):
                sample_loaded[k] = torch.from_numpy(v).float()
