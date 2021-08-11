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

from PIL import Image
import numpy as np
from skimage.transform import resize as imresize


def read_image(path, load_alpha=False):
    im = np.asarray(Image.open(path))
    dims = len(im.shape)
    if dims == 2:
        return im
    elif dims == 3:
        if im.shape[-1] == 3:
            return im
        elif load_alpha:
            return im
        else:
            return im[..., :3]
    else:
        raise ValueError(f'invalid dimensions encoutered. Only except dims 2,3 but encoutered {dims}')


def resize_image(im, size=None, scale=None):
    H, W = im.shape[:2]
    if scale:
        th = H // scale
        tw = W // scale
        s = (th, tw)
    else:
        s = size
    im = imresize(im, s)
    return im


def hwc2chw(im):
    dims = len(im.shape)
    if dims == 2:
        return im[None, ...]
    elif dims == 3:
        return np.transpose(im, (2, 0, 1))
    else:
        raise ValueError(f'invalid dimensions encoutered. Only except dims 2,3 but encoutered {dims}')
