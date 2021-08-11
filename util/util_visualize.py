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
from third_party.util_colormap import heatmap_to_pseudo_color
KEYWORDS = ['depth', 'edgegradient', 'flow', 'img', 'rgb', 'image', 'edge', 'contour', 'softmask']


def detach_to_cpu(tensor):
    if type(tensor) == np.ndarray:
        return tensor
    else:
        if tensor.requires_grad:
            tensor.requires_grad = False
        tensor = tensor.cpu()
    return tensor.numpy()


class Converter:
    def __init__(self):
        pass

    @staticmethod
    def depth2img(tensor, normalize=True, disparity=True, eps=1e-6, **kargs):
        t = detach_to_cpu(tensor)
        assert len(t.shape) == 4
        assert t.shape[1] == 1
        t = 1 / (t + eps)
        # if normalize:
        max_v = np.max(t, axis=(2, 3), keepdims=True)
        min_v = np.min(t, axis=(2, 3), keepdims=True)
        t = (t - min_v) / (max_v - min_v + eps)
        #    return t
        # else:
        #    return t
        cs = []
        for b in range(t.shape[0]):
            c = heatmap_to_pseudo_color(t[b, 0, ...])
            cs.append(c[None, ...])
        cs = np.concatenate(cs, axis=0)
        cs = np.transpose(cs, [0, 3, 1, 2])
        return cs

    @staticmethod
    def edge2img(tensor, normalize=True, eps=1e-6, **kargs):
        t = detach_to_cpu(tensor)
        if np.max(t) > 1 or np.min(t) < 0:
            t = 1 / (1 + np.exp(-t))
        assert len(t.shape) == 4
        assert t.shape[1] == 1
        return t

    @staticmethod
    def image2img(tensor, **kargs):
        return Converter.img2img(tensor)

    @staticmethod
    def softmask2img(tensor, **kargs):
        t = detach_to_cpu(tensor)  # [:, None, ...]
        # t = #detach_to_cpu(tensor)
        return t

    @staticmethod
    def scenef2img(tensor, **kargs):
        t = detach_to_cpu(tensor.squeeze(3))
        assert len(t.shape) == 4
        return np.linalg.norm(t, ord=1, axis=-1, keepdims=True)

    @staticmethod
    def rgb2img(tensor, **kargs):
        return Converter.img2img(tensor)

    @staticmethod
    def img2img(tensor, **kargs):
        t = detach_to_cpu(tensor)
        if np.min(t) < -0.1:
            t = (t + 1) / 2
        elif np.max(t) > 1.5:
            t = t / 255
        return t

    @staticmethod
    def edgegradient2img(tensor, **kargs):
        t = detach_to_cpu(tensor)
        mag = np.max(abs(t))
        positive = np.where(t > 0, t, 0)
        positive /= mag
        negative = np.where(t < 0, abs(t), 0)
        negative /= mag
        rgb = np.concatenate((positive, negative, np.zeros(negative.shape)), axis=1)
        return rgb

    @staticmethod
    def flow2img(tensor, **kargs):
        t = detach_to_cpu(tensor)
        return t


def convert2rgb(tensor, key, **kargs):
    found = False
    for k in KEYWORDS:
        if k in key:
            convert = getattr(Converter, k + '2img')
            found = True
            break
    if not found:
        return None
    else:
        return convert(tensor, **kargs)


def is_key_image(key):
    """check if the given key correspondes to images

    Arguments:
        key {str} -- key of a data pack

    Returns:
        bool -- [True if the given key correspondes to an image]
    """

    for k in KEYWORDS:
        if k in key:
            return True
    return False


def parse_key(key):
    rkey = None
    found = False
    mode = None
    for k in KEYWORDS:
        if k in key:
            rkey = k
            found = True
            break
    if 'pred' in key:
        mode = 'pred'
    elif 'gt' in key:
        mode = 'gt'
    if not found:
        return None, None
    else:
        return rkey, mode
