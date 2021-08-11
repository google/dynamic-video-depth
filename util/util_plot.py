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

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from third_party.util_colormap import turbo_colormap_data
import matplotlib
from matplotlib.colors import ListedColormap

matplotlib.cm.register_cmap('turbo', cmap=ListedColormap(turbo_colormap_data))


def save_img_tensor_anim(img_tensor, path, markers=None, dpi=80, fps=30):
    fig = plt.figure(dpi=dpi)
    f = plt.imshow(img_tensor[0, ...].numpy())
    if markers is not None:
        s = plt.plot(markers[:, 0, 0], markers[:, 1, 0], 'w+', markersize=5)
    plt.axis('off')
    plt.tight_layout(h_pad=0, w_pad=0)
    plt.margins(0.0)
    fig.subplots_adjust(0, 0, 1, 1)

    def anim(step):
        f.set_data(img_tensor[step, ...])
        if markers is not None:
            s[0].set_data([markers[:, 0, step], markers[:, 1, step]])
    all_steps = min(img_tensor.shape[0], markers.shape[-1])
    line_ani = animation.FuncAnimation(fig, anim, all_steps,
                                       interval=1000 / fps, blit=False)
    line_ani.save(path)
    plt.close()


def save_depth_tensor_anim(depth_tensor, path, minv=None, maxv=None, markers=None, dpi=80, fps=30):

    fig = plt.figure()
    f = plt.imshow(1 / depth_tensor[0, 0, ...].numpy(), cmap='turbo', vmax=1 / minv, vmin=1 / maxv)
    s = plt.plot(markers[:, 0, 0], markers[:, 1, 0], 'w+', markersize=5)

    def anim(step):
        f.set_data(1 / depth_tensor[step, 0, ...].numpy())
        if markers is not None:
            s[0].set_data([markers[:, 0, step], markers[:, 1, step]])

    all_steps = markers.shape[-1]

    line_ani = animation.FuncAnimation(fig, anim, all_steps,
                                       interval=1000 / fps, blit=False)
    line_ani.save(path)
    plt.close()
