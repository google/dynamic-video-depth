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

from util.util_visualize import convert2rgb, is_key_image
from torch.multiprocessing import Pool
from util.util_flow import flow2img
from os.path import join
import atexit
import numpy as np
from PIL import Image


class HTMLVisualizer():
    """
    Async Visulization Worker
    """

    def __init__(self, html_logger, n_workers=4):
        # read global configs
        if n_workers == 0:
            pool = None
        elif n_workers > 0:
            pool = Pool(n_workers)
        else:
            raise ValueError(n_workers)
        self.pool = pool
        self.html_logger = html_logger
        self.header_lut = None

        def cleanup():
            if pool:
                pool.close()
                pool.join()
        atexit.register(cleanup)

    def visualize(self, pack, batch_idx, outdir, is_test=False):
        # first append to the shared content
        # then launch the subprocesses for dumping images.
        # b_size = pack['batch_size']
        epoch_folder = outdir.split('/')[-1]
        if is_test:
            epoch_folder = None
        self.prepare_HTML_string(pack, batch_idx, epoch_folder)

        if self.pool:
            self.pool.apply_async(
                self._visualize,
                [pack, batch_idx, outdir],
                error_callback=self._error_callback
            )
        else:
            self._visualize(pack, batch_idx, outdir)
        # prepare HTML string

    def prepare_HTML_string(self, pack, batch_idx, epoch_folder):
        if self.html_logger.epoch_content is None:
            self.html_logger.epoch_content = {}
            #self.html_logger.epoch_content['header'] = ''
            header = ''
            for k in sorted(list(pack.keys())):
                # get the ones that are useful:
                if is_key_image(k):
                    header += f"<th>{k}</th>\n"
            self.html_logger.epoch_content['header'] = header
            self.html_logger.epoch_content['content'] = ''
        content = ''
        batch_size = pack['batch_size']
        tags = pack['tags'] if 'tags' in pack.keys() else None
        for b in range(batch_size):
            content += "<tr>\n"
            for k in sorted(list(pack.keys())):
                # get the ones that are useful:
                if is_key_image(k):
                    if tags is not None:
                        prefix = '%s_%s.png' % (k, tags[b])
                    else:
                        prefix = '%s_%04d_%04d.png' % (k, batch_idx, b)
                    if epoch_folder is not None:
                        link = join(epoch_folder, prefix)
                    else:
                        link = prefix
                    # html is at outdir/../, so link is epochXXX/
                    content += f"<td><img src=\"{link}\" style=\"max-width:100%;height:auto;\"></td>\n"
            content += "</tr>\n"
        self.html_logger.epoch_content['content'] += content

    @staticmethod
    def _visualize(pack, batch_idx, outdir):
        # this thread saves the packed tensor into individual images
        batch_size = pack['batch_size']
        tags = pack['tags'] if 'tags' in pack.keys() else None
        for k, v in pack.items():
            rgb_tensor = convert2rgb(v, k)
            if rgb_tensor is None:
                continue
            for b in range(batch_size):
                if 'flow' in k:
                    img = flow2img(rgb_tensor[b, :, :, :]) / 255
                else:
                    img = np.squeeze(np.transpose(rgb_tensor[b, :, :, :], (1, 2, 0)))
                if tags is not None:
                    prefix = '%s_%s.png' % (k, tags[b])
                else:
                    prefix = '%s_%04d_%04d.png' % (k, batch_idx, b)
                Image.fromarray((img * 255).astype(np.uint8)).save(join(outdir, prefix), 'PNG')

    @staticmethod
    def _error_callback(e):
        print(str(e))
