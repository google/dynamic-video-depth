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

from torch.multiprocessing import Pool
from os.path import join, dirname
from os import makedirs
import atexit
from util.util_config import get_project_config


class BaseVisualizer():
    """
    Async Visulization Worker
    """

    def __init__(self, n_workers=4):
        # read global configs
        self.cfg = get_project_config()
        if n_workers == 0:
            pool = None
        elif n_workers > 0:
            pool = Pool(n_workers)
        else:
            raise ValueError(n_workers)
        self.pool = pool

        def cleanup():
            if pool:
                pool.close()
                pool.join()
        atexit.register(cleanup)

    def visualize(self, pack, batch_idx, outdir):
        if self.pool:
            self.pool.apply_async(
                self._visualize,
                [pack, batch_idx, outdir, self.cfg],
                error_callback=self._error_callback
            )
        else:
            self._visualize(pack, batch_idx, outdir, self.cfg)

    @staticmethod
    def _visualize(pack, batch_idx, param_f, outdir):
        # main visualiztion thread.
        raise NotImplementedError

    @staticmethod
    def _error_callback(e):
        print(str(e))
