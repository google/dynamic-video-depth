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

import configparser
from os.path import join, abspath, dirname


def get_project_config(file_path=None):
    if file_path is None:
        file_path = join(dirname(abspath(__file__)), '../configs/project_config.cfg')
    config = configparser.ConfigParser()
    config.read(file_path)
    assert 'Paths' in config
    config_dict = {}
    for k, v in config['Paths'].items():
        config_dict[k] = v
    return config_dict
