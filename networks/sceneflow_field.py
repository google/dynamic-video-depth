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
from torch import nn
from .blocks import PeriodicEmbed, Conv2dBlock


class SceneFlowFieldNet(nn.Module):
    def __init__(self, time_dependent=True, N_freq_xyz=0, N_freq_t=0, output_dim=3, net_width=32, n_layers=3, activation='lrelu', norm='none'):
        super().__init__()
        N_input_channel_xyz = 3 + 3 * 2 * N_freq_xyz
        N_input_channel_t = 1 + 1 * 2 * N_freq_t
        N_input_channel = N_input_channel_xyz + N_input_channel_t if time_dependent else N_input_channel_xyz
        if N_freq_xyz == 0:
            xyz_embed = nn.Identity()
        else:
            xyz_embed = PeriodicEmbed(max_freq=N_freq_xyz, N_freq=N_freq_xyz)
        if N_freq_t == 0:
            t_embed = nn.Identity()
        else:
            t_embed = PeriodicEmbed(max_freq=N_freq_t, N_freq=N_freq_t)
        convs = [Conv2dBlock(N_input_channel, net_width, 1, 1, norm=norm, activation=activation)]
        for i in range(n_layers):
            convs.append(Conv2dBlock(net_width, net_width, 1, 1, norm=norm, activation=activation))
        convs.append(Conv2dBlock(net_width, output_dim, 1, 1, norm='none', activation='none'))
        self.convs = nn.Sequential(*convs)
        self.t_embed = t_embed
        self.xyz_embed = xyz_embed
        self.time_dependent = time_dependent

    def forward(self, x, t=None):
        x = x.contiguous()
        if t is None and self.time_dependent:
            raise ValueError
        xyz_embedded = self.xyz_embed(x)
        if self.time_dependent:
            t_embedded = self.t_embed(t)
            input_feat = torch.cat([t_embedded, xyz_embedded], 1)
        else:
            input_feat = xyz_embedded
        return self.convs(input_feat)
