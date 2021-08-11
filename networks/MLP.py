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
from .blocks import PeriodicEmbed


class EmbededMLP(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, depth=3, width=64, N_freq=8, skip=3, act_fn=nn.functional.leaky_relu, output_act=None, norm=None, init_val=None):
        super().__init__()
        self.embed = PeriodicEmbed(N_freq=N_freq, linspace=False)
        N_input_channel = in_ch + in_ch * 2 * N_freq
        self.layers = []
        self.skip = skip
        self.layers.append(DenseLayer(N_input_channel, width, act_fn, norm))
        for d in range(depth - 1):
            if (d + 1) % skip == 0 and d > 0:
                self.layers.append(DenseLayer(width + N_input_channel, width, act_fn, norm))
            else:
                self.layers.append(DenseLayer(width, width, act_fn, norm))
        self.layers.append(DenseLayer(width, out_ch, output_act, norm=None))
        for idl, l in enumerate(self.layers):
            self.add_module(f'layer_{idl:03d}', l)

        if init_val is not None:
            self.layers[-1].linear.bias.data.fill_(init_val)

    def forward(self, x):
        x = self.embed(x)
        embed = x

        for idl, l in enumerate(self.layers):
            if idl % self.skip == 0 and idl > 0 and idl < len(self.layers) - 1:
                x = torch.cat([x, embed], -1)
            x = l(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_ch=64, out_ch=3, depth=3, width=64, act_fn=nn.functional.relu, output_act=None, norm=None):
        super().__init__()
        layers = []
        layers.append(DenseLayer(in_ch, width, act_fn, norm))
        for d in range(depth - 1):
            layers.append(DenseLayer(width, width, act_fn, norm))
        layers.append(DenseLayer(width, out_ch, output_act, norm=None))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DenseLayer(nn.Module):
    def __init__(self, in_ch, out_ch, act_fn=None, norm=None):
        super().__init__()
        self.linear = nn.Linear(in_ch, out_ch)
        if act_fn is None:
            self.act_fn = nn.Identity()
        else:
            self.act_fn = act_fn
        if norm is None:
            self.norm = nn.Identity()
        else:
            self.norm = norm

    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x)
        x = self.act_fn(x)
        return x
