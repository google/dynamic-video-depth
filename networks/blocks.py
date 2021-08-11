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

from torch import nn
import torch


class PeriodicEmbed(nn.Module):
    def __init__(self, max_freq=5, N_freq=4, linspace=True):
        super().__init__()
        self.embed_functions = [torch.cos, torch.sin]
        if linspace:
            self.freqs = torch.linspace(1, max_freq + 1, steps=N_freq)
        else:
            exps = torch.linspace(0, N_freq - 1, steps=N_freq)
            self.freqs = 2**exps

    def forward(self, x):
        output = [x]
        for f in self.embed_functions:
            for freq in self.freqs:
                output.append(f(freq * x))
        return torch.cat(output, 1)


class DoubleConv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride=1,
                 padding=0, dilation=1, norm='weight', activation='relu', pad_type='zero', use_bias=True, **kargs):
        super().__init__()
        self.model = nn.Sequential(Conv2dBlock(input_dim, output_dim, kernel_size, stride,
                                               padding, dilation, norm, activation, pad_type, use_bias),
                                   Conv2dBlock(output_dim, output_dim, kernel_size, stride,
                                               padding, dilation, norm, activation, pad_type, use_bias))

    def forward(self, x):
        return self.model(x)


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, dilation=1, norm='weight', activation='relu', pad_type='zero', use_bias=True, *args, **karg):
        super(Conv2dBlock, self).__init__()

        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride,
                              padding=0, dilation=dilation, bias=use_bias)

        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'batch':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'inst':
            self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=False)
        elif norm == 'ln':
            self.norm = nn.LayerNorm(norm_dim)
        elif norm == 'none':
            self.norm = nn.Identity()
        elif norm == 'weight':
            self.conv = nn.utils.weight_norm(self.conv)
            self.norm = nn.Identity()
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = nn.Identity()
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        x = self.conv(self.pad(x))
        x = self.norm(x)
        x = self.activation(x)
        return x


class ResConv2DBlock(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, stride=1, padding=0, dilation=1,
                 norm='weight', activation='relu', pad_type='zero', use_bias=True):
        model = []
        model += [Conv2dBlock(dim_in, dim_out, kernel_size, stride, padding, dilation=dilation, norm=norm,
                              activation=activation, pad_type=pad_type, use_bias=use_bias)]
        model += [Conv2dBlock((dim_in + dim_out) // 2, dim_out, kernel_size, stride, padding, dilation=dilation, norm=norm,
                              activation=activation, pad_type=pad_type, use_bias=use_bias)]
        if dim_in != dim_out:
            self.skip = Conv2dBlock(dim_in, dim_out, 1, stride, padding, dilation=dilation, norm=norm,
                                    activation=activation, pad_type=pad_type, use_bias=use_bias)
        else:
            self.skip = nn.Indentity()
        self.model = nn.Sequential(*model)

    def forward(self, x):
        res = self.skip(x)
        out = self.model(x)
        return out + res
