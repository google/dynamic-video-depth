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
from .blocks import Conv2dBlock, DoubleConv2dBlock


class FCNUnet(nn.Module):
    #
    def __init__(self, conv_setup, n_down=4, feat=32, block_type='conv',
                 down_sample_type='avgpool', in_channel=2, out_channel=64, dialated_pool=False, output_activation=None):
        super().__init__()
        assert down_sample_type in ['avgpool', 'maxpool', 'none']
        if block_type == 'conv':
            Block = Conv2dBlock
        elif block_type == 'double_conv':
            Block = DoubleConv2dBlock
        else:
            raise NotImplementedError(f'block type {block_type} not supported')
        # 2x downsampling using avgpool
        if down_sample_type == 'avgpool':
            self.down_sample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        elif down_sample_type == 'maxpool':
            self.down_sample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.down_sample = None

        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        # this is not Implemented yet

        self.n_down = n_down
        self.down_conv = []
        self.up_conv = []

        ch_in = in_channel
        ch_out = feat
        for k in range(n_down):
            self.down_conv += [Block(ch_in, ch_out, kernel_size=3, padding=1, **conv_setup)]
            self.add_module('down_%02d' % k, self.down_conv[-1])
            ch_in = ch_out
            ch_out = ch_out * 2
        self.mid_conv = Block(ch_in, ch_in, kernel_size=3, padding=1, **conv_setup)

        for k in range(n_down - 1):
            self.up_conv += [Block(ch_in * 2, ch_in // 2,
                                   padding=1, kernel_size=3, **conv_setup)]
            self.add_module('up_%04d' % k, self.up_conv[-1])
            ch_in = ch_in // 2
        # This is for matching original unet implementation.
        self.up_conv += [Block(ch_in * 2, ch_in, padding=1,
                               kernel_size=3, **conv_setup)]
        self.add_module('up_%04d' % (k + 1), self.up_conv[-1])

        conv_setup['activation'] = 'none'
        conv_setup['norm'] = 'none'
        self.output_conv = Conv2dBlock(
            ch_in, out_channel, kernel_size=1, **conv_setup)
        # self.add_module('output', self.up_conv[-1])
        if output_activation == 'tanh':
            self.final_act = nn.Tanh()
        elif output_activation == 'sigmoid':
            self.final_act = nn.Sigmoid()
        else:
            self.final_act = nn.Identity()

    def forward(self, x):
        feat = []
        for module in self.down_conv:
            x = module(x)
            feat.append(x)
            x = self.down_sample(x)
        x = self.mid_conv(x)
        for idm, module in enumerate(self.up_conv):
            up_x = self.upsample(x)
            f = feat[-(idm + 1)]
            x = module(torch.cat([f, up_x], 1))
        x = self.output_conv(x)
        return self.final_act(x)
