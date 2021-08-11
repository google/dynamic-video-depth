
""" MIT License

Copyright (c) 2019 Intel ISL (Intel Intelligent Systems Lab)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. """


import torch
import torch.nn as nn
from .midas_blocks import FeatureFusionBlock, Interpolate, _make_encoder


class BaseModel(torch.nn.Module):
    def load(self, path):
        """Load model from file.
        Args:
            path (str): file path
        """
        parameters = torch.load(path)

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)


class MidasNet_mod(BaseModel):
    def __init__(self, path=None, features=256, non_negative=True, normalize_input=False, resize=None, freeze_backbone=False, mask_branch=False):
        """Init.
        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 256.
            backbone (str, optional): Backbone network for encoder. Defaults to resnet50
        """
        print("Loading weights: ", path)

        super(MidasNet_mod, self).__init__()

        use_pretrained = False if path is None else True

        self.pretrained, self.scratch = _make_encoder(features, use_pretrained)

        self.scratch.refinenet4 = FeatureFusionBlock(features)
        self.scratch.refinenet3 = FeatureFusionBlock(features)
        self.scratch.refinenet2 = FeatureFusionBlock(features)
        self.scratch.refinenet1 = FeatureFusionBlock(features)

        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
        )

        if path:
            self.load(path)

        if mask_branch:
            self.scratch.output_conv_mask = nn.Sequential(
                nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1),
                Interpolate(scale_factor=2, mode="bilinear"),
                nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid()
            )
        self.mask_branch = mask_branch

        if normalize_input:
            self.mean = torch.FloatTensor([0.485, 0.456, 0.406])
            self.std = torch.FloatTensor([0.229, 0.224, 0.225])
        self.normalize_input = normalize_input
        self.resize = resize
        self.freeze_backbone = freeze_backbone

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def defrost(self):
        if self.freeze_backbone:
            for p in self.scratch.parameters():
                p.requires_grad = True
        else:
            for p in self.parameters():
                p.requires_grad = True

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input data (image)
        Returns:
            tensor: depth
        """
        if self.normalize_input:
            self.mean = self.mean.to(x.device)
            self.std = self.std.to(x.device)
            x = x.permute([0, 2, 3, 1])
            x = (x - self.mean) / self.std
            x = x.permute([0, 3, 1, 2]).contiguous()

        orig_shape = x.shape[-2:]
        if self.resize is not None:
            x = torch.nn.functional.interpolate(x, size=self.resize, mode='bicubic', align_corners=True)

        if self.freeze_backbone:
            with torch.no_grad():
                layer_1 = self.pretrained.layer1(x)
                layer_2 = self.pretrained.layer2(layer_1)
                layer_3 = self.pretrained.layer3(layer_2)
                layer_4 = self.pretrained.layer4(layer_3)
        else:
            layer_1 = self.pretrained.layer1(x)
            layer_2 = self.pretrained.layer2(layer_1)
            layer_3 = self.pretrained.layer3(layer_2)
            layer_4 = self.pretrained.layer4(layer_3)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv(path_1)
        out = torch.clamp(out, min=1e-2)

        out = 10000 / (out)

        if self.mask_branch:
            mask = self.scratch.output_conv_mask(path_1)

        else:
            mask = torch.zeros_like(out)

        if self.resize is not None:
            out = torch.nn.functional.interpolate(out, size=orig_shape, mode='bicubic', align_corners=True)
            mask = torch.nn.functional.interpolate(mask, size=orig_shape, mode='bicubic', align_corners=True)
        return out, mask


class MidasNet(BaseModel):
    """Network for monocular depth estimation.
    """

    def __init__(self, path=None, features=256, non_negative=True, normalize_input=False, resize=None):
        """Init.
        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 256.
            backbone (str, optional): Backbone network for encoder. Defaults to resnet50
        """
        print("Loading weights: ", path)

        super(MidasNet, self).__init__()

        use_pretrained = False if path is None else True

        self.pretrained, self.scratch = _make_encoder(features, use_pretrained)

        self.scratch.refinenet4 = FeatureFusionBlock(features)
        self.scratch.refinenet3 = FeatureFusionBlock(features)
        self.scratch.refinenet2 = FeatureFusionBlock(features)
        self.scratch.refinenet1 = FeatureFusionBlock(features)

        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
        )

        if path:
            self.load(path)

        if normalize_input:
            self.mean = torch.FloatTensor([0.485, 0.456, 0.406])
            self.std = torch.FloatTensor([0.229, 0.224, 0.225])
        self.normalize_input = normalize_input
        self.resize = resize

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input data (image)
        Returns:
            tensor: depth
        """
        if self.normalize_input:
            self.mean = self.mean.to(x.device)
            self.std = self.std.to(x.device)
            x = x.permute([0, 2, 3, 1])
            x = (x - self.mean) / self.std
            x = x.permute([0, 3, 1, 2]).contiguous()

        orig_shape = x.shape[-2:]
        if self.resize is not None:
            x = torch.nn.functional.interpolate(x, size=self.resize, mode='bicubic', align_corners=True)

        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv(path_1)
        out = torch.clamp(out, min=1e-2)

        out = 10000 / (out)

        if self.resize is not None:
            out = torch.nn.functional.interpolate(out, size=orig_shape, mode='bicubic', align_corners=True)
        return out
