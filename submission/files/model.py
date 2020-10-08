# MIT License

# Original work Copyright (c) 2018 Joris
# Modified work Copyright (C) 2020 Canon Medical Systems Corporation.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from math import sqrt
import random

import torch
import torch.nn.functional as F
from torch import nn



class UNet(nn.Module):
    def __init__(
            self,
            in_channels=1,
            n_classes=2,
            depth=5,
            wf=6,
            padding=False,
            batch_norm=False,
            up_mode='upconv',
            grid=False,
            bias=True
    ):

        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        self.grid = grid
        self.bias = bias
        prev_channels = in_channels + 3 if grid else in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm, bias=self.bias)
            )
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm, bias=self.bias)
            )
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1, bias=self.bias)

    def forward_down(self, x):
        if self.grid:
            with torch.no_grad():
                grid_x, grid_y = torch.meshgrid(torch.arange(x.shape[2]), torch.arange(x.shape[3]))
                grid_x = grid_x.view(1, 1, x.shape[-2], x.shape[-1]).expand_as(x).to(x.device).float() / x.shape[
                    -2] - 0.5
                grid_y = grid_y.view(1, 1, x.shape[-2], x.shape[-1]).expand_as(x).to(x.device).float() / x.shape[
                    -1] - 0.5
                dist = (grid_x ** 2 + grid_y ** 2) ** 0.5

            x = torch.cat([grid_x, grid_y, x, dist], dim=1)

        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)

        return x, blocks

    def forward_up_without_last(self, x, blocks):
        for i, up in enumerate(self.up_path):
            skip = blocks[-i - 1]
            x = up(x, skip)

        return x

    def forward_without_last(self, x):
        blocks = []
        x, blocks = self.forward_down(x)

        x = self.forward_up_without_last(x, blocks)

        return x

    def forward(self, x):

        x = self.get_features(x)
        return self.last(x)

    def get_features(self, x):
        return self.forward_without_last(x)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm, bias=True):
        super(UNetConvBlock, self).__init__()
        block = []
        if padding:
            block.append(nn.ReflectionPad2d(1))

        block.append(WNConv2d(in_size, out_size, kernel_size=3, bias=bias))
        block.append(CustomSwish())

        if batch_norm:
            block.append(nn.BatchNorm2d(out_size, affine=bias))
        else:
            pass
            block.append(nn.GroupNorm(get_groups(out_size), out_size, affine=bias))

        if padding:
            block.append(nn.ReflectionPad2d(1))

        block.append(WNConv2d(out_size, out_size, kernel_size=3, bias=bias))
        block.append(CustomSwish())

        if batch_norm:
            block.append(nn.BatchNorm2d(out_size, affine=bias))
        else:
            block.append(nn.GroupNorm(get_groups(out_size), out_size, affine=bias))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm, bias=True):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=bias)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=bias),
            )

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm, bias=bias)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
               :, :, diff_y: (diff_y + target_size[0]), diff_x: (diff_x + target_size[1])
               ]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out


# from https://github.com/joe-siyuan-qiao/WeightStandardization
class WNConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(WNConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


# From https://medium.com/the-artificial-impostor/more-memory-efficient-swish-activation-function-e07c22c12a76
class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class CustomSwish(nn.Module):
    def forward(self, input_tensor):
        return Swish.apply(input_tensor)


def get_groups(channels: int) -> int:
    """
    :param channels:
    :return: return the appropriate (I think) number of normalisation groups for group norm given channel number.
    """
    divisors = []
    for i in range(1, int(sqrt(channels)) + 1):
        if channels % i == 0:
            divisors.append(i)
            other = channels // i
            if i != other:
                divisors.append(other)
    return sorted(divisors)[len(divisors) // 2]



def noise(x):
    std = 0.05
    ns = torch.normal(mean=torch.zeros(x.shape[0], x.shape[1], 16, 16), std=std).to(x.device)
    ns = F.upsample_bilinear(ns, scale_factor=8)

    res = x + ns

    return res

