import torch
import torch.nn as nn
import torch.nn.functional as F


from typing import Callable

import torch.nn as nn


def create_initializer(mode: str) -> Callable:
    if mode in ['kaiming_fan_out', 'kaiming_fan_in']:
        mode = mode[8:]

        def initializer(module):
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight.data,
                                        mode=mode,
                                        nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight.data)
                nn.init.zeros_(module.bias.data)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight.data,
                                        mode=mode,
                                        nonlinearity='relu')
                nn.init.zeros_(module.bias.data)
    else:
        raise ValueError()

    return initializer


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,  # downsample with first conv
            padding=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module(
                'conv',
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,  # downsample
                    padding=0,
                    bias=False))
            self.shortcut.add_module('bn', nn.BatchNorm2d(out_channels))  # BN

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))
        y += self.shortcut(x)
        y = F.relu(y, inplace=True)  # apply ReLU after addition
        return y


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        bottleneck_channels = out_channels // self.expansion

        self.conv1 = nn.Conv2d(in_channels,
                               bottleneck_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)

        self.conv2 = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride,  # downsample with 3x3 conv
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)

        self.conv3 = nn.Conv2d(bottleneck_channels,
                               out_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()  # identity
        if in_channels != out_channels:
            self.shortcut.add_module(
                'conv',
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,  # downsample
                    padding=0,
                    bias=False))
            self.shortcut.add_module('bn', nn.BatchNorm2d(out_channels))  # BN

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = F.relu(self.bn2(self.conv2(y)), inplace=True)
        y = self.bn3(self.conv3(y))  # not apply ReLU
        y += self.shortcut(x)
        y = F.relu(y, inplace=True)  # apply ReLU after addition
        return y


class Network(nn.Module):
    def __init__(self):
        super().__init__()

        # model_config = config.model.resnet
        initial_channels = 64
        block_type = 'basic'
        n_blocks = [2, 2, 2, 2]

        assert block_type in ['basic', 'bottleneck']
        if block_type == 'basic':
            block = BasicBlock
        else:
            block = BottleneckBlock

        n_channels = [
            initial_channels,
            initial_channels * 2 * block.expansion,
            initial_channels * 4 * block.expansion,
            initial_channels * 8 * block.expansion,
        ]

        # # self.conv = nn.Conv2d(config.dataset.n_channels,
        #                       n_channels[0],
        #                       kernel_size=7,
        #                       stride=2,
        #                       padding=3,
        #                       bias=False)
        self.conv = nn.Conv2d(3,
                              n_channels[0],
                              kernel_size=7,
                              stride=2,
                              padding=3,
                              bias=False)
        self.bn = nn.BatchNorm2d(initial_channels)

        self.stage1 = self._make_stage(n_channels[0],
                                       n_channels[0],
                                       n_blocks[0],
                                       block,
                                       stride=1)
        self.stage2 = self._make_stage(n_channels[0],
                                       n_channels[1],
                                       n_blocks[1],
                                       block,
                                       stride=2)
        self.stage3 = self._make_stage(n_channels[1],
                                       n_channels[2],
                                       n_blocks[2],
                                       block,
                                       stride=2)
        self.stage4 = self._make_stage(n_channels[2],
                                       n_channels[3],
                                       n_blocks[3],
                                       block,
                                       stride=2)

        # compute conv feature size
        with torch.no_grad():
            dummy_data = torch.zeros(
                (1, 3, 224,
                 224),
                dtype=torch.float32)
            self.feature_size = self._forward_conv(dummy_data).view(
                -1).shape[0]

        self.fc = nn.Linear(self.feature_size, 5)

        # initialize weights
        initializer = create_initializer('kaiming_fan_out')
        self.apply(initializer)

    def _make_stage(self, in_channels, out_channels, n_blocks, block, stride):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_name = f'block{index + 1}'
            if index == 0:
                stage.add_module(
                    block_name, block(in_channels, out_channels,
                                      stride=stride))
            else:
                stage.add_module(block_name,
                                 block(out_channels, out_channels, stride=1))
        return stage

    def _forward_conv(self, x):
        x = F.relu(self.bn(self.conv(x)), inplace=True)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
