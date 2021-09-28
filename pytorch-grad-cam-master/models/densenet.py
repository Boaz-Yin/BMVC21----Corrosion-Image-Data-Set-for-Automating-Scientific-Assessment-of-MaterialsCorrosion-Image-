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
    def __init__(self, in_channels, out_channels, drop_rate):
        super().__init__()

        self.drop_rate = drop_rate

        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias=False)

    def forward(self, x):
        y = self.conv(F.relu(self.bn(x), inplace=True))
        if self.drop_rate > 0:
            y = F.dropout(y,
                          p=self.drop_rate,
                          training=self.training,
                          inplace=False)
        return torch.cat([x, y], dim=1)


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate):
        super().__init__()

        self.drop_rate = drop_rate

        bottleneck_channels = out_channels * 4

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels,
                               bottleneck_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)

        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)

    def forward(self, x):
        y = self.conv1(F.relu(self.bn1(x), inplace=True))
        if self.drop_rate > 0:
            y = F.dropout(y,
                          p=self.drop_rate,
                          training=self.training,
                          inplace=False)
        y = self.conv2(F.relu(self.bn2(y), inplace=True))
        if self.drop_rate > 0:
            y = F.dropout(y,
                          p=self.drop_rate,
                          training=self.training,
                          inplace=False)
        return torch.cat([x, y], dim=1)


class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate):
        super().__init__()

        self.drop_rate = drop_rate

        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=1,
                              stride=1,
                              padding=0,
                              bias=False)

    def forward(self, x):
        x = self.conv(F.relu(self.bn(x), inplace=True))
        if self.drop_rate > 0:
            x = F.dropout(x,
                          p=self.drop_rate,
                          training=self.training,
                          inplace=False)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class Network(nn.Module):
    def __init__(self):
        super().__init__()

        # model_config = config.model.densenet
        block_type = 'bottleneck'
        n_blocks = [6, 12, 24, 16]
        self.growth_rate = 32
        self.drop_rate = 0.0
        self.compression_rate = 0.5

        assert block_type in ['basic', 'bottleneck']
        if block_type == 'basic':
            block = BasicBlock
        else:
            block = BottleneckBlock

        in_channels = [2 * self.growth_rate]
        for index in range(4):
            denseblock_out_channels = int(in_channels[-1] +
                                          n_blocks[index] * self.growth_rate)
            if index < 3:
                transitionblock_out_channels = int(denseblock_out_channels *
                                                   self.compression_rate)
            else:
                transitionblock_out_channels = denseblock_out_channels
            in_channels.append(transitionblock_out_channels)

        self.conv = nn.Conv2d(3,
                              in_channels[0],
                              kernel_size=7,
                              stride=2,
                              padding=3,
                              bias=False)
        self.bn = nn.BatchNorm2d(in_channels[0])
        self.stage1 = self._make_stage(in_channels[0], n_blocks[0], block,
                                       True)
        self.stage2 = self._make_stage(in_channels[1], n_blocks[1], block,
                                       True)
        self.stage3 = self._make_stage(in_channels[2], n_blocks[2], block,
                                       True)
        self.stage4 = self._make_stage(in_channels[3], n_blocks[3], block,
                                       False)
        self.bn_last = nn.BatchNorm2d(in_channels[4])

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

    def _make_stage(self, in_channels, n_blocks, block, add_transition_block):
        stage = nn.Sequential()
        for index in range(n_blocks):
            stage.add_module(
                f'block{index + 1}',
                block(in_channels + index * self.growth_rate, self.growth_rate,
                      self.drop_rate))
        if add_transition_block:
            in_channels = int(in_channels + n_blocks * self.growth_rate)
            out_channels = int(in_channels * self.compression_rate)
            stage.add_module(
                'transition',
                TransitionBlock(in_channels, out_channels, self.drop_rate))
        return stage

    def _forward_conv(self, x):
        x = F.relu(self.bn(self.conv(x)), inplace=True)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.bn_last(x)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
