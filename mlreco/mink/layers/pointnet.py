import numpy as np
import torch
import torch.nn as nn

import MinkowskiEngine as ME
import MinkowskiFunctional as MF


class MinkowskiPointNet(ME.MinkowskiNetwork):
    def __init__(self, in_channel, out_channel, embedding_channel=1024, dimension=3):
        ME.MinkowskiNetwork.__init__(self, dimension)
        self.conv1 = nn.Sequential(
            ME.MinkowskiLinear(3, 64, bias=False),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(),
        )
        self.conv2 = nn.Sequential(
            ME.MinkowskiLinear(64, 64, bias=False),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(),
        )
        self.conv3 = nn.Sequential(
            ME.MinkowskiLinear(64, 64, bias=False),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(),
        )
        self.conv4 = nn.Sequential(
            ME.MinkowskiLinear(64, 128, bias=False),
            ME.MinkowskiBatchNorm(128),
            ME.MinkowskiReLU(),
        )
        self.conv5 = nn.Sequential(
            ME.MinkowskiLinear(128, embedding_channel, bias=False),
            ME.MinkowskiBatchNorm(embedding_channel),
            ME.MinkowskiReLU(),
        )
        self.max_pool = ME.MinkowskiGlobalMaxPooling()

        self.linear1 = nn.Sequential(
            ME.MinkowskiLinear(embedding_channel, 512, bias=False),
            ME.MinkowskiBatchNorm(512),
            ME.MinkowskiReLU(),
        )
        self.dp1 = ME.MinkowskiDropout()
        self.linear2 = ME.MinkowskiLinear(512, out_channel, bias=True)

    def forward(self, x: ME.TensorField):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.max_pool(x)
        x = self.linear1(x)
        x = self.dp1(x)
        return self.linear2(x).F