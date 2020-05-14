import torch
import sys
import torch.nn as nn
import numpy as np
import sparseconvnet as scn
from collections import defaultdict

from mlreco.models.layers.uresnet import UResNet


class FPN(UResNet):
    """
    Feature Pyramid Network

    Original Paper: https://arxiv.org/abs/1612.03144

    FPN shares most layers with our UResNet architecture except for
    the lateral skip connections that involve 1x1 convolutions with
    an additive block, replacing U-Net's concat-convolution blocks.
    FPN is a common backbone network architecture for several SOTA
    semantic/instance segmenation architectures.

    Configuration
    -------------
    num_strides : int
        Depth of UResNet, also corresponds to how many times we down/upsample.
    filters : int
        Number of filters in the first convolution of UResNet.
        Will increase linearly with depth.
    reps : int, optional
        Convolution block repetition factor
    kernel_size : int, optional
        Kernel size for the SC (sparse convolutions for down/upsample).
    features: int, optional
        How many features are given to the network initially
    """
    def __init__(self, cfg, name='uresnet'):
        super(FPN, self).__init__(cfg, name=name)
        self.model_config = cfg[name]
        # Now decoding block does not reduce features from 2f -> f.
        self.decoding_block = scn.Sequential()
        self.decoding_conv = scn.Sequential()
        for i in range(self.num_strides-2, -1, -1):
            m = scn.Sequential().add(
                scn.BatchNormLeakyReLU(self.nPlanes[i+1], leakiness=self.leakiness)).add(
                scn.Deconvolution(self.dimension, self.nPlanes[i+1], self.nPlanes[i],
                    self.downsample[0], self.downsample[1], self.allow_bias))
            self.decoding_conv.add(m)
            m = scn.Sequential()
            for j in range(self.reps):
                self._resnet_block(m, self.nPlanes[i], self.nPlanes[i])
            self.decoding_block.add(m)

        # Skip Connecting NIN Layers unique to FPN.
        self.skip_connections = scn.Sequential()
        for i in range(self.num_strides-2, -1, -1):
            self.skip_connections.add(
                scn.NetworkInNetwork(self.nPlanes[i], self.nPlanes[i], self.allow_bias)
            )
        # We keep the encoder/forward method of UResNet as they are identical.

    def decoder(self, features_enc, deepest_layer):
        '''
        Vanilla FPN Decoder

        INPUTS:
            - features_enc (list of scn.SparseConvNetTensor): output of encoder.

        RETURNS:
            - features_dec (list of scn.SparseConvNetTensor): list of feature
            tensors in decoding path at each spatial resolution.
        '''
        x = deepest_layer
        features_dec = [x]
        for i, layer in enumerate(self.decoding_conv):
            x = layer(x)
            encoder_feature = features_enc[-i-2]
            skip = self.skip_connections[i](encoder_feature)
            x = self.add([skip, x])
            x = self.decoding_block[i](x)
            features_dec.append(x)
        return features_dec
