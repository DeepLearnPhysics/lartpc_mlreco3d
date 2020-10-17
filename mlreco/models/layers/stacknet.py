import torch
import torch.nn as nn
import numpy as np
import sparseconvnet as scn

from mlreco.models.layers.base import SCNNetworkBase
from mlreco.models.layers.uresnet import UResNet
from mlreco.models.layers.fpn import FPN


class StackUNet(UResNet):
    '''
    Simple StackNet architecture with UResNet backbone without
    intermediate layer losses.
    '''
    def __init__(self, cfg, name='stacknet'):
        super(StackUNet, self).__init__(cfg)

        # StackNet Model Parameters
        self.model_config = cfg[name]
        self.reduce_feature = self.model_config.get('reduce_feature', 'nin')
        self.sum_features = sum(self.nPlanes)
        self.embedding_dim = self.model_config.get('embedding_dim', 8)

        # UnPooling Layers
        self.learnable_upsampling = self.model_config.get('learnable_upsampling', False)
        self.unpooling = scn.Sequential()

        if self.learnable_upsampling:
            # Using transpose convolution to upsample hierarchy of feature maps.
            for i in range(self.num_strides-1):
                m = scn.Sequential()
                for j in range(self.num_strides-2-i, -1, -1):
                    m.add(
                        scn.BatchNormLeakyReLU(self.nPlanes[j+1], leakiness=self.leakiness)).add(
                        scn.Deconvolution(self.dimension, self.nPlanes[j+1], self.nPlanes[j],
                            self.downsample[0], self.downsample[1], self.allow_bias))
                    self._resnet_block(m, self.nPlanes[j], self.nPlanes[j])
                self.unpooling.add(m)
            self.stackPlanes = self.nPlanes[::-1]
        else:
            # Using normal unpooling layers to upsample feature maps.
            for i in range(self.num_strides-1):
                m = scn.Sequential()
                for _ in range(self.num_strides-1-i):
                    m.add(
                        scn.UnPooling(self.dimension, self.downsample[0], self.downsample[1]))
                self.unpooling.add(m)
            self.stackPlanes = [i * int(self.sum_features / self.num_strides) \
                for i in range(self.num_strides, 0, -1)]

        self.reduction_layers = scn.Sequential()
        if self.reduce_feature == 'resnet':
            reduceBlock = self._resnet_block
        elif self.reduce_feature == 'conv':
            reduceBlock = self._block
        elif self.reduce_feature == 'nin':
            reduceBlock = self._nin_block
        else:
            raise ValueError('Invalid option for StackNet feature reducing layers.')

        # Feature Reducing Layers
        self.cluster_decoder = scn.Sequential()
        self.stackPlanes.append(self.num_filters)
        for i in range(self.num_strides):
            m = scn.Sequential()
            reduceBlock(m, self.stackPlanes[i], self.stackPlanes[i+1])
            self.cluster_decoder.add(m)

        # 1x1 Convolutions to Final Embeddings
        self.embedding = scn.Sequential().add(
            scn.BatchNormLeakyReLU(
                self.num_filters, leakiness=self.leakiness)).add(
            scn.NetworkInNetwork(
                self.num_filters, self.embedding_dim, self.allow_bias)).add(
            scn.OutputLayer(self.dimension))

        self.concat = scn.JoinTable()


    def forward(self, input):
        '''
        point_cloud is a list of length minibatch size (assumes mbs = 1)
        point_cloud[0] has 3 spatial coordinates + 1 batch coordinate + 1 feature
        label has shape (point_cloud.shape[0] + 5*num_labels, 1)
        label contains segmentation labels for each point + coords of gt points

        RETURNS:
            - feature_enc: encoder features at each spatial resolution.
            - feature_dec: decoder features at each spatial resolution.
        '''
        point_cloud, = input
        coords = point_cloud[:, 0:self.dimension+1].float()
        features = point_cloud[:, self.dimension+1:].float()

        x = self.input((coords, features))
        encoder_res = self.encoder(x)
        features_enc = encoder_res['features_enc']
        deepest_layer = encoder_res['deepest_layer']
        features_dec = self.decoder(features_enc, deepest_layer)
        features_dec = [deepest_layer] + features_dec
        stack_feature = []

        for i, layer in enumerate(features_dec):
            if i < self.num_strides-1:
                f = self.unpooling[i](layer)
                stack_feature.append(f)
            else:
                stack_feature.append(layer)

        stack_feature = self.concat(stack_feature)
        out = self.cluster_decoder(stack_feature)
        out = self.embedding(out)

        res = {
            'features_dec': features_dec,
            'cluster_feature': [out],
            'stacked_features': [stack_feature]
            }

        return res
