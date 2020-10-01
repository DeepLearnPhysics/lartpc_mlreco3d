import torch
import torch.nn as nn
import numpy as np
import sparseconvnet as scn

from mlreco.models.layers.base import SCNNetworkBase
from mlreco.models.layers.uresnet import UResNet
from mlreco.models.layers.fpn import FPN
from mlreco.models.clustercnn.clusternet import ClusterUNet


class CascadeNet(SCNNetworkBase):
    '''
    Simple CascadeNet architecture with interchangable backbone
    architectures.
    '''
    def __init__(self, cfg, name='cascadenet'):
        super(CascadeNet, self).__init__(cfg)

        # StackNet Model Parameters
        self.model_config = cfg['modules'][name]
        self.reduce_feature = self.model_config.get('reduce_feature', 'resnet')
        self.sum_features = sum(self.nPlanes)
        self.stackPlanes = [i * int(self.sum_features / self.num_strides) \
            for i in range(self.num_strides, 0, -1)]
        self.backbone = self.model_config.get('backbone', 'clusterunet')

        # Backbone Network
        if self.backbone == 'uresnet':
            self.net = UResNet(cfg)
            self.upsampling_name = 'feature_dec'
        elif self.backbone == 'fpn':
            self.net = FPN(cfg)
            self.upsampling_name = 'feature_dec'
        elif self.backbone == 'clusterunet':
            self.net = ClusterUNet(cfg)
            self.upsampling_name = 'cluster_feature'
        else:
            raise ValueError('Invalid option for StackNet backbone architecture.')

        # Upsampling Layers
        self.unpooling = scn.Sequential()
        for i in range(self.num_strides-1):
            module_unpool = scn.Sequential()
            for _ in range(self.num_strides-1-i):
                module_unpool.add(
                    scn.UnPooling(self.dimension, downsample[0], downsample[1]))
            self.unpooling.add(module_unpool)

        self.reduction_layers = scn.Sequential()
        if self.reduce_feature == 'resnet':
            reduceBlock = self._resnet_block
        elif self.reduce_feature == 'conv':
            reduceBlock = self._block
        elif self.reduce_feature = 'nin':
            reduceBlock = self._nin_block
        else:
            raise ValueError('Invalid option for StackNet feature reducing layers.')

        # Feature Reducing Layers
        self.cluster_decoder = scn.Sequential()
        for i in range(num_strides-1):
            m = scn.Sequential()
            reduceBlock(m, stackPlanes[i], stackPlanes[i+1])
            self.cluster_decoder.add(m)

        self.embedding = scn.Sequential().add(
            scn.BatchNormLeakyReLU(
                nPlanes_decoder[-1], leakiness=self.leakiness)).add(
            scn.NetworkInNetwork(
                nPlanes_decoder[-1], self.embedding_dim, self.allow_bias)).add(
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
        res = self.net(input)
        cluster_feature = res[self.upsampling_name]
        stack_feature = []

        for i, layer in enumerate(cluster_feature[0][::-1]):
            if i < self._num_strides-1:
                f = self.unpooling[i](layer)
                stack_feature.append(f)
            else:
                stack_feature.append(layer)
        
        out = self.concat(stack_feature)
        out = self.cluster_decoder(out)
        out = self.embedding(out)

        res = {
            'segmentation': cnet_output['segmentation'],
            'cluster_features': cluster_feature
            'final_layer': [out]
            }

        return res