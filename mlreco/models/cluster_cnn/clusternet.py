import torch
import torch.nn as nn
import numpy as np
import sparseconvnet as scn
from collections import defaultdict

from mlreco.models.layers.uresnet import UResNet
from .utils import add_normalized_coordinates, pairwise_distances


class ClusterUNet(UResNet):

    def __init__(self, cfg, name='clusterunet'):
        '''
        Multi-Scale Clustering Network for Particle Instance Segmentation.

        ClusterUNet adds N convolution layers to each of the decoding path
        feature tensor of UResNet and outputs an embedding for each spatial
        resolution.

        This module serves as a base architecture for clustering on
        multiple spatial resolutions.

        Configuration
        -------------
        N: int, optional
            Performs N convolution operation at each layer
            to transform from segmentation features to clustering features
        coordConv: bool, optional
            If True, network concatenates normalized coordinates (between -1 and 1) to
            the feature tensor at every spatial resolution in the decoding path.
        simpleN: bool, optional
            Uses ResNet blocks by default. If True, N-convolution blocks
            are replaced with simple BN + SubMfdConv layers.
        embedding_dim: int, optional
            Dimension of clustering hyperspace.
        '''
        super(ClusterUNet, self).__init__(cfg, name=name)
        self.model_config = cfg[name]
        self.num_classes = self.model_config.get('num_classes', 5)
        self.N = self.model_config.get('N', 1)
        self.simpleN = self.model_config.get('simple_conv', False)
        self.embedding_dim = self.model_config.get('embedding_dim', 8)
        self.coordConv = self.model_config.get('coordConv', False)

        if self.simpleN:
            clusterBlock = self._block
        else:
            clusterBlock = self._resnet_block

        # Intermediate transformations and clustering upsamplings
        self.cluster_conv = scn.Sequential()
        self.cluster_transform = scn.Sequential()
        clusterPlanes = self.nPlanes[::-1]
        for i, fDim in enumerate(clusterPlanes):
            if i < len(clusterPlanes)-1:
                m = scn.Sequential().add(
                    scn.BatchNormLeakyReLU(fDim, leakiness=self.leakiness)).add(
                    scn.Deconvolution(self.dimension, fDim, clusterPlanes[i+1],
                        self.downsample[0], self.downsample[1], self.allow_bias))
                self.cluster_conv.add(m)
            m = scn.Sequential()
            for j in range(self.N):
                num_input = fDim
                if i > 0 and j == 0:
                    num_input *= 2
                if self.coordConv and j == 0:
                    num_input += self.dimension
                clusterBlock(m, num_input, fDim)
            self.cluster_transform.add(m)

        # NetworkInNetwork layer for final embedding space.
        self.embedding = scn.NetworkInNetwork(
            self.num_filters, self.embedding_dim, self.allow_bias)


    def decoder(self, features_enc, deepest_layer):
        '''
        ClusterUNet Decoder

        INPUTS:
            - features_enc (list of scn.SparseConvNetTensor): output of encoder.

        RETURNS:
            - features_dec (list of scn.SparseConvNetTensor):
            list of feature tensors in decoding path at each spatial resolution.
            - features_cluster (list of scn.SparseConvNetTensor):
            list of transformed features on which we apply clustering loss
            at every spatial resolution.
        '''
        features_dec = [deepest_layer]
        cluster_feature = []
        x_seg = deepest_layer
        x_emb = deepest_layer
        for i, layer in enumerate(self.decoding_conv):
            encoder_feature = features_enc[-i-2]
            if self.coordConv:
                x_emb = add_normalized_coordinates(x_emb)
            x_emb = self.cluster_transform[i](x_emb)
            cluster_feature.append(x_emb)
            x_emb = self.cluster_conv[i](x_emb)
            x_seg = layer(x_seg)
            x_seg = self.concat([encoder_feature, x_seg])
            x_seg = self.decoding_block[i](x_seg)
            features_dec.append(x_seg)
            x_emb = self.concat([x_emb, x_seg])
        # Compensate for last clustering convolution
        if self.coordConv:
            x_emb = add_normalized_coordinates(x_emb)
        x_emb = self.cluster_transform[-1](x_emb)
        cluster_feature.append(x_emb)
        x_emb = self.embedding(x_emb)

        result = {
            "features_dec": features_dec,
            "cluster_feature": cluster_feature,
            'final_embedding': x_emb
        }

        return result


    def forward(self, input):
        '''
        point_cloud is a list of length minibatch size (assumes mbs = 1)
        point_cloud[0] has 3 spatial coordinates + 1 batch coordinate + 1 feature
        label has shape (point_cloud.shape[0] + 5*num_labels, 1)
        label contains segmentation labels for each point + coords of gt points

        RETURNS:
            - feature_dec: decoder features at each spatial resolution.
            - cluster_feature: clustering features at each spatial resolution.
        '''
        point_cloud, = input
        coords = point_cloud[:, 0:self.dimension+1].float()
        features = point_cloud[:, self.dimension+1:].float()
        res = {}

        x = self.input((coords, features))
        encoder_output = self.encoder(x)
        decoder_output = self.decoder(encoder_output['features_enc'],
                                encoder_output['deepest_layer'])

        res['features_dec'] = [decoder_output['features_dec']]
        # Reverse cluster feature tensor list to agree with label ordering.
        res['cluster_feature'] = [decoder_output['cluster_feature'][::-1]]
        res['final_embedding'] = [decoder_output['final_embedding']]

        return res


class StackedClusterUNet(ClusterUNet):
    '''
    Cluster-UResNet with ally/enemy estimation map enhancement.
    Both ClusterNet and StackNet architectures are mixed.
    '''
    def __init__(self, cfg, name='clusterunet_stack'):
        super(StackedClusterUNet, self).__init__(cfg, name=name)
        self.model_config = cfg['modules'][name]

        # StackNet Model Parameters
        self.model_config = cfg['modules'][name]
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
            self.stackPlanes = [self.sum_features, 100, 20, 2]

        self.reduction_layers = scn.Sequential()
        if self.reduce_feature == 'resnet':
            reduceBlock = self._resnet_block
        elif self.reduce_feature == 'conv':
            reduceBlock = self._block
        elif self.reduce_feature == 'nin':
            reduceBlock = self._nin_block
        else:
            raise ValueError('Invalid option for StackNet feature reducing layers.')

        # NINs from segmentation feature planes + clustering feature planes -> distance est.
        self.distance_features = scn.Sequential()
        for fDim in self.nPlanes[::-1]:
            m = scn.Sequential()
            m.add(scn.BatchNormLeakyReLU(fDim * 2, leakiness=self.leakiness)).add(
                scn.NetworkInNetwork(fDim * 2, fDim, self.allow_bias))
            self.distance_features.add(m)

        # NIN Feature Reducing Layers to Distance Estimation
        self.dist_reduce = scn.Sequential()
        for i in range(len(self.stackPlanes)-1):
            m = scn.Sequential()
            reduceBlock(m, self.stackPlanes[i], self.stackPlanes[i+1])
            self.dist_reduce.add(m)

        self.concat = scn.JoinTable()

    def forward(self, input):
        '''
        point_cloud is a list of length minibatch size (assumes mbs = 1)
        point_cloud[0] has 3 spatial coordinates + 1 batch coordinate + 1 feature
        label has shape (point_cloud.shape[0] + 5*num_labels, 1)
        label contains segmentation labels for each point + coords of gt points

        RETURNS:
            - feature_dec: decoder features at each spatial resolution.
            - cluster_feature: clustering features at each spatial resolution.
            - distance_estimation: distance estimation map at highest resolution.
        '''
        point_cloud, = input
        coords = point_cloud[:, 0:self.dimension+1].float()
        features = point_cloud[:, self.dimension+1:].float()
        res = {}

        x = self.input((coords, features))
        encoder_output = self.encoder(x)
        decoder_output = self.decoder(encoder_output['features_enc'],
                                encoder_output['deepest_layer'])

        features_dec = decoder_output['features_dec']
        cluster_feature = decoder_output['cluster_feature']
        stack_feature = []

        for i, layer in enumerate(features_dec):
            if i < self.num_strides-1:
                f = self.concat([layer, cluster_feature[i]])
                f = self.distance_features[i](f)
                f = self.unpooling[i](f)
                stack_feature.append(f)
            else:
                stack_feature.append(layer)

        stack_feature = self.concat(stack_feature)
        distance_estimation = self.dist_reduce(stack_feature)
        res['features_dec'] = [features_dec]
        # Reverse cluster feature tensor list to agree with label ordering.
        res['cluster_feature'] = [cluster_feature[::-1]]
        res['distance_estimation'] = [distance_estimation]
        res['final_embedding'] = [decoder_output['final_embedding']]

        return res
