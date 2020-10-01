# a basic sparse UResNet layer that expects to be fed data

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import sparseconvnet as scn

from mlreco.models.layers.base import SCNNetworkBase


class UResNet(SCNNetworkBase):
    '''
    Vanilla UResNet with access to intermediate layers in
    encoder/decoder path.

    Configurations
    -------------
    num_strides : int
        Depth of UResNet, also corresponds to how many times we down/upsample.
    filters : int
        Number of filters in the first convolution of UResNet.
        Will increase linearly with depth.
    spatial_size : int
        Size of the cube containing the data, e.g. 192, 512 or 768px.
    reps : int, optional
        Convolution block repetition factor
    kernel_size : int, optional
        Kernel size for the SC (sparse convolutions for down/upsample).
    input_kernel_size: int, optional
        Receptive field size for very first convolution after input layer.
    '''
    def __init__(self, cfg, name="uresnet"):
        super(UResNet, self).__init__(cfg)
        self.model_config = cfg[name]
        # UResNet Configurations
        self.reps = self.model_config.get('reps', 2)  # Conv block repetition factor
        self.kernel_size = self.model_config.get('kernel_size', 2)
        self.num_strides = self.model_config.get('num_strides', 5)
        # Unet number of features
        self.num_filters = self.model_config.get('filters', 16)
        # UNet number of features per level
        self.nPlanes = [i*self.num_filters for i in range(1, self.num_strides+1)]
        self.downsample = [self.kernel_size, 2]  # [filter size, filter stride]
        self.inputKernel = self.model_config.get('input_kernel_size', 3)

        # Input Layer Configurations and commonly used scn operations.
        self.input = scn.Sequential().add(
            scn.InputLayer(self.dimension, self.spatial_size, mode=3)).add(
            scn.SubmanifoldConvolution(self.dimension, self.nInputFeatures, \
            self.num_filters, self.inputKernel, self.allow_bias)) # Kernel size 3, no bias
        self.concat = scn.JoinTable()
        self.add = scn.AddTable()

        # Define Sparse UResNet Encoder
        self.encoding_block = scn.Sequential()
        self.encoding_conv = scn.Sequential()
        for i in range(self.num_strides):
            m = scn.Sequential()
            for _ in range(self.reps):
                self._resnet_block(m, self.nPlanes[i], self.nPlanes[i])
            self.encoding_block.add(m)
            m = scn.Sequential()
            if i < self.num_strides-1:
                m.add(
                    scn.BatchNormLeakyReLU(self.nPlanes[i], leakiness=self.leakiness)).add(
                    scn.Convolution(self.dimension, self.nPlanes[i], self.nPlanes[i+1], \
                        self.downsample[0], self.downsample[1], self.allow_bias))
            self.encoding_conv.add(m)

        # Define Sparse UResNet Decoder.
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
                self._resnet_block(m, self.nPlanes[i] * (2 if j == 0 else 1), self.nPlanes[i])
            self.decoding_block.add(m)


    def encoder(self, x):
        '''
        Vanilla UResNet Encoder

        INPUTS:
            - x (scn.SparseConvNetTensor): output from inputlayer (self.input)

        RETURNS:
            - features_encoder (list of SparseConvNetTensor): list of feature
            tensors in encoding path at each spatial resolution.
        '''
        # Embeddings at each layer
        features_enc = [x]
        # Loop over Encoding Blocks to make downsampled segmentation/clustering masks.
        for i, layer in enumerate(self.encoding_block):
            x = self.encoding_block[i](x)
            features_enc.append(x)
            x = self.encoding_conv[i](x)

        res = {
            "features_enc": features_enc,
            "deepest_layer": x
        }

        return res



    def decoder(self, features_enc, deepest_layer):
        '''
        Vanilla UResNet Decoder

        INPUTS:
            - features_enc (list of scn.SparseConvNetTensor): output of encoder.

        RETURNS:
            - features_dec (list of scn.SparseConvNetTensor): list of feature
            tensors in decoding path at each spatial resolution.
        '''
        features_dec = []
        x = deepest_layer
        for i, layer in enumerate(self.decoding_conv):
            encoder_feature = features_enc[-i-2]
            x = layer(x)
            x = self.concat([encoder_feature, x])
            x = self.decoding_block[i](x)
            features_dec.append(x)
        return features_dec


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

        res = {
            "features_enc": [features_enc],
            "features_dec": [features_dec]
        }

        return res


class UResNetEncoder(SCNNetworkBase):

    def __init__(self, cfg, name='uresnet_encoder'):
        super(UResNetEncoder, self).__init__(cfg, name='network_base')
        self.model_config = cfg[name]
        # UResNet Configurations
        self.reps = self.model_config.get('reps', 2)  # Conv block repetition factor
        self.kernel_size = self.model_config.get('kernel_size', 2)
        self.num_strides = self.model_config.get('num_strides', 5)
        self.num_filters = self.model_config.get('filters', 16)
        self.nPlanes = [i*self.num_filters for i in range(1, self.num_strides+1)]
        self.downsample = [self.kernel_size, 2]  # [filter size, filter stride]

        # Define Sparse UResNet Encoder
        self.encoding_block = scn.Sequential()
        self.encoding_conv = scn.Sequential()
        for i in range(self.num_strides):
            m = scn.Sequential()
            for _ in range(self.reps):
                self._resnet_block(m, self.nPlanes[i], self.nPlanes[i])
            self.encoding_block.add(m)
            m = scn.Sequential()
            if i < self.num_strides-1:
                m.add(
                    scn.BatchNormLeakyReLU(self.nPlanes[i], leakiness=self.leakiness)).add(
                    scn.Convolution(self.dimension, self.nPlanes[i], self.nPlanes[i+1], \
                        self.downsample[0], self.downsample[1], self.allow_bias))
            self.encoding_conv.add(m)


    def forward(self, x):
        '''
        Vanilla UResNet Encoder

        INPUTS:
            - x (scn.SparseConvNetTensor): output from inputlayer (self.input)

        RETURNS:
            - features_encoder (list of SparseConvNetTensor): list of feature
            tensors in encoding path at each spatial resolution.
        '''
        # Embeddings at each layer
        features_enc = [x]
        # Loop over Encoding Blocks to make downsampled segmentation/clustering masks.
        for i, layer in enumerate(self.encoding_block):
            x = self.encoding_block[i](x)
            features_enc.append(x)
            x = self.encoding_conv[i](x)

        res = {
            "features_enc": features_enc,
            "deepest_layer": x
        }

        return res


class UResNetDecoder(SCNNetworkBase):

    def __init__(self, cfg, name='uresnet_decoder'):
        super(UResNetDecoder, self).__init__(cfg, name='network_base')
        self.model_config = cfg[name]
        # print(name)
        # print(self.model_config)
        # print('\n')
        # UResNet Configurations
        self.model_config = cfg[name]
        self.reps = self.model_config.get('reps', 2)  # Conv block repetition factor
        self.kernel_size = self.model_config.get('kernel_size', 2)
        self.num_strides = self.model_config.get('num_strides', 5)
        self.num_filters = self.model_config.get('filters', 16)
        self.nPlanes = [i*self.num_filters for i in range(1, self.num_strides+1)]
        self.downsample = [self.kernel_size, 2]  # [filter size, filter stride]
        self.concat = scn.JoinTable()
        self.add = scn.AddTable()

        self.encoder_num_filters = self.model_config.get('encoder_num_filters', None)
        if self.encoder_num_filters is None:
            self.encoder_num_filters = self.num_filters
        self.encoder_nPlanes = [i*self.encoder_num_filters for i in range(1, self.num_strides+1)]

        # Define Sparse UResNet Decoder.
        self.decoding_block = scn.Sequential()
        self.decoding_conv = scn.Sequential()
        for idx, i in enumerate(list(range(self.num_strides-2, -1, -1))):
            if idx == 0:
                m = scn.Sequential().add(
                    scn.BatchNormLeakyReLU(self.encoder_nPlanes[i+1], leakiness=self.leakiness)).add(
                    scn.Deconvolution(self.dimension, self.encoder_nPlanes[i+1], self.nPlanes[i],
                        self.downsample[0], self.downsample[1], self.allow_bias))
            else:
                m = scn.Sequential().add(
                    scn.BatchNormLeakyReLU(self.nPlanes[i+1], leakiness=self.leakiness)).add(
                    scn.Deconvolution(self.dimension, self.nPlanes[i+1], self.nPlanes[i],
                        self.downsample[0], self.downsample[1], self.allow_bias))
            self.decoding_conv.add(m)
            m = scn.Sequential()
            for j in range(self.reps):
                self._resnet_block(m, self.nPlanes[i] + (self.encoder_nPlanes[i] \
                    if j == 0 else 0), self.nPlanes[i])
            self.decoding_block.add(m)


    def forward(self, features_enc, deepest_layer):
        '''
        Vanilla UResNet Decoder

        INPUTS:
            - features_enc (list of scn.SparseConvNetTensor): output of encoder.

        RETURNS:
            - features_dec (list of scn.SparseConvNetTensor): list of feature
            tensors in decoding path at each spatial resolution.
        '''
        features_dec = []
        x = deepest_layer
        for i, layer in enumerate(self.decoding_conv):
            encoder_feature = features_enc[-i-2]
            x = layer(x)
            x = self.concat([encoder_feature, x])
            x = self.decoding_block[i](x)
            features_dec.append(x)
        return features_dec
