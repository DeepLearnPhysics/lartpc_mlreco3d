import numpy as np
import torch
import torch.nn as nn

import MinkowskiEngine as ME
import MinkowskiFunctional as MF

from collections import defaultdict
from mlreco.models.layers.common.activation_normalization_factories import activations_dict, activations_construct, normalizations_construct
from mlreco.models.layers.common.configuration import setup_cnn_configuration
from mlreco.models.layers.common.blocks import ResNetBlock, ConvolutionBlock
from mlreco.models.layers.common.uresnet_layers import UResNetEncoder


class SparseResidualEncoder(UResNetEncoder):
    '''
    Minkowski Net Autoencoder for sparse tensor reconstruction.
    '''
    def __init__(self, cfg, name='res_encoder'):
        #print("RESENCODER = ", cfg)
        super(SparseResidualEncoder, self).__init__(cfg, name=name)

        self.model_config = cfg.get(name, {})
        self.latent_size = self.model_config.get('latent_size', 512)
        final_tensor_shape = self.spatial_size // (2**(self.depth-1))
        self.coordConv = self.model_config.get('coordConv', False)
        #print("Final Tensor Shape = ", final_tensor_shape)

        self.pool_mode = self.model_config.get('pool_mode', 'avg')

        # Initialize Input Layer
        if self.coordConv:
            self.input_layer = ME.MinkowskiConvolution(
                in_channels=self.num_input + self.D,
                out_channels=self.num_filters,
                kernel_size=self.input_kernel, stride=1, dimension=self.D)
        else:
            self.input_layer = ME.MinkowskiConvolution(
                in_channels=self.num_input,
                out_channels=self.num_filters,
                kernel_size=self.input_kernel, stride=1, dimension=self.D)

        if self.pool_mode == 'avg':
            self.pool = ME.MinkowskiGlobalPooling()
        elif self.pool_mode == 'conv':
            self.pool = nn.Sequential(
                ME.MinkowskiConvolution(
                    in_channels=self.nPlanes[-1],
                    out_channels=self.nPlanes[-1],
                    kernel_size=final_tensor_shape,
                    stride=final_tensor_shape,
                    dimension=self.D),
                ME.MinkowskiGlobalPooling())
        elif self.pool_mode == 'max':
            self.pool = nn.Sequential(
                ME.MinkowskiMaxPooling(final_tensor_shape, stride=final_tensor_shape, dimension=self.D),
                ME.MinkowskiGlobalPooling())
        else:
            raise NotImplementedError

        self.linear1 = ME.MinkowskiLinear(self.nPlanes[-1], self.latent_size)

    def forward(self, input_tensor):

        # print(input_tensor)
        features = input_tensor[:, -1].view(-1, 1)
        if self.coordConv:
            normalized_coords = (input_tensor[:, 1:4] - float(self.spatial_size) / 2) \
                    / (float(self.spatial_size) / 2)
            features = torch.cat([normalized_coords, features], dim=1)

        x = ME.SparseTensor(coordinates=input_tensor[:, :4].int(),
                            features=features)
        # Encoder
        encoderOutput = self.encoder(x)
        encoderTensors = encoderOutput['encoderTensors']
        finalTensor = encoderOutput['finalTensor']

        z = self.pool(finalTensor)
        latent = self.linear1(z)

        return latent.F


class SparseLightEncoder(nn.Module):

    def __init__(self, cfg, name='light_encoder'):
        super(SparseLightEncoder, self).__init__()

        setup_cnn_configuration(self, cfg, name)

        self.model_config = cfg.get(name, {})
        self.latent_size = self.model_config.get('latent_size', 512)
        final_tensor_shape = self.spatial_size // (2**(self.depth-1))
        self.coordConv = self.model_config.get('coordConv', False)
        #print("Final Tensor Shape = ", final_tensor_shape)

        self.pool_mode = self.model_config.get('pool_mode', 'avg')

        # UResNet Configurations
        self.reps = self.model_config.get('reps', 2)
        self.depth = self.model_config.get('depth', 5)
        self.num_filters = self.model_config.get('filters', 16)
        self.nPlanes = [i * self.num_filters for i in range(1, self.depth+1)]
        # self.kernel_size = cfg.get('kernel_size', 3)
        # self.downsample = cfg.get(downsample, 2)
        self.input_kernel = self.model_config.get('input_kernel', 3)

        # Initialize Input Layer
        # print(self.num_input)
        # print(self.input_kernel)
        self.input_layer = ME.MinkowskiConvolution(
            in_channels=self.num_input,
            out_channels=self.num_filters,
            kernel_size=self.input_kernel, stride=1, dimension=self.D,
            bias=self.allow_bias)

        # Initialize Encoder
        self.encoding_conv = []
        self.encoding_block = []
        for i, F in enumerate(self.nPlanes):
            m = []
            for _ in range(self.reps):
                m.append(ResNetBlock(F, F,
                    dimension=self.D,
                    activation=self.activation_name,
                    activation_args=self.activation_args,
                    normalization=self.norm,
                    normalization_args=self.norm_args,
                    bias=self.allow_bias))
            m = nn.Sequential(*m)
            self.encoding_block.append(m)
            m = []
            if i < self.depth-1:
                m.append(normalizations_construct(self.norm, F, **self.norm_args))
                m.append(activations_construct(
                    self.activation_name, **self.activation_args))
                m.append(ME.MinkowskiMaxPooling(kernel_size=2, 
                                                stride=2, 
                                                dimension=self.D))
            m = nn.Sequential(*m)
            self.encoding_conv.append(m)
        self.encoding_conv = nn.Sequential(*self.encoding_conv)
        self.encoding_block = nn.Sequential(*self.encoding_block)

        self.pool = nn.Sequential(
                ME.MinkowskiMaxPooling(final_tensor_shape, 
                                       stride=final_tensor_shape),
                ME.MinkowskiGlobalPooling())


    def encoder(self, x):
        '''
        Vanilla UResNet Encoder.

        INPUTS:
            - x (SparseTensor): MinkowskiEngine SparseTensor

        RETURNS:
            - result (dict): dictionary of encoder output with
            intermediate feature planes:
              1) encoderTensors (list): list of intermediate SparseTensors
              2) finalTensor (SparseTensor): feature tensor at
              deepest layer.
        # '''
        # print('input' , self.input_layer)
        # for name, param in self.input_layer.named_parameters():
        #     print(name, param.shape, param)
        x = self.input_layer(x)
        encoderTensors = [x]
        features_ppn = [x]
        for i, layer in enumerate(self.encoding_block):
            x = self.encoding_block[i](x)
            encoderTensors.append(x)
            x = self.encoding_conv[i](x)
            features_ppn.append(x)

        result = {
            "encoderTensors": encoderTensors,
            "features_ppn": features_ppn,
            "finalTensor": x
        }
        return result


    def forward(self, input):

        features = input[:, -1].view(-1, 1)
        if self.coordConv:
            normalized_coords = (input[:, 1:4] - float(self.spatial_size) / 2) \
                    / (float(self.spatial_size) / 2)
            features = torch.cat([normalized_coords, features], dim=1)

        x = ME.SparseTensor(coordinates=input[:, :4].int(),
                            features=features)
        encoderOutput = self.encoder(x)
        finalTensor = encoderOutput['finalTensor']

        z = self.pool(finalTensor)
        latent = self.linear1(z)

        return latent.F