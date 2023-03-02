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
                            features=features.float())
        # Encoder
        encoderOutput = self.encoder(x)
        encoderTensors = encoderOutput['encoderTensors']
        finalTensor = encoderOutput['finalTensor']

        z = self.pool(finalTensor)
        latent = self.linear1(z)

        return latent.F
