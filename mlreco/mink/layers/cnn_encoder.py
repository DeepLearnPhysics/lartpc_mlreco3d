import numpy as np
import torch
import torch.nn as nn

import MinkowskiEngine as ME
import MinkowskiFunctional as MF

from collections import defaultdict
from mlreco.mink.layers.factories import activations_dict, activations_construct, normalizations_construct
from mlreco.mink.layers.network_base import MENetworkBase
from mlreco.mink.layers.blocks import ResNetBlock, ConvolutionBlock


class SparseEncoder(MENetworkBase):
    '''
    Minkowski Net Autoencoder for sparse tensor reconstruction.
    '''
    def __init__(self, cfg, name='sparse_encoder'):
        super(SparseEncoder, self).__init__(cfg)
        self.model_config = cfg[name]
        print(name, self.model_config)
        self.reps = self.model_config.get('reps', 2)
        self.depth = self.model_config.get('depth', 7)
        self.num_filters = self.model_config.get('num_filters', 16)
        self.nPlanes = [i * self.num_filters for i in range(1, self.depth+1)]
        self.input_kernel = self.model_config.get('input_kernel', 7)
        self.latent_size = self.model_config.get('latent_size', 512)
        final_tensor_shape = self.spatial_size // (2**(self.depth-1))
        self.coordConv = self.model_config.get('coordConv', False)
        print("Final Tensor Shape = ", final_tensor_shape)

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

        # Initialize Encoder
        self.encoding_conv = []
        self.encoding_block = []
        for i, F in enumerate(self.nPlanes):
            m = []
            for _ in range(self.reps):
                m.append(ConvolutionBlock(F, F,
                    dimension=self.D,
                    activation=self.activation_name,
                    activation_args=self.activation_args))
            m = nn.Sequential(*m)
            self.encoding_block.append(m)
            m = []
            if i < self.depth-1:
                m.append(ME.MinkowskiBatchNorm(F))
                m.append(activations_construct(
                    self.activation_name, **self.activation_args))
                m.append(ME.MinkowskiConvolution(
                    in_channels=self.nPlanes[i],
                    out_channels=self.nPlanes[i+1],
                    kernel_size=2, stride=2, dimension=self.D))
            m = nn.Sequential(*m)
            self.encoding_conv.append(m)
        self.encoding_conv = nn.Sequential(*self.encoding_conv)
        self.encoding_block = nn.Sequential(*self.encoding_block)

        self.global_pool = ME.MinkowskiConvolution(
            in_channels=self.nPlanes[-1], 
            out_channels=self.nPlanes[-1],
            kernel_size=final_tensor_shape, 
            stride=final_tensor_shape, 
            dimension=self.D)

        self.max_pool = ME.MinkowskiGlobalPooling()

        self.linear1 = nn.Sequential(
            ME.MinkowskiBatchNorm(self.nPlanes[-1]),
            activations_construct(
                self.activation_name, **self.activation_args),
            ME.MinkowskiLinear(
                in_channels=self.nPlanes[-1], out_channels = self.latent_size)
        )

        self.union = ME.MinkowskiUnion()


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
        '''
        x = self.input_layer(x)
        encoderTensors = [x]
        for i, layer in enumerate(self.encoding_block):
            x = self.encoding_block[i](x)
            encoderTensors.append(x)
            x = self.encoding_conv[i](x)

        result = {
            "encoderTensors": encoderTensors,
            "finalTensor": x
        }
        return result


    def forward(self, input_tensor):

        # print(input_tensor)
        x = ME.SparseTensor(coordinates=input_tensor[:, :4],
                            features=input_tensor[:, -1].view(-1, 1))
        # Encoder
        encoderOutput = self.encoder(x)
        encoderTensors = encoderOutput['encoderTensors']
        finalTensor = encoderOutput['finalTensor']

        # z = self.global_pool(finalTensor)
        z = self.max_pool(finalTensor)
        # print(z.C, z.F.sum(dim=1))
        latent = self.linear1(z)

        return latent.F


class SparseResidualEncoder(MENetworkBase):
    '''
    Minkowski Net Autoencoder for sparse tensor reconstruction.
    '''
    def __init__(self, cfg, name='mink_encoder'):
        super(SparseResidualEncoder, self).__init__(cfg)
        self.model_config = cfg[name]
        self.reps = self.model_config.get('reps', 2)
        self.depth = self.model_config.get('depth', 7)
        self.num_filters = self.model_config.get('num_filters', 16)
        self.nPlanes = [i * self.num_filters for i in range(1, self.depth+1)]
        self.input_kernel = self.model_config.get('input_kernel', 7)
        self.latent_size = self.model_config.get('latent_size', 512)
        final_tensor_shape = self.spatial_size // (2**(self.depth-1))
        self.coordConv = self.model_config.get('coordConv', False)

        self.pool_mode = self.model_config.get('pool_mode', 'global_average')

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
                m.append(ME.MinkowskiConvolution(
                    in_channels=self.nPlanes[i],
                    out_channels=self.nPlanes[i+1],
                    kernel_size=2, stride=2, dimension=self.D,
                    bias=self.allow_bias))
            m = nn.Sequential(*m)
            self.encoding_conv.append(m)
        self.encoding_conv = nn.Sequential(*self.encoding_conv)
        self.encoding_block = nn.Sequential(*self.encoding_block)

        if self.pool_mode == 'global_average':
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
                ME.MinkowskiMaxPooling(final_tensor_shape, stride=final_tensor_shape),
                ME.MinkowskiGlobalPooling())
        else:
            raise NotImplementedError

        self.linear1 = ME.MinkowskiLinear(self.nPlanes[-1], self.latent_size)


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
        '''
        x = self.input_layer(x)
        encoderTensors = [x]
        for i, layer in enumerate(self.encoding_block):
            x = self.encoding_block[i](x)
            encoderTensors.append(x)
            x = self.encoding_conv[i](x)

        result = {
            "encoderTensors": encoderTensors,
            "finalTensor": x
        }
        return result


    def forward(self, input_tensor):

        # print(input_tensor)
        x = ME.SparseTensor(coordinates=input_tensor[:, :4],
                            features=input_tensor[:, -1].view(-1, 1))
        # Encoder
        encoderOutput = self.encoder(x)
        encoderTensors = encoderOutput['encoderTensors']
        finalTensor = encoderOutput['finalTensor']

        z = self.pool(finalTensor)
        latent = self.linear1(z)

        return latent.F


class SparseResidualEncoder2(MENetworkBase):
    '''
    Minkowski Net Autoencoder for sparse tensor reconstruction.
    '''
    def __init__(self, cfg, name='sparse_encoder'):
        super(SparseResidualEncoder2, self).__init__(cfg)
        self.model_config = cfg[name]
        self.reps = self.model_config.get('reps', 2)
        self.depth = self.model_config.get('depth', 7)
        self.num_filters = self.model_config.get('num_filters', 16)
        self.nPlanes = [i * self.num_filters for i in range(1, self.depth+1)]
        self.input_kernel = self.model_config.get('input_kernel', 7)
        self.latent_size = self.model_config.get('latent_size', 512)
        final_tensor_shape = self.spatial_size // (2**(self.depth-1))
        self.coordConv = self.model_config.get('coordConv', False)
        print("Final Tensor Shape = ", final_tensor_shape)

        self.pool_mode = self.model_config.get('pool_mode', 'global_average')

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
                    has_bias=self.allow_bias))
            m = nn.Sequential(*m)
            self.encoding_block.append(m)
            m = []
            if i < self.depth-1:
                m.append(normalizations_construct(self.norm, F, **self.norm_args))
                m.append(activations_construct(
                    self.activation_name, **self.activation_args))
                m.append(ME.MinkowskiConvolution(
                    in_channels=self.nPlanes[i],
                    out_channels=self.nPlanes[i+1],
                    kernel_size=2, stride=2, dimension=self.D,
                    bias=self.allow_bias))
            m = nn.Sequential(*m)
            self.encoding_conv.append(m)
        self.encoding_conv = nn.Sequential(*self.encoding_conv)
        self.encoding_block = nn.Sequential(*self.encoding_block)

        if self.pool_mode == 'global_average':
            self.pool = ME.MinkowskiGlobalPooling()
        elif self.pool_mode == 'conv':
            self.pool = ME.MinkowskiConvolution(
                in_channels=self.nPlanes[-1],
                out_channels=self.nPlanes[-1],
                kernel_size=final_tensor_shape,
                stride=final_tensor_shape,
                dimension=self.D)
        elif self.pool_mode == 'max':
            ME.MinkowskiMaxPooling(final_tensor_shape, stride=final_tensor_shape)
        else:
            raise NotImplementedError

        self.linear1 = nn.Sequential(
            normalizations_construct(self.norm, self.nPlanes[-1], **self.norm_args),
            activations_construct(
                self.activation_name, **self.activation_args),
            ME.MinkowskiLinear(self.nPlanes[-1], self.latent_size)
        )


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
        '''
        x = self.input_layer(x)
        encoderTensors = [x]
        for i, layer in enumerate(self.encoding_block):
            x = self.encoding_block[i](x)
            encoderTensors.append(x)
            x = self.encoding_conv[i](x)

        result = {
            "encoderTensors": encoderTensors,
            "finalTensor": x
        }
        return result


    def forward(self, x):

        # Encoder
        encoderOutput = self.encoder(x)
        encoderTensors = encoderOutput['encoderTensors']
        finalTensor = encoderOutput['finalTensor']

        z = self.pool(finalTensor)
        latent = self.linear1(z)

        return latent


class SparseResEncoderNoPooling(MENetworkBase):
    '''
    Minkowski Net Autoencoder for sparse tensor reconstruction.
    '''
    def __init__(self, cfg, name='sparse_encoder'):
        super(SparseResEncoderNoPooling, self).__init__(cfg)
        self.model_config = cfg[name]
        self.reps = self.model_config.get('reps', 2)
        self.depth = self.model_config.get('depth', 7)
        self.num_filters = self.model_config.get('num_filters', 16)
        self.nPlanes = [i * self.num_filters for i in range(1, self.depth+1)]
        self.input_kernel = self.model_config.get('input_kernel', 7)
        self.latent_size = self.model_config.get('latent_size', 512)
        final_tensor_shape = self.spatial_size // (2**(self.depth-1))
        self.coordConv = self.model_config.get('coordConv', False)
        print("Final Tensor Shape = ", final_tensor_shape)

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
                    has_bias=self.allow_bias))
            m = nn.Sequential(*m)
            self.encoding_block.append(m)
            m = []
            if i < self.depth-1:
                m.append(normalizations_construct(self.norm, F, **self.norm_args))
                m.append(activations_construct(
                    self.activation_name, **self.activation_args))
                m.append(ME.MinkowskiConvolution(
                    in_channels=self.nPlanes[i],
                    out_channels=self.nPlanes[i+1],
                    kernel_size=2, stride=2, dimension=self.D,
                    bias=self.allow_bias))
            m = nn.Sequential(*m)
            self.encoding_conv.append(m)
        self.encoding_conv = nn.Sequential(*self.encoding_conv)
        self.encoding_block = nn.Sequential(*self.encoding_block)

        self.final = nn.Sequential(
            normalizations_construct(self.norm, self.nPlanes[-1], **self.norm_args),
            activations_construct(
                self.activation_name, **self.activation_args),
            ME.MinkowskiConvolution(self.nPlanes[-1], self.latent_size, 
                kernel_size=3, stride=1, 
                dimension=self.D, bias=self.allow_bias)
        )


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
        '''
        x = self.input_layer(x)
        encoderTensors = [x]
        for i, layer in enumerate(self.encoding_block):
            x = self.encoding_block[i](x)
            encoderTensors.append(x)
            x = self.encoding_conv[i](x)

        result = {
            "encoderTensors": encoderTensors,
            "finalTensor": x
        }
        return result


    def forward(self, x):

        # Encoder
        encoderOutput = self.encoder(x)
        encoderTensors = encoderOutput['encoderTensors']
        finalTensor = encoderOutput['finalTensor']

        z = self.final(finalTensor)

        return z
