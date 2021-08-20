import torch
import torch.nn as nn
import MinkowskiEngine as ME

import MinkowskiFunctional as MF

from collections import defaultdict
from mlreco.models.layers.common.activation_normalization_factories import activations_dict, activations_construct, normalizations_construct
from mlreco.models.layers.common.configuration import setup_cnn_configuration
from mlreco.models.layers.common.blocks import DropoutBlock, ResNetBlock, Identity


class MCDropoutEncoder(torch.nn.Module):
    """
    Convolutional decoder with dropout layers.

    The architecture is exactly the same as the ME ResidualEncoders,
    except for the additional DropoutBlocks

    Attributes:

        dropout_p: dropping probability value for dropout layers
        dropout_layer_index: layer numbers to swap resnet blocks with
        dropout resnet blocks.
    """
    def __init__(self, cfg, name='mcdropout_encoder'):
        super(MCDropoutEncoder, self).__init__()
        setup_cnn_configuration(self, cfg, name)

        self.model_config = cfg.get(name, {})
        self.reps = self.model_config.get('reps', 2)
        self.depth = self.model_config.get('depth', 7)
        self.num_filters = self.model_config.get('num_filters', 16)
        self.nPlanes = [i * self.num_filters for i in range(1, self.depth+1)]
        self.input_kernel = self.model_config.get('input_kernel', 3)
        self.latent_size = self.model_config.get('latent_size', 512)
        final_tensor_shape = self.spatial_size // (2**(self.depth-1))
        self.coordConv = self.model_config.get('coordConv', False)

        self.pool_mode = self.model_config.get('pool_mode', 'global_average')

        self.dropout_p = self.model_config.get('dropout_p', 0.5)
        self.dropout_layer_index = self.model_config.get(
            'dropout_layers', set([i for i in range(self.depth // 2, self.depth)]))

        self.add_classifier = self.model_config.get('add_classifier', True)

        # Initialize Input Layer
        if self.coordConv:
            self.input_layer = nn.Sequential(
                ME.MinkowskiConvolution(
                    in_channels=self.num_input + self.D,
                    out_channels=self.num_filters,
                    kernel_size=self.input_kernel, stride=1, dimension=self.D))
        else:
            self.input_layer = nn.Sequential(
                ME.MinkowskiConvolution(
                    in_channels=self.num_input,
                    out_channels=self.num_filters,
                    kernel_size=self.input_kernel, stride=1, dimension=self.D))

        # Initialize Encoder
        self.encoding_conv = []
        self.encoding_block = []
        for i, F in enumerate(self.nPlanes):
            m = []
            for _ in range(self.reps):
                if i in self.dropout_layer_index:
                    m.append(DropoutBlock(F, F,
                        dimension=self.D,
                        p=self.dropout_p,
                        activation=self.activation_name,
                        activation_args=self.activation_args,
                        normalization=self.norm,
                        normalization_args=self.norm_args,
                        bias=self.allow_bias))
                else:
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
                if i in self.dropout_layer_index:
                    m.append(ME.MinkowskiDropout(p=self.dropout_p))
                else:
                    pass
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
                ME.MinkowskiDropout(p=self.dropout_p),
                ME.MinkowskiGlobalPooling())
        elif self.pool_mode == 'max':
            self.pool = nn.Sequential(
                ME.MinkowskiMaxPooling(final_tensor_shape, stride=final_tensor_shape),
                ME.MinkowskiGlobalPooling())
        elif self.pool_mode == 'no_pooling':
            self.pool = Identity()
        else:
            raise NotImplementedError

        if self.add_classifier:
            self.linear1 = nn.Sequential(
                ME.MinkowskiReLU(),
                ME.MinkowskiLinear(self.nPlanes[-1], self.latent_size))
        else:
            self.linear1 = Identity()


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
