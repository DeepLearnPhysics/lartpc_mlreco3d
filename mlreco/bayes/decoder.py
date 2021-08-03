import torch
import torch.nn as nn
import MinkowskiEngine as ME

import MinkowskiFunctional as MF

from collections import defaultdict
from mlreco.mink.layers.factories import activations_dict, activations_construct, normalizations_construct
from mlreco.mink.layers.network_base import MENetworkBase
from mlreco.mink.layers.blocks import DropoutBlock, ResNetBlock


class BayesianDecoder(MENetworkBase):

    def __init__(self, cfg, name='bayesian_decoder'):
        super(BayesianDecoder, self).__init__(cfg, name='network_base')
        self.model_config = cfg[name]

        print(self.model_config)

        # UResNet Configurations
        self.model_config = cfg[name]
        self.reps = self.model_config.get('reps', 2)  # Conv block repetition factor
        self.kernel_size = self.model_config.get('kernel_size', 2)
        self.depth = self.model_config.get('depth', 5)
        self.num_filters = self.model_config.get('num_filters', 16)
        self.nPlanes = [i*self.num_filters for i in range(1, self.depth+1)]
        self.downsample = [self.kernel_size, 2]  # [filter size, filter stride]

        self.encoder_num_filters = self.model_config.get('encoder_num_filters', None)
        if self.encoder_num_filters is None:
            self.encoder_num_filters = self.num_filters
        self.encoder_nPlanes = [i*self.encoder_num_filters for i in range(1, self.depth+1)]
        self.nPlanes[-1] = self.encoder_nPlanes[-1]

        self.dropout_p = self.model_config['dropout_p']
        self.dropout_layer_index = self.model_config.get(
            'dropout_layers', set([i for i in range(self.depth // 2, self.depth)]))

        self.debug = self.model_config.get('debug', False)

        print("Dropout Layers = ", self.dropout_layer_index)
        print("Planes = ", len(self.nPlanes))

        # Initialize Decoder
        self.decoding_block = []
        self.decoding_conv = []
        for i in range(self.depth-2, -1, -1):
            m = []
            m.append(normalizations_construct(self.norm, self.nPlanes[i+1], **self.norm_args))
            m.append(activations_construct(
                self.activation_name, **self.activation_args))
            m.append(ME.MinkowskiConvolutionTranspose(
                in_channels=self.nPlanes[i+1],
                out_channels=self.nPlanes[i],
                kernel_size=2,
                stride=2,
                dimension=self.D))
            if i in self.dropout_layer_index:
                m.append(ME.MinkowskiDropout(p=self.dropout_p))
            m = nn.Sequential(*m)
            self.decoding_conv.append(m)
            m = []
            for j in range(self.reps):
                if i in self.dropout_layer_index:
                    m.append(DropoutBlock(self.nPlanes[i] * (2 if j == 0 else 1),
                                          self.nPlanes[i],
                                          dimension=self.D,
                                          p=self.dropout_p,
                                          activation=self.activation_name,
                                          activation_args=self.activation_args,
                                          normalization=self.norm,
                                          normalization_args=self.norm_args,
                                          bias=self.allow_bias,
                                          debug=self.debug))
                else:
                    m.append(ResNetBlock(self.nPlanes[i] * (2 if j == 0 else 1),
                                         self.nPlanes[i],
                                         dimension=self.D,
                                         activation=self.activation_name,
                                         activation_args=self.activation_args,
                                         normalization=self.norm,
                                         normalization_args=self.norm_args,
                                         bias=self.allow_bias))
            m = nn.Sequential(*m)
            self.decoding_block.append(m)
        self.decoding_block = nn.Sequential(*self.decoding_block)
        self.decoding_conv = nn.Sequential(*self.decoding_conv)


    def forward(self, final, encoderTensors):
        '''
        Vanilla UResNet Decoder
        INPUTS:
            - encoderTensors (list of SparseTensor): output of encoder.
        RETURNS:
            - decoderTensors (list of SparseTensor):
            list of feature tensors in decoding path at each spatial resolution.
        '''
        decoderTensors = []
        x = final
        for i, layer in enumerate(self.decoding_conv):
            eTensor = encoderTensors[-i-2]
            x = layer(x)
            x = ME.cat(eTensor, x)
            x = self.decoding_block[i](x)
            decoderTensors.append(x)
        return decoderTensors