import torch
import torch.nn as nn
import MinkowskiEngine as ME

from mlreco.models.layers.activation_normalization_factories import activations_construct
from mlreco.models.layers.activation_normalization_factories import normalizations_construct
from mlreco.models.layers.configuration import setup_cnn_configuration
from mlreco.models.layers.blocks import DropoutBlock, ResNetBlock

class MCDropoutDecoder(torch.nn.Module):
    """
    Convolutional decoder with dropout layers.

    The architecture is exactly the same as the ME ResidualEncoders,
    except for the additional DropoutBlocks

    Attributes:

        dropout_p: dropping probability value for dropout layers
        dropout_layer_index: layer numbers to swap resnet blocks with
        dropout resnet blocks.
    """

    def __init__(self, cfg, name='mcdropout_decoder'):
        super(MCDropoutDecoder, self).__init__()
        setup_cnn_configuration(self, cfg, name)

        self.model_config = cfg[name]

        print(self.model_config)

        # UResNet Configurations
        self.model_config = cfg[name]
        self.reps         = self.model_config.get('reps', 2)
        self.kernel_size  = self.model_config.get('kernel_size', 2)
        self.depth        = self.model_config.get('depth', 5)
        self.num_filters  = self.model_config.get('num_filters', 16)
        self.nPlanes      = [i * self.num_filters for i in range(1, self.depth+1)]
        self.downsample   = [self.kernel_size, 2]

        self.enc_nfilters = self.model_config.get(
            'encoder_num_filters', None)
        if self.enc_nfilters is None:
            self.enc_nfilters = self.num_filters
        self.encoder_nPlanes = [i * self.enc_nfilters for i in range(1, self.depth + 1)]
        self.nPlanes[-1] = self.encoder_nPlanes[-1]

        self.dropout_p = self.model_config['dropout_p']
        self.dropout_layer_index = self.model_config.get(
            'dropout_layers', set([i for i in range(self.depth // 2, self.depth)]))

        # Initialize Decoder
        self.decoding_block = []
        self.decoding_conv = []
        for i in range(self.depth-2, -1, -1):
            m = []
            m.append(normalizations_construct(self.norm,
                                              self.nPlanes[i+1],
                                              **self.norm_args))
            m.append(activations_construct(self.activation_name,
                                           **self.activation_args))
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
                                          bias=self.allow_bias))
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
