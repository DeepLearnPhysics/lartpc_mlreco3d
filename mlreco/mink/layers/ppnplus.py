import numpy as np
import torch
import torch.nn as nn

import MinkowskiEngine as ME
import MinkowskiFunctional as MF

from mlreco.mink.layers.blocks import ResNetBlock, CascadeDilationBlock, SPP, ASPP
from mlreco.mink.layers.factories import activations_dict, activations_construct
from mlreco.mink.layers.network_base import MENetworkBase


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, x, scores):
        features = x.F
        features = features * scores
        coords = x.C
        output = ME.SparseTensor(
            coords=coords, feats=features)
        return output


class ExpandAs(nn.Module):
    def __init__(self):
        super(ExpandAs, self).__init__()

    def forward(self, x, shape):
        device = x.F.device
        features = x.F.expand(*shape)
        output = ME.SparseTensor(
            feats=features,
            coords_key=x.coords_key,
            coords_manager=x.coords_man)
        return output

class PPNTest(MENetworkBase):
    '''
    Vanilla UResNet with access to intermediate feature planes.

    Configurations
    --------------
    depth : int
        Depth of UResNet, also corresponds to how many times we down/upsample.
    num_filters : int
        Number of filters in the first convolution of UResNet.
        Will increase linearly with depth.
    reps : int, optional
        Convolution block repetition factor
    kernel_size : int, optional
        Kernel size for the SC (sparse convolutions for down/upsample).
    input_kernel : int, optional
        Receptive field size for very first convolution after input layer.
    '''
    def __init__(self, cfg, name='ppn'):
        super(PPNTest, self).__init__(cfg)
        model_cfg = cfg[name]
        # UResNet Configurations
        self.reps = model_cfg.get('reps', 2)
        self.depth = model_cfg.get('depth', 5)
        self.num_filters = model_cfg.get('num_filters', 16)
        self.nPlanes = [i * self.num_filters for i in range(1, self.depth+1)]
        self.ppn_score_threshold = model_cfg.get('ppn_score_threshold', 0.5)
        # self.kernel_size = cfg.get('kernel_size', 3)
        # self.downsample = cfg.get(downsample, 2)
        self.input_kernel = model_cfg.get('input_kernel', 3)

        # Initialize Input Layer
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

        # Initialize Decoder
        self.decoding_block = []
        self.decoding_conv = []
        self.ppn_pred = nn.ModuleList()
        for i in range(self.depth-2, -1, -1):
            m = []
            m.append(ME.MinkowskiBatchNorm(self.nPlanes[i+1]))
            m.append(activations_construct(
                self.activation_name, **self.activation_args))
            m.append(ME.MinkowskiConvolutionTranspose(
                in_channels=self.nPlanes[i+1],
                out_channels=self.nPlanes[i],
                kernel_size=2,
                stride=2,
                dimension=self.D))
            m = nn.Sequential(*m)
            self.decoding_conv.append(m)
            m = []
            for j in range(self.reps):
                m.append(ResNetBlock(self.nPlanes[i] * (2 if j == 0 else 1),
                                     self.nPlanes[i],
                                     dimension=self.D,
                                     activation=self.activation_name,
                                     activation_args=self.activation_args))
            m = nn.Sequential(*m)
            self.decoding_block.append(m)
            self.ppn_pred.append(ME.MinkowskiLinear(self.nPlanes[i], 1))
        self.decoding_block = nn.Sequential(*self.decoding_block)
        self.decoding_conv = nn.Sequential(*self.decoding_conv)

        self.sigmoid = ME.MinkowskiSigmoid()
        self.bcst = ME.MinkowskiBroadcastMultiplication()
        self.pool = ME.MinkowskiGlobalPooling()
        self.expand_as = ExpandAs()


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


    def decoder(self, final, encoderTensors):
        '''
        Vanilla UResNet Decoder
        INPUTS:
            - encoderTensors (list of SparseTensor): output of encoder.
        RETURNS:
            - decoderTensors (list of SparseTensor):
            list of feature tensors in decoding path at each spatial resolution.
        '''
        decoderTensors = []
        ppn_scores, ppn_logits = [], []
        tmp = []
        x = final
        for i, layer in enumerate(self.decoding_conv):
            eTensor = encoderTensors[-i-2]
            x = layer(x)
            x = ME.cat(eTensor, x)
            x = self.decoding_block[i](x)
            decoderTensors.append(x)
            scores = self.ppn_pred[i](x)
            ppn_logits.append(scores.F)
            scores = self.sigmoid(scores)
            tmp.append(x.C)
            ppn_scores.append(scores.F)
            with torch.no_grad():
                s_expanded = self.expand_as(scores, x.F.shape)
            x = x * s_expanded
        device = x.F.device
        points = []
        for p in tmp:
            a = p.to(dtype=torch.float32, device=device)
            points.append(a)
        return decoderTensors, ppn_scores, ppn_logits, points


    def forward(self, input):
        coords = input[:, 0:self.D+1].cpu().int()
        features = input[:, self.D+1:].float()

        x = ME.SparseTensor(features, coords=coords)
        encoderOutput = self.encoder(x)
        encoderTensors = encoderOutput['encoderTensors']
        finalTensor = encoderOutput['finalTensor']
        decoderTensors, ppn_scores, ppn_logits, points = self.decoder(finalTensor, encoderTensors)

        res = {
            'encoderTensors': [encoderTensors],
            'decoderTensors': [decoderTensors],
            'ppn_scores': [ppn_scores],
            'ppn_logits': [ppn_logits],
            'points': [points], 
            'finalTensor': [finalTensor]
        }
        return res