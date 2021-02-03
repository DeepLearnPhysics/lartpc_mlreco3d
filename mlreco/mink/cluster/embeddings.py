import numpy as np
import torch
import torch.nn as nn

import MinkowskiEngine as ME
import MinkowskiFunctional as MF

from collections import defaultdict
from mlreco.mink.layers.factories import activations_dict, activations_construct
from mlreco.mink.layers.network_base import MENetworkBase
from mlreco.mink.layers.blocks import ResNetBlock, ConvolutionBlock
from mlreco.mink.layers.uresnet import UResNetEncoder, UResNetDecoder


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, x, scores):
        features = x.F
        features = features * scores
        coords = x.C
        output = ME.SparseTensor(
            coordinates=coords, feats=features)
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


class ClusterResNet(MENetworkBase):

    def __init__(self, cfg, name='cluster_resnet'):
        super(ClusterResNet, self).__init__(cfg)
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

        self.coordConv = model_cfg.get('coordConv', True)

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

        self.point_prediction = ME.MinkowskiLinear(self.nPlanes[0], 3)
        self.embedding = ME.MinkowskiLinear(self.nPlanes[0], 4)
        self.tanh = nn.Tanh()

        self.margin_fn_name = model_cfg.get('margin_fn', 'sigmoid')
        self.sigmoid = ME.MinkowskiSigmoid()

        if self.margin_fn_name == 'sigmoid':
            self.margin_fn = nn.Sigmoid()
        elif self.margin_fn_name == 'exp':
            self.margin_fn = torch.exp
        elif self.margin_fn_name == 'softplus':
            self.margin_fn = nn.Softplus()
        else:
            raise ValueError
        self.expand_as = ExpandAs()

        self.embedding_dec = UResNetDecoder(cfg, name='embedding_dec')


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
        return decoderTensors, ppn_scores, ppn_logits, points, x


    def forward(self, input):
        coords = input[:, 0:self.D+1]
        features = input[:, self.D+1:].float()

        normalized_coords = (coords[:, 1:4] - float(self.spatial_size) / 2) \
                    / (float(self.spatial_size) / 2)
        if self.coordConv:
            features = torch.cat([normalized_coords, features], dim=1)

        x = ME.SparseTensor(features, coordinates=coords.cpu().int())
        encoderOutput = self.encoder(x)
        encoderTensors = encoderOutput['encoderTensors']
        finalTensor = encoderOutput['finalTensor']
        decoderTensors, ppn_scores, ppn_logits, points, final = self.decoder(finalTensor, encoderTensors)
        point_predictions = self.point_prediction(final).F
        embedding_dec = self.embedding_dec(finalTensor, encoderTensors)
        embeddings = self.embedding(embedding_dec[-1]).F
        embeddings[:, :self.D] = self.tanh(embeddings[:, :self.D])
        embeddings[:, :self.D] += normalized_coords

        res = {
            'encoderTensors': [encoderTensors],
            'decoderTensors': [decoderTensors],
            'ppn_scores': [ppn_scores],
            'ppn_logits': [ppn_logits],
            'points': [points], 
            'point_predictions': [point_predictions],
            'finalTensor': [finalTensor],
            'embeddings': [embeddings[:, :self.D]],
            'margins': [self.margin_fn(embeddings[:, self.D:])]
        }
        return res


class SPICE(MENetworkBase):

    def __init__(self, cfg, name='spice'):
        super(SPICE, self).__init__(cfg)
        self.model_config = cfg[name]
        self.encoder = UResNetEncoder(cfg, name='uresnet_encoder')
        self.embedding_decoder = UResNetDecoder(cfg, name='embedding_decoder')
        self.seed_decoder = UResNetDecoder(cfg, name='seediness_decoder')

        self.num_filters = self.model_config.get('num_filters', 16)
        self.seedDim = self.model_config.get('seediness_dim', 1)
        self.coordConv = self.model_config.get('coordConv', False)
        self.sigmaDim = self.model_config.get('sigma_dim', 1)
        self.seed_freeze = self.model_config.get('seed_freeze', False)
        self.coordConv = self.model_config.get('coordConv', True)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()


        self.outputEmbeddings = nn.Sequential(
            ME.MinkowskiBatchNorm(self.num_filters, **self.norm_args),
            ME.MinkowskiLinear(self.num_filters, self.D + self.sigmaDim, bias=False)
        )

        self.outputSeediness = nn.Sequential(
            ME.MinkowskiBatchNorm(self.num_filters, **self.norm_args),
            ME.MinkowskiLinear(self.num_filters, self.seedDim, bias=False)
        )

        if self.seed_freeze:
            print('Seediness Branch Freezed')
            for p in self.seed_decoder.parameters():
                p.requires_grad = False
            for p in self.outputSeediness.parameters():
                p.requires_grad = False



    def forward(self, input):

        point_cloud, = input
        device = point_cloud.device

        coords = point_cloud[:, 0:self.D+1].to(device)
        features = point_cloud[:, self.D+1:].float().view(-1, 1)

        normalized_coords = (coords[:, 1:self.D+1] - float(self.spatial_size) / 2) \
                    / (float(self.spatial_size) / 2)
        normalized_coords = normalized_coords.float().cuda()
        if self.coordConv:
            features = torch.cat([normalized_coords, features], dim=1)


        x = ME.SparseTensor(features, coordinates=coords)

        encoderOutput = self.encoder(x)
        encoderTensors = encoderOutput['encoderTensors']
        finalTensor = encoderOutput['finalTensor']
        features_cluster = self.embedding_decoder(finalTensor, encoderTensors)
        features_seediness = self.seed_decoder(finalTensor, encoderTensors)

        embeddings = self.outputEmbeddings(features_cluster[-1])
        embeddings_feats = embeddings.F
        embeddings_feats[:, :self.D] = self.tanh(embeddings_feats[:, :self.D])
        embeddings_feats[:, :self.D] += normalized_coords
        seediness = self.outputSeediness(features_seediness[-1])

        res = {
            'embeddings': [embeddings_feats[:, :self.D]],
            'seediness': [self.sigmoid(seediness.F)],
            'margins': [2 * self.sigmoid(embeddings_feats[:, self.D:])], 
        }
        return res



