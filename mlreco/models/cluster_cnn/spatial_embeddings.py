import torch
import torch.nn as nn
import numpy as np
import sparseconvnet as scn
from collections import defaultdict

from mlreco.models.layers.uresnet import UResNet
from mlreco.models.ppn import PPN, PPNLoss
from mlreco.models.layers.stacknet import StackUNet
from .utils import add_normalized_coordinates, distance_matrix

class SpatialEmbeddings1(UResNet):

    def __init__(self, cfg, name='spatial_embeddings'):
        super(SpatialEmbeddings1, self).__init__(cfg, name='uresnet')
        if 'modules' in cfg:
            self.model_config = cfg['modules'][name]
        else:
            self.model_config = cfg
        self.seedDim = self.model_config.get('seediness_dim', 1)
        self.sigmaDim = self.model_config.get('sigma_dim', 1)
        self.seed_freeze = self.model_config.get('seed_freeze', False)
        # Define Separate Sparse UResNet Decoder for seediness.
        self.decoding_block2 = scn.Sequential()
        self.decoding_conv2 = scn.Sequential()
        for i in range(self.num_strides-2, -1, -1):
            m = scn.Sequential().add(
                scn.BatchNormLeakyReLU(self.nPlanes[i+1], leakiness=self.leakiness)).add(
                scn.Deconvolution(self.dimension, self.nPlanes[i+1], self.nPlanes[i],
                    self.downsample[0], self.downsample[1], self.allow_bias))
            self.decoding_conv2.add(m)
            m = scn.Sequential()
            for j in range(self.reps):
                self._resnet_block(m, self.nPlanes[i] * (2 if j == 0 else 1), self.nPlanes[i])
            self.decoding_block2.add(m)
        
        # Define outputlayers
        self.outputEmbeddings = scn.Sequential()
        self._nin_block(self.outputEmbeddings, self.num_filters, self.dimension + self.sigmaDim)
        self.outputEmbeddings.add(scn.OutputLayer(self.dimension))
        self.outputSeediness = scn.Sequential()
        self._nin_block(self.outputSeediness, self.num_filters, self.seedDim)
        self.outputSeediness.add(scn.OutputLayer(self.dimension))

        if self.seed_freeze:
            print('Seediness Branch Freezed')
            for p in self.decoding_block2.parameters():
                p.requires_grad = False
            for p in self.decoding_conv2.parameters():
                p.requires_grad = False
            for p in self.outputSeediness.parameters():
                p.requires_grad = False

        # Pytorch Activations
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()


    def seed_decoder(self, features_enc, deepest_layer):
        '''
        Decoder for seediness map.

        INPUTS:
            - features_enc (list of scn.SparseConvNetTensor): output of encoder.

        RETURNS:
            - features_dec (list of scn.SparseConvNetTensor): list of feature
            tensors in decoding path at each spatial resolution. 
        '''
        features_seediness = []
        x = deepest_layer
        for i, layer in enumerate(self.decoding_conv2):
            encoder_feature = features_enc[-i-2]
            x = layer(x)
            x = self.concat([encoder_feature, x])
            x = self.decoding_block2[i](x)
            features_seediness.append(x)
        return features_seediness


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
        features = features[:, -1].view(-1, 1)

        x = self.input((coords, features))
        encoder_res = self.encoder(x)
        features_enc = encoder_res['features_enc']
        deepest_layer = encoder_res['deepest_layer']
        features_cluster = self.decoder(features_enc, deepest_layer)
        features_seediness = self.seed_decoder(features_enc, deepest_layer)

        embeddings = self.outputEmbeddings(features_cluster[-1])
        embeddings[:, :self.dimension] = self.tanh(embeddings[:, :self.dimension])
        embeddings[:, :self.dimension] += coords[:, :self.dimension] / self.spatial_size
        seediness = self.outputSeediness(features_seediness[-1])

        res = {
            "embeddings": [embeddings[:, :self.dimension]],
            "margins": [self.sigmoid(embeddings[:, self.dimension:])],
            "seediness": [seediness]
        }

        return res


class SpatialEmbeddings2(StackUNet):

    def __init__(self, cfg, name='spatial_embeddings'):
        super(SpatialEmbeddings2, self).__init__(cfg, name=name)
        if 'modules' in cfg:
            self.model_config = cfg['modules'][name]
        else:
            self.model_config = cfg
        self.seedDim = self.model_config.get('seediness_dim', 1)
        self.sigmaDim = self.model_config.get('sigma_dim', 1)
        # Define Separate Sparse UResNet Decoder for seediness.
        self.decoding_block2 = scn.Sequential()
        self.decoding_conv2 = scn.Sequential()
        for i in range(self.num_strides-2, -1, -1):
            m = scn.Sequential().add(
                scn.BatchNormLeakyReLU(self.nPlanes[i+1], leakiness=self.leakiness)).add(
                scn.Deconvolution(self.dimension, self.nPlanes[i+1], self.nPlanes[i],
                    self.downsample[0], self.downsample[1], self.allow_bias))
            self.decoding_conv2.add(m)
            m = scn.Sequential()
            for j in range(self.reps):
                self._resnet_block(m, self.nPlanes[i] * (2 if j == 0 else 1), self.nPlanes[i])
            self.decoding_block2.add(m)
        
        # Define outputlayers
        self.embedding = None

        self.outputEmbeddings = scn.Sequential()
        self._nin_block(self.outputEmbeddings, self.num_filters, self.dimension + self.sigmaDim)
        self.outputEmbeddings.add(scn.OutputLayer(self.dimension))
        self.outputSeediness = scn.Sequential()
        self._nin_block(self.outputSeediness, self.num_filters, self.seedDim)
        self.outputSeediness.add(scn.Sigmoid())
        self.outputSeediness.add(scn.OutputLayer(self.dimension))

        # Pytorch Activations
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()


    def seed_decoder(self, features_enc, deepest_layer):
        '''
        Decoder for seediness map.

        INPUTS:
            - features_enc (list of scn.SparseConvNetTensor): output of encoder.

        RETURNS:
            - features_dec (list of scn.SparseConvNetTensor): list of feature
            tensors in decoding path at each spatial resolution. 
        '''
        features_seediness = []
        x = deepest_layer
        for i, layer in enumerate(self.decoding_conv2):
            encoder_feature = features_enc[-i-2]
            x = layer(x)
            x = self.concat([encoder_feature, x])
            x = self.decoding_block2[i](x)
            features_seediness.append(x)
        return features_seediness


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
        features_dec = [deepest_layer] + features_dec
        stack_feature = []

        for i, layer in enumerate(features_dec):
            if i < self.num_strides-1:
                f = self.unpooling[i](layer)
                stack_feature.append(f)
            else:
                stack_feature.append(layer)
        
        stack_feature = self.concat(stack_feature)
        out = self.cluster_decoder(stack_feature)
        features_seediness = self.seed_decoder(features_enc, deepest_layer)

        embeddings = self.outputEmbeddings(out)
        embeddings[:, :self.dimension] = self.tanh(embeddings[:, :self.dimension])
        embeddings[:, :self.dimension] += coords[:, :self.dimension] / self.spatial_size
        seediness = self.outputSeediness(features_seediness[-1])

        res = {
            "embeddings": [embeddings[:, :self.dimension]],
            "margins": [self.sigmoid(embeddings[:, self.dimension:])],
            "seediness": [seediness]
        }

        return res


# class UResNetClust(UResNet):
#     '''
#     Clustering model for new dataset and full reconstruction chain.

#     TODO: Implement full multivariate gaussian mixtures. This can be done via
#     generating 6 dim margins and computing the covariance matrix via the 
#     Cholesky decomposition \Sigma = LL^T. 
#     '''

#     def __init__(self, cfg, name='full_chain'):
#         super(UResNetClust, self).__init__(cfg, name='uresnet')
#         self.model_config = cfg[name]
#         # print(self.model_config)

#         self.ghost = self.model_config.get('ghost', False)
#         self.seedDim = self.model_config.get('seediness_dim', 1)
#         self.sigmaDim = self.model_config.get('sigma_dim', 6)
#         self.embedding_dim = self.model_config.get('embedding_dim', 3)
#         self.num_classes = self.model_config.get('num_classes', 5)
#         self.num_aggFeatures = self.model_config.get('num_aggFeatures', 16)

#         self.ppn_freeze = self.model_config.get('ppn_freeze', True)
#         self.segmentation_freeze = self.model_config.get('segmentation_freeze', False)
#         self.embedding_freeze = self.model_config.get('embedding_freeze', True)

#         # Define PPN
#         self.ppn  = PPN(cfg)

#         # Define Separate Sparse UResNet Decoder for Seediness.
#         self.decoding_block2 = scn.Sequential()
#         self.decoding_conv2 = scn.Sequential()
#         for i in range(self.num_strides-2, -1, -1):
#             m = scn.Sequential().add(
#                 scn.BatchNormLeakyReLU(self.nPlanes[i+1], leakiness=self.leakiness)).add(
#                 scn.Deconvolution(self.dimension, self.nPlanes[i+1], self.nPlanes[i],
#                     self.downsample[0], self.downsample[1], self.allow_bias))
#             self.decoding_conv2.add(m)
#             m = scn.Sequential()
#             for j in range(self.reps):
#                 self._resnet_block(m, self.nPlanes[i] * (2 if j == 0 else 1), self.nPlanes[i])
#             self.decoding_block2.add(m)

#         # Embeddings
#         self.decoding_block3 = scn.Sequential()
#         self.decoding_conv3 = scn.Sequential()
#         for i in range(self.num_strides-2, -1, -1):
#             m = scn.Sequential().add(
#                 scn.BatchNormLeakyReLU(self.nPlanes[i+1], leakiness=self.leakiness)).add(
#                 scn.Deconvolution(self.dimension, self.nPlanes[i+1], self.nPlanes[i],
#                     self.downsample[0], self.downsample[1], self.allow_bias))
#             self.decoding_conv3.add(m)
#             m = scn.Sequential()
#             for j in range(self.reps):
#                 self._resnet_block(m, self.nPlanes[i] * (2 if j == 0 else 1), self.nPlanes[i])
#             self.decoding_block3.add(m)
    
#         # Define outputlayers
#         self.outputEmbeddings = scn.Sequential()
#         self._nin_block(self.outputEmbeddings, self.num_filters, self.dimension + self.sigmaDim)
#         self.outputEmbeddings.add(scn.OutputLayer(self.dimension))
#         self.outputSeediness = scn.Sequential()
#         self._nin_block(self.outputSeediness, self.num_filters, self.seedDim)
#         self.outputSeediness.add(scn.OutputLayer(self.dimension))
#         self.outputSegmentation = scn.Sequential()
#         self._nin_block(self.outputSegmentation, self.num_filters, self.num_classes)
#         self.outputSegmentation.add(scn.OutputLayer(self.dimension))

#         if self.ppn_freeze:
#             print('Seediness Branch Freezed')
#             for p in self.decoding_block2.parameters():
#                 p.requires_grad = False
#             for p in self.decoding_conv2.parameters():
#                 p.requires_grad = False
#             for p in self.outputSeediness.parameters():
#                 p.requires_grad = False

#         if self.segmentation_freeze:
#             print('Classification Branch Freezed')
#             for p in self.decoding_block.parameters():
#                 p.requires_grad = False
#             for p in self.decoding_conv.parameters():
#                 p.requires_grad = False
#             for p in self.outputSegmentation.parameters():
#                 p.requires_grad = False

#         if self.embedding_freeze:
#             print('Embedding Branch Freezed')
#             for p in self.decoding_block3.parameters():
#                 p.requires_grad = False
#             for p in self.decoding_conv3.parameters():
#                 p.requires_grad = False
#             for p in self.outputEmbeddings.parameters():
#                 p.requires_grad = False

#         # Pytorch Activations
#         self.tanh = nn.Tanh()
#         self.sigmoid = nn.Sigmoid()

#         # if self.ghost:
#         #     self.linear_ghost = scn.Sequential()
#         #     self._nin_block(self.linear_ghost, self.num_filters, 2)
#         #     self.linear_ghost.add(scn.OutputLayer(self.dimension))

#         self.outputAggFeatures = scn.Sequential()
#         self._nin_block(self.outputAggFeatures, self.num_filters*3, self.num_aggFeatures)
#         self.outputAggFeatures.add(scn.OutputLayer(self.dimension))


#     def encoder(self, x):
#         '''
#         Vanilla UResNet Encoder

#         INPUTS:
#             - x (scn.SparseConvNetTensor): output from inputlayer (self.input)

#         RETURNS:
#             - features_encoder (list of SparseConvNetTensor): list of feature 
#             tensors in encoding path at each spatial resolution.
#         '''
#         # Embeddings at each layer
#         feature_maps = [x]
#         feature_ppn = [x]
#         # Loop over Encoding Blocks to make downsampled segmentation/clustering masks.
#         for i, layer in enumerate(self.encoding_block):
#             x = self.encoding_block[i](x)
#             feature_maps.append(x)
#             x = self.encoding_conv[i](x)
#             feature_ppn.append(x)
        
#         res = {
#             "features_enc": feature_maps,
#             "features_ppn": feature_ppn,
#             "deepest_layer": x
#         }

#         return res


#     def embedding_decoder(self, features_enc, deepest_layer):
#         '''
#         Decoder for seediness map.

#         INPUTS:
#             - features_enc (list of scn.SparseConvNetTensor): output of encoder.

#         RETURNS:
#             - features_dec (list of scn.SparseConvNetTensor): list of feature
#             tensors in decoding path at each spatial resolution. 
#         '''
#         features_emb = []
#         x = deepest_layer
#         for i, layer in enumerate(self.decoding_conv3):
#             encoder_feature = features_enc[-i-2]
#             x = layer(x)
#             x = self.concat([encoder_feature, x])
#             x = self.decoding_block3[i](x)
#             features_emb.append(x)
#         return features_emb


#     def seediness_decoder(self, features_enc, deepest_layer):
#         '''
#         Decoder for seediness map.

#         INPUTS:
#             - features_enc (list of scn.SparseConvNetTensor): output of encoder.

#         RETURNS:
#             - features_dec (list of scn.SparseConvNetTensor): list of feature
#             tensors in decoding path at each spatial resolution. 
#         '''
#         features_seediness = []
#         x = deepest_layer
#         for i, layer in enumerate(self.decoding_conv2):
#             encoder_feature = features_enc[-i-2]
#             x = layer(x)
#             x = self.concat([encoder_feature, x])
#             x = self.decoding_block2[i](x)
#             features_seediness.append(x)
#         return features_seediness


#     def segment_decoder(self, features_enc, deepest_layer):
#         '''
#         Decoder for segmentation map.

#         INPUTS:
#             - features_enc (list of scn.SparseConvNetTensor): output of encoder.

#         RETURNS:
#             - features_dec (list of scn.SparseConvNetTensor): list of feature
#             tensors in decoding path at each spatial resolution. 
#         '''
#         features_seg = []
#         x = deepest_layer
#         for i, layer in enumerate(self.decoding_conv):
#             encoder_feature = features_enc[-i-2]
#             x = layer(x)
#             x = self.concat([encoder_feature, x])
#             x = self.decoding_block[i](x)
#             features_seg.append(x)
#         return features_seg

    
#     def forward(self, input):
#         '''
#         point_cloud is a list of length minibatch size (assumes mbs = 1)
#         point_cloud[0] has 3 spatial coordinates + 1 batch coordinate + 1 feature
#         label has shape (point_cloud.shape[0] + 5*num_labels, 1)
#         label contains segmentation labels for each point + coords of gt points

#         RETURNS:
#             - feature_enc: encoder features at each spatial resolution.
#             - feature_dec: decoder features at each spatial resolution.
#         '''
#         point_cloud, = input
#         coords = point_cloud[:, 0:self.dimension+1].float()
#         # print(coords.shape)
#         normalized_coords = (coords[:, :self.embedding_dim] - float(self.spatial_size) / 2) \
#                     / (float(self.spatial_size) / 2)
#         energy = point_cloud[:, self.dimension+1].float().view(-1, 1)
#         # if self.coordConv:
#         #     features = torch.cat([normalized_coords, features], dim=1)

#         x = self.input((coords, energy))
#         # coords = x.get_spatial_locations()
#         # print(coords.shape)
#         # normalized_coords = (coords[:, :self.embedding_dim].float() - float(self.spatial_size) / 2) \
#         #             / (float(self.spatial_size) / 2)
#         encoder_res = self.encoder(x)
#         features_enc = encoder_res['features_enc']
#         deepest_layer = encoder_res['deepest_layer']
#         features_cluster = self.embedding_decoder(features_enc, deepest_layer)
#         features_seediness = self.seediness_decoder(features_enc, deepest_layer)
#         features_seg = self.segment_decoder(features_enc, deepest_layer)

#         segmentation = features_seg[-1]
#         embeddings = features_cluster[-1]
#         seediness = features_seediness[-1]

#         features_aggregate = self.concat([segmentation, seediness, embeddings])
#         features_aggregate = self.outputAggFeatures(features_aggregate)

#         embeddings = self.outputEmbeddings(embeddings)
#         embeddings[:, :self.embedding_dim] = self.tanh(embeddings[:, :self.embedding_dim])
#         embeddings[:, :self.embedding_dim] += normalized_coords
#         embeddings[:, self.embedding_dim:] = self.tanh(embeddings[:, self.embedding_dim:])

#         ppn_inputs = {
#             'ppn_feature_enc': encoder_res["features_enc"],
#             'ppn_feature_dec': [deepest_layer] + features_seg
#         }
#         print(ppn_inputs['ppn_feature_dec'][-1].features.shape)

#         res = {}

#         if self.ghost:
#             ghost_mask = self.linear_ghost(segmentation)
#             res['ghost'] = [ghost_mask.features]
#             ppn_inputs['ghost'] = res['ghost'][0]

#         seediness = self.outputSeediness(seediness)
#         segmentation = self.outputSegmentation(segmentation)

#         res.update({
#             'embeddings': [embeddings[:, :self.embedding_dim]],
#             'margins': [embeddings[:, self.embedding_dim:]],
#             'seediness': [seediness],
#             'features_aggregate': [features_aggregate],
#             'segmentation': [segmentation],
#             'coords': [coords]
#         })

#         ppn_res = self.ppn(ppn_inputs)
#         res.update(ppn_res)
#         print('PPN RES')
#         for key, val in ppn_res.items():
#             print(key, val[0].shape)

#         return res


# class SpatialEmbeddings3(SpatialEmbeddings1):


#     def __init__(self, cfg, name='spatial_embeddings'):
#         super(SpatialEmbeddings3, self).__init__(cfg, name=name)
#         self.embedding_dim = self.model_config.get('embedding_dim', 4)
#         self.coordConv = self.model_config.get('coordConv', True)
#         # Define outputlayers
#         self.outputEmbeddings = scn.Sequential()
#         self._resnet_block(self.outputEmbeddings, self.num_filters, self.embedding_dim)
#         self.outputEmbeddings.add(scn.OutputLayer(self.dimension))
#         # Seediness Output
#         self.outputSeediness = scn.Sequential()
#         self._resnet_block(self.outputSeediness, self.num_filters, self.seedDim)
#         self.outputSeediness.add(scn.OutputLayer(self.dimension))
#         # Margin Output
#         self.outputMargins = scn.Sequential()
#         self._resnet_block(self.outputMargins, self.num_filters, self.sigmaDim)
#         self.outputMargins.add(scn.OutputLayer(self.dimension))


#     def forward(self, input):
#         '''
#         point_cloud is a list of length minibatch size (assumes mbs = 1)
#         point_cloud[0] has 3 spatial coordinates + 1 batch coordinate + 1 feature
#         label has shape (point_cloud.shape[0] + 5*num_labels, 1)
#         label contains segmentation labels for each point + coords of gt points

#         RETURNS:
#             - feature_enc: encoder features at each spatial resolution.
#             - feature_dec: decoder features at each spatial resolution.
#         '''
#         point_cloud, = input
#         coords = point_cloud[:, 0:self.dimension+1].float()
#         normalized_coords = (coords[:, :3] - float(self.spatial_size) / 2) \
#                     / (float(self.spatial_size) / 2)
#         features = point_cloud[:, self.dimension+1:].float()
#         if self.coordConv:
#             features = torch.cat([normalized_coords, features], dim=1)

#         x = self.input((coords, features))
#         encoder_res = self.encoder(x)
#         features_enc = encoder_res['features_enc']
#         deepest_layer = encoder_res['deepest_layer']
#         features_cluster = self.decoder(features_enc, deepest_layer)
#         features_seediness = self.seed_decoder(features_enc, deepest_layer)

#         embeddings = self.outputEmbeddings(features_cluster[-1])
#         margins = self.outputMargins(features_cluster[-1])
#         seediness = self.outputSeediness(features_seediness[-1])

#         res = {
#             "embeddings": [self.tanh(embeddings)],
#             "margins": [self.sigmoid(margins)],
#             "seediness": [self.sigmoid(seediness)]
#         }

#         print(res)
        
#         return res