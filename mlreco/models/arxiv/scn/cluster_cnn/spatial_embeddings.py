import torch
import torch.nn as nn
import sparseconvnet as scn

from mlreco.models.scn.layers.uresnet import UResNet
from mlreco.models.scn.layers.ppn import PPN, PPNLoss
from .utils import add_normalized_coordinates, distance_matrix


class CoordinateEmbeddings(UResNet):
    def __init__(self, cfg, name='coordinate_embeddings'):
        super(CoordinateEmbeddings, self).__init__(cfg, name='uresnet')
        if 'modules' in cfg:
            self.model_config = cfg['modules'][name]
        else:
            self.model_config = cfg[name]
        self.seedDim = self.model_config.get('seediness_dim', 1)
        self.embeddingDim = self.model_config.get('embedding_dim', 3)
        self.coordConv = self.model_config.get('coordConv', True)

        # Define outputlayers
        self.outputEmbeddings = scn.Sequential()
        self._nin_block(self.outputEmbeddings, self.num_filters, self.embeddingDim)
        self.outputEmbeddings.add(scn.OutputLayer(self.dimension))

        # Pytorch Activations
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()


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
        # print("Point Cloud: ", point_cloud)
        coords = point_cloud[:, 0:self.dimension+1].float()
        features = point_cloud[:, self.dimension+1:].float()
        features = features[:, -1].view(-1, 1)

        normalized_coords = (coords[:, :3] - float(self.spatial_size) / 2) \
                    / (float(self.spatial_size) / 2)
        if self.coordConv:
            features = torch.cat([normalized_coords, features], dim=1)

        x = self.input((coords, features))
        encoder_res = self.encoder(x)
        features_enc = encoder_res['features_enc']
        deepest_layer = encoder_res['deepest_layer']
        features_cluster = self.decoder(features_enc, deepest_layer)

        embeddings = self.outputEmbeddings(features_cluster[-1])
        # embeddings[:, :self.dimension] += coords[:, :3]

        res = {
            "embeddings": [embeddings[:, :self.dimension]]
        }

        return res


class SpatialEmbeddings(UResNet):

    def __init__(self, cfg, name='spatial_embeddings'):
        super(SpatialEmbeddings, self).__init__(cfg, name='uresnet')
        if 'modules' in cfg:
            self.model_config = cfg['modules'][name]
        else:
            self.model_config = cfg[name]
        self.seedDim = self.model_config.get('seediness_dim', 1)
        self.coordConv = self.model_config.get('coordConv', False)
        self.sigmaDim = self.model_config.get('sigma_dim', 1)
        self.seed_freeze = self.model_config.get('seed_freeze', False)
        self.coordConv = self.model_config.get('coordConv', True)
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

        print(self.outputEmbeddings)


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
        # print("Point Cloud: ", point_cloud)
        coords = point_cloud[:, 0:self.dimension+1].float()
        features = point_cloud[:, self.dimension+1:].float()
        features = features[:, -1].view(-1, 1)

        normalized_coords = (coords[:, :3] - float(self.spatial_size) / 2) \
                    / (float(self.spatial_size) / 2)
        if self.coordConv:
            features = torch.cat([normalized_coords, features], dim=1)

        x = self.input((coords, features))
        encoder_res = self.encoder(x)
        features_enc = encoder_res['features_enc']
        deepest_layer = encoder_res['deepest_layer']
        features_cluster = self.decoder(features_enc, deepest_layer)
        features_seediness = self.seed_decoder(features_enc, deepest_layer)

        embeddings = self.outputEmbeddings(features_cluster[-1])
        embeddings[:, :self.dimension] = self.tanh(embeddings[:, :self.dimension])
        embeddings[:, :self.dimension] += normalized_coords
        seediness = self.outputSeediness(features_seediness[-1])

        res = {
            "embeddings": [embeddings[:, :self.dimension]],
            "margins": [2 * self.sigmoid(embeddings[:, self.dimension:])],
            "seediness": [self.sigmoid(seediness)]
        }

        return res
