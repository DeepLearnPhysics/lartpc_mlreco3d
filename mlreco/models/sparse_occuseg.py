import torch
import torch.nn as nn
import numpy as np
import sparseconvnet as scn
from collections import defaultdict

from mlreco.models.layers.uresnet import UResNet


class SparseOccuSeg(UResNet):


    def __init__(self, cfg, name='sparse_occuseg'):
        super(SparseOccuSeg, self).__init__(cfg, name='uresnet')
        if 'modules' in cfg:
            self.model_config = cfg['modules'][name]
        else:
            self.model_config = cfg[name]
        self.feature_embedding_dim = self.model_config.get('feature_embedding_dim', 8)
        self.spatial_embedding_dim = self.model_config.get('spatial_embedding_dim', 3)
        self.num_classses = self.model_config.get('num_classes', 5)
        self.coordConv = self.model_config.get('coordConv', True)

        self.covariance_mode = self.model_config.get('covariance_mode', 'exp')

        if self.covariance_mode == 'exp':
            self.cov_func = torch.exp
        elif self.covariance_mode == 'softplus':
            self.cov_func = nn.Softplus()
        else:
            self.cov_func = nn.Sigmoid()

        self.occupancy_mode = self.model_config.get('occupancy_mode', 'exp')

        if self.occupancy_mode == 'exp':
            self.occ_func = torch.exp
        else self.occupancy_mode == 'softplus':
            self.occ_func = nn.Softplus()

        # Define outputlayers
        self.outputSpatialEmbeddings = scn.Sequential()
        self.outputSpatialEmbeddings.add(
            scn.NetworkInNetwork(self.num_filters, 
                                 self.spatial_embedding_dim, 
                                 self.allow_bias))
        self.outputSpatialEmbeddings.add(scn.OutputLayer(self.dimension))

        self.outputFeatureEmbeddings = scn.Sequential()
        self.outputFeatureEmbeddings.add(
            scn.NetworkInNetwork(self.num_filters, 
                                 self.feature_embedding_dim, 
                                 self.allow_bias))
        self.outputFeatureEmbeddings.add(scn.OutputLayer(self.dimension))

        self.outputSegmentation = scn.Sequential()
        self.outputSegmentation.add(
            scn.NetworkInNetwork(self.num_filters, 
                                 self.num_classses, 
                                 self.allow_bias))
        self.outputSegmentation.add(scn.OutputLayer(self.dimension))

        self.outputCovariance = scn.Sequential()
        self.outputCovariance.add(
            scn.NetworkInNetwork(self.num_filters, 
                                 2, 
                                 self.allow_bias))
        self.outputCovariance.add(scn.OutputLayer(self.dimension))

        self.outputOccupancy = scn.Sequential()
        self.outputOccupancy.add(
            scn.NetworkInNetwork(self.num_filters, 
                                 1, 
                                 self.allow_bias))
        self.outputOccupancy.add(scn.OutputLayer(self.dimension))

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

        output_features = features_cluster[-1]

        # Spatial Embeddings
        spatial_embeddings = self.outputSpatialEmbeddings(output_features)
        spatial_embeddings = self.tanh(spatial_embeddings)
        spatial_embeddings += normalized_coords

        # Covariance
        cov = self.outputCovariance(output_features)
        covariance = self.cov_func(cov)

        # Feature Embeddings
        feature_embeddings = self.outputFeatureEmbeddings(output_features)

        # Occupancy
        occ = self.outputOccupancy(output_features)
        occupancy = self.occ_func(occ)

        # Segmentation
        segmentation = self.outputSegmentation(output_features)

        res = {
            "spatial_embeddings": [spatial_embeddings],
            "covariance": [covariance],
            "feature_embeddings": [feature_embeddings], 
            "occupancy": [occupancy],
            "segmentation": [segmentation]
        }

        return res


class SparseOccuSegLoss(torch.nn.modules.loss._Loss):

    def __init__(self, cfg, name='occuseg_loss'):
        super(SparseOccuSegLoss, self).__init__()


    def forward(self, result, label):

        segment_label = [cluster_label[0][:, [0, 1, 2, 3, -1]]]
        group_label = [cluster_label[0][:, [0, 1, 2, 3, 5]]]