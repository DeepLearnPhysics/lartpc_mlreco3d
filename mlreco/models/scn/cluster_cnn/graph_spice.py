import torch
import torch.nn as nn
import sparseconvnet as scn

from mlreco.models.scn.layers.uresnet import UResNet
from mlreco.models.cluster_cnn.losses.gs_embeddings import *

from pprint import pprint


class GraphSPICEEmbedder(UResNet):

    MODULES = ['network_base', 'uresnet', 'graph_spice_embedder']

    def __init__(self, cfg, name='graph_spice_embedder'):
        print('-----------------GraphSPICEEmbedder-------------------')
        pprint(cfg)
        super(GraphSPICEEmbedder, self).__init__(cfg, name='uresnet')
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
        elif self.occupancy_mode == 'softplus':
            self.occ_func = nn.Softplus()
        else:
            self.occ_func = torch.exp

        # Define outputlayers
        self.outputLayer = scn.OutputLayer(self.dimension)

        self.outputSpatialEmbeddings = nn.Linear(self.num_filters,
                                           self.spatial_embedding_dim)

        self.outputFeatureEmbeddings = nn.Linear(self.num_filters,
                                           self.feature_embedding_dim)

        self.outputSegmentation = nn.Linear(self.num_filters,
                                            self.num_classses)


        self.outputCovariance = nn.Linear(self.num_filters, 2)

        self.outputOccupancy = nn.Linear(self.num_filters, 1)

        self.hyper_dimension = self.spatial_embedding_dim + \
                               self.feature_embedding_dim + \
                               3

        # Pytorch Activations
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()


    def get_embeddings(self, input):
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

        output_features = self.outputLayer(features_cluster[-1])

        # Spatial Embeddings
        out = self.outputSpatialEmbeddings(output_features)
        spatial_embeddings = self.tanh(out)
        # spatial_embeddings += normalized_coords

        # Covariance
        out = self.outputCovariance(output_features)
        covariance = self.cov_func(out)

        # Feature Embeddings
        feature_embeddings = self.outputFeatureEmbeddings(output_features)

        # Occupancy
        out = self.outputOccupancy(output_features)
        occupancy = self.occ_func(out)

        # Segmentation
        segmentation = self.outputSegmentation(output_features)

        hypergraph_features = torch.cat([
            spatial_embeddings,
            feature_embeddings,
            covariance,
            occupancy], dim=1)

        res = {
            "spatial_embeddings": [spatial_embeddings + normalized_coords],
            "covariance": [covariance],
            "feature_embeddings": [feature_embeddings],
            "occupancy": [occupancy],
            "segmentation": [segmentation],
            "features": [output_features],
            "hypergraph_features": [hypergraph_features]
        }

        # for key, val in res.items():
        #     print((val[0] != val[0]).any())

        return res

    def forward(self, input):
        '''
        Train time forward
        '''
        point_cloud, = input
        out = self.get_embeddings([point_cloud])

        return out


class GraphSPICEGeoEmbedder(GraphSPICEEmbedder):

    MODULES = ['network_base', 'uresnet', 'graph_spice_embedder']

    def __init__(self, cfg, name='graph_spice_embedder'):
        super(GraphSPICEGeoEmbedder, self).__init__(cfg)

        self.outputPrincipalAxis = nn.Linear(self.num_filters, 4)

        self.outputCovariance = nn.Linear(self.num_filters, 6)

        self.hyper_dimension = self.spatial_embedding_dim + \
                               self.feature_embedding_dim + \
                               3

        # Pytorch Activations
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()


    def get_embeddings(self, input):
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

        output_features = self.outputLayer(features_cluster[-1])

        # Spatial Embeddings
        out = self.outputSpatialEmbeddings(output_features)
        spatial_embeddings = self.tanh(out)
        # spatial_embeddings += normalized_coords

        # Covariance
        out = self.outputCovariance(output_features)
        covariance = self.cov_func(out)

        # Feature Embeddings
        feature_embeddings = self.outputFeatureEmbeddings(output_features)

        # Segmentation
        segmentation = self.outputSegmentation(output_features)

        # Occupancy
        out = self.outputOccupancy(output_features)
        occupancy = self.occ_func(out)

        print(segmentation.shape)
        print(occupancy.shape)

        # Tangent Vectors
        trig = self.tanh(self.outputPrincipalAxis(output_features))

        trig_phi = trig[:, :2] / (torch.norm(
            trig[:, :2], dim=1, keepdim=True) + 1e-6)
        trig_theta = trig[:, 2:4] / (torch.norm(
            trig[:, 2:4], dim=1, keepdim=True) + 1e-6)

        tangent = torch.cat(
            [trig_phi[:, 0].view(-1, 1) * trig_theta[:, 1].view(-1, 1),
             trig_phi[:, 1].view(-1, 1) * trig_theta[:, 1].view(-1, 1),
             trig_theta[:, 0].view(-1, 1)], dim=1)

        hypergraph_features = torch.cat([
            spatial_embeddings,
            feature_embeddings,
            covariance,
            occupancy.view(-1, 1)], dim=1)

        res = {
            "spatial_embeddings": [spatial_embeddings + normalized_coords],
            "covariance": [covariance],
            "feature_embeddings": [feature_embeddings],
            "segmentation": [segmentation],
            "occupancy": [occupancy.view(-1, 1)],
            "features": [output_features],
            "tangents": [tangent],
            "hypergraph_features": [hypergraph_features]
        }

        # for key, val in res.items():
        #     print((val[0] != val[0]).any())

        return res


    def forward(self, input):
        '''
        Train time forward
        '''
        point_cloud, = input
        out = self.get_embeddings([point_cloud])

        return out
