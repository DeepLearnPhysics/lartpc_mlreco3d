import torch
import torch.nn as nn
import MinkowskiEngine as ME

from mlreco.models.layers.common.uresnet_layers import UResNet
# from mlreco.models.layers.cluster_cnn.losses.gs_embeddings import *

from pprint import pprint


class GraphSPICEEmbedder(UResNet):

    MODULES = ['network_base', 'uresnet', 'graph_spice_embedder']

    RETURNS = {
        'spatial_embeddings': ['tensor', 'coordinates'],
        'covariance': ['tensor', 'coordinates'],
        'feature_embeddings': ['tensor', 'coordinates'],
        'occupancy': ['tensor', 'coordinates'],
        'features': ['tensor', 'coordinates'],
        'hypergraph_features': ['tensor', 'coordinates'],
        'segmentation': ['tensor', 'coordinates']
    }

    def __init__(self, cfg, name='graph_spice_embedder'):
        super(GraphSPICEEmbedder, self).__init__(cfg)
        self.model_config = cfg.get(name, {})
        self.feature_embedding_dim = self.model_config.get(
            'feature_embedding_dim', 8)
        self.spatial_embedding_dim = self.model_config.get(
            'spatial_embedding_dim', 3)
        self.num_classes = self.model_config.get('num_classes', 5)
        self.coordConv = self.model_config.get('coordConv', True)
        self.segmentationLayer = self.model_config.get('segmentationLayer', False)

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

        self.outputSpatialEmbeddings = nn.Linear(self.num_filters,
                                                 self.spatial_embedding_dim)

        self.outputFeatureEmbeddings = nn.Linear(self.num_filters,
                                                 self.feature_embedding_dim)

        if self.segmentationLayer:
            self.outputSegmentation = nn.Linear(self.num_filters,
                                               self.num_classes)

        self.outputCovariance = nn.Linear(self.num_filters, 2)

        self.outputOccupancy = nn.Linear(self.num_filters, 1)

        self.hyper_dimension = self.spatial_embedding_dim \
                             + self.feature_embedding_dim + 3

        # Pytorch Activations
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        # print('Total Number of Trainable Parameters (graph_spice_embedder)= {}'.format(
        #             sum(p.numel() for p in self.parameters() if p.requires_grad)))
        # print([name for name, param in self.named_parameters()])

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
        coords = point_cloud[:, 0:self.D+1].int()
        features = point_cloud[:, self.D+1:].float()

        normalized_coords = (coords[:, 1:self.D+1] - float(self.spatial_size) / 2) \
                    / (float(self.spatial_size) / 2)
        if self.coordConv:
            features = torch.cat([normalized_coords, features], dim=1)

        x = ME.SparseTensor(features, coordinates=coords)

        encoder_res = self.encoder(x)
        encoderTensors = encoder_res['encoderTensors']
        finalTensor = encoder_res['finalTensor']
        decoderTensors = self.decoder(finalTensor, encoderTensors)

        output_features = decoderTensors[-1].F

        # Spatial Embeddings
        out = self.outputSpatialEmbeddings(output_features)
        spatial_embeddings = self.tanh(out)

        # Covariance
        out = self.outputCovariance(output_features)
        covariance = self.cov_func(out)

        # Feature Embeddings
        feature_embeddings = self.outputFeatureEmbeddings(output_features)

        # Occupancy
        out = self.outputOccupancy(output_features)
        occupancy = self.occ_func(out)

        # Segmentation
        if self.segmentationLayer:
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
            "features": [output_features],
            "hypergraph_features": [hypergraph_features],
        }
        if self.segmentationLayer:
            res["segmentation"] = [segmentation]

        return res

    def forward(self, input):
        '''
        Train time forward
        '''
        point_cloud, = input
        out = self.get_embeddings([point_cloud])

        return out
