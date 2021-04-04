import torch
import torch.nn as nn
import numpy as np
import sparseconvnet as scn
from collections import defaultdict


from mlreco.models.layers.uresnet import UResNet
from mlreco.models.cluster_cnn.losses.occuseg import OccuSegLoss, WeightedEdgeLoss
from mlreco.models.arxiv.cluster import ParametricGDC


class SparseOccuSeg(UResNet):

    MODULES = ['network_base', 'uresnet', 'spice_loss', 'sparse_occuseg']

    def __init__(self, cfg, name='sparse_occuseg'):
        super(SparseOccuSeg, self).__init__(cfg, name='uresnet')
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

        self.constructor = ParametricGDC(cfg['constructor_cfg'])


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
            # "features": [output_features],
            "hypergraph_features": [hypergraph_features]
        }

        # for key, val in res.items():
        #     print((val[0] != val[0]).any())

        return res

    def forward(self, input):
        '''
        Train time forward
        '''
        point_cloud, labels = input
        coordinates = point_cloud[:, :3]
        batch_indices = point_cloud[:, 3].int()
        out = self.get_embeddings([point_cloud])
        out['coordinates'] = [coordinates]
        out['batch_indices'] = [batch_indices]
        graph_data = self.constructor(out, labels)
        out['edge_score'] = [graph_data.edge_attr]
        out['edge_index'] = [graph_data.edge_index]
        out['graph'] = [graph_data]
        if self.training:
            out['edge_truth'] = [graph_data.edge_truth]

        return out


class SparseOccuSegLoss(torch.nn.modules.loss._Loss):

    def __init__(self, cfg, name='graph_spice_loss'):
        super(SparseOccuSegLoss, self).__init__()
        # print("CFG + ", cfg)
        self.loss_config = cfg[name]
        self.loss_fn = OccuSegLoss(cfg)
        self.edge_loss = WeightedEdgeLoss()
        self.is_eval = cfg['eval']

    def forward(self, result, segment_label, cluster_label):

        group_label = [cluster_label[0][:, [0, 1, 2, 3, 5]]]

        res = self.loss_fn(result, segment_label, group_label)
        # print(result)
        edge_score = result['edge_score'][0].squeeze()
        if not self.is_eval:
            edge_truth = result['edge_truth'][0]
            edge_loss = self.edge_loss(edge_score.squeeze(), edge_truth.float())
            edge_loss = edge_loss.mean()
            with torch.no_grad():
                true_negatives = float(torch.sum(( (edge_score < 0) & ~edge_truth.bool() ).int()))
                precision = true_negatives / (float(torch.sum( (edge_truth == 0).int() )) + 1e-6)
                recall = true_negatives / (float(torch.sum( (edge_score < 0).int() )) + 1e-6)
                f1 = precision * recall / (precision + recall + 1e-6)

            res['edge_accuracy'] = f1
        else:
            edge_loss = 0

        res['loss'] += edge_loss
        res['edge_loss'] = float(edge_loss)
        return res
