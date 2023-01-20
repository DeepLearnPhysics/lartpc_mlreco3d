import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sparseconvnet as scn
from collections import defaultdict

from mlreco.models.scn.layers.uresnet import UResNet
from mlreco.models.layers.cluster_cnn.losses.occuseg import OccuSegLoss

from pprint import pprint

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

        # Pytorch Activations
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.predictor = OccuSegPredictor(cfg['predictor_cfg'])
        self.constructor = GraphDataConstructor(self.predictor, cfg['constructor_cfg'])

    def forward(self, input):

        res = self._forward(input)
        if not self.training:
            graph_data = self.constructor.construct_batched_graphs(res)
            res['graph'] = [graph_data]
        return res


    def _forward(self, input):
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
        else:
            features = normalized_coords

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

        res = {
            "spatial_embeddings": [spatial_embeddings + normalized_coords],
            "covariance": [covariance],
            "feature_embeddings": [feature_embeddings],
            "occupancy": [occupancy],
            "segmentation": [segmentation],
            "features": [output_features],
            "coordinates": [coords[:, :3]],
            "batch_indices": [coords[:, 3].long()]
        }


        return res


class WeightedEdgeLoss(nn.Module):

    def __init__(self, reduction='none'):
        super(WeightedEdgeLoss, self).__init__()
        self.reduction = reduction
        self.loss_fn = F.binary_cross_entropy

    def forward(self, x, y):
        device = x.device
        weight = torch.ones(y.shape[0]).to(device)
        with torch.no_grad():
            num_pos = torch.sum(y).item()
            num_edges = y.shape[0]
            w = 1.0 / (1.0 - float(num_pos) / num_edges)
            weight[~y.bool()] = w
        loss = self.loss_fn(x, y, weight=weight, reduction=self.reduction)
        return loss


class SparseOccuSegLoss(torch.nn.modules.loss._Loss):

    def __init__(self, cfg, name='occuseg_loss'):
        super(SparseOccuSegLoss, self).__init__()
        self.loss_fn = OccuSegLoss(cfg)

    def forward(self, result, label):

        segment_label = [label[0][:, [0, 1, 2, 3, -1]]]
        group_label = [label[0][:, [0, 1, 2, 3, 5]]]

        res = self.loss_fn(result, segment_label, group_label)
        return res


class SparseOccuSegEdgeLoss(SparseOccuSegLoss):

    def __init__(self, cfg, name='occuseg_loss'):
        super(SparseOccuSegEdgeLoss, self).__init__(cfg)
        self.edge_loss = WeightedEdgeLoss()
        self.loss_config = cfg['occuseg_loss']
        self.mode = self.loss_config.get('mode', 'knn')
        if self.mode == 'knn':
            self.graph_param = self.loss_config.get('args', dict(k=6))
            self.graph_gen = partial(knn_graph, **self.graph_param)
        elif self.mode == 'radius':
            self.graph_param = self.loss_config.get('args', dict(r=2.0))
            self.graph_gen = partial(radius_graph, **self.graph_param)
        else:
            raise NotImplementedError

    @staticmethod
    def get_edge_truth(edge_indices, labels):
        '''

            - edge_indices: 2 x E
            - labels: N
        '''
        with torch.no_grad():
            u = labels[edge_indices[0, :]]
            v = labels[edge_indices[1, :]]
            return (u == v).bool()


    def forward(self, result, label):

        segment_label = [label[0][:, [0, 1, 2, 3, -1]]]
        group_label = [label[0][:, [0, 1, 2, 3, 5]]]

        coords = result['coordinates'][0]
        batch_indices = result['batch_indices'][0]

        edge_indices = self.graph_gen(coords, batch=batch_indices)
        res = self.loss_fn(result, segment_label, group_label)

        probs = get_edge_weight(
            result['spatial_embeddings'][0],
            result['feature_embeddings'][0],
            result['covariance'][0],
            edge_indices,
            occ=result['occupancy'][0].squeeze())

        edge_truth = self.get_edge_truth(edge_indices, group_label[0][:, -1])

        # for i in range(20):
        #     print("{0:.4f}, {1}".format(probs[i], edge_truth[i]))
        edge_loss = self.edge_loss(probs, edge_truth.float()).mean()
        with torch.no_grad():
            edge_pred = probs < 0.5
            tp = float(torch.sum(edge_truth & edge_pred))
            tn = float(torch.sum(~edge_truth & ~edge_pred))
            fp = float(torch.sum(~edge_truth & edge_pred))
            fn = float(torch.sum(edge_truth & ~edge_pred))

            precision = (tp + 1e-6) / (tp + fp + 1e-6)
            recall = (tp + 1e-6) / (tp + fn + 1e-6)
            tnr = (tn + 1e-6) / (tn + fp + 1e-6)
            npv = (tn + 1e-6) / (tn + fn + 1e-6)

            f1_pos = 2.0 * precision * recall / (precision + recall + 1e-6)
            f1_neg = 2.0 * tnr * npv / (tnr + npv + 1e-6)

            res['precision'] = precision
            res['recall'] = recall
            res['tnr'] = tnr
            res['npv'] = npv
            res['f1_pos'] = f1_pos
            res['f1_neg'] = f1_neg
        res['edge_loss'] = float(edge_loss)
        res['loss'] += edge_loss
        return res
