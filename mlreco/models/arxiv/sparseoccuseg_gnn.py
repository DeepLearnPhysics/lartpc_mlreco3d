import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sparseconvnet as scn
from collections import defaultdict

import time

from .sparse_occuseg import SparseOccuSeg, SparseOccuSegLoss
from mlreco.models.layers.gnn.message_passing.nnconv import NNConvModel
from mlreco.utils.cluster.cluster_graph_constructor import *
from pprint import pprint

class WeightedEdgeLoss(nn.Module):

    def __init__(self, reduction='none'):
        super(WeightedEdgeLoss, self).__init__()
        self.reduction = reduction
        self.loss_fn = F.binary_cross_entropy_with_logits

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


class SparseOccuSegGNN(nn.Module):

    MODULES = ['network_base', 'uresnet', 'spice_loss', 'sparse_occuseg', 'predictor_cfg', 'constructor', 'modular_nnconv']

    def __init__(self, cfg, name='sparse_occuseg_gnn'):
        super(SparseOccuSegGNN, self).__init__()
        self.sparse_occuseg = SparseOccuSeg(cfg)

        # Define Message Passing
        self.gnn = edge_model_construct(cfg)
        print(self.gnn)
        # self.gnn = NNConvModel(cfg['modular_nnconv'])
        self.predictor = OccuSegPredictor(cfg['predictor_cfg'])
        self.constructor = GraphDataConstructor(self.predictor, cfg['constructor_cfg'])

    def _forward(self, input):
        '''
        Train-time forward
        '''
        input, labels = input
        coordinates = input[:, :3]
        batch_indices = input[:, 3].int()
        out = self.sparse_occuseg([input])
        out['coordinates'] = [coordinates]
        out['batch_indices'] = [batch_indices]
        graph_data = self.constructor.construct_batched_graphs_with_labels(out, labels)
        gnn_out = self.gnn(graph_data.x,
                           graph_data.edge_index,
                           graph_data.edge_attr,
                           graph_data.batch)
        out['node_pred'] = gnn_out['node_pred']
        out['edge_pred'] = gnn_out['edge_pred']
        out['edge_truth'] = [graph_data.edge_truth]
        out['graph'] = [graph_data]
        return out

    def forward(self, input):
        if self.training:
            out = self._forward(input)
        else:
            # TODO
            assert len(input) == 1
            coordinates = input[0][:, :3]
            batch_indices = input[0][:, 3].int()
            out = self.sparse_occuseg(input)
            out['coordinates'] = [coordinates]
            out['batch_indices'] = [batch_indices]
            graph_data = self.constructor.construct_batched_graphs(out)
            gnn_out = self.gnn(graph_data.x,
                               graph_data.edge_index,
                               graph_data.edge_attr,
                               graph_data.batch)
            out['node_pred'] = gnn_out['node_pred']
            out['edge_pred'] = gnn_out['edge_pred']
            out['graph'] = [graph_data]
        return out


class GraphSPICEGNN(nn.Module):

    MODULES = ['network_base', 'uresnet', 'spice_loss', 'sparse_occuseg', 'predictor_cfg', 'constructor', 'modular_nnconv']

    def __init__(self, cfg, name='sparse_occuseg_gnn'):
        super(GraphSPICEGNN, self).__init__()
        self.sparse_occuseg = SparseOccuSeg(cfg)

        # Define Message Passing
        self.gnn = edge_model_construct(cfg)
        # self.gnn = NNConvModel(cfg['modular_nnconv'])
        self.predictor = OccuSegPredictor(cfg['predictor_cfg'])
        self.constructor = GraphDataConstructor(self.predictor, cfg['constructor_cfg'])

    def _forward(self, input):
        '''
        Train-time forward
        '''
        input, labels = input
        coordinates = input[:, :3]
        batch_indices = input[:, 3].int()
        out = self.sparse_occuseg([input])
        out['coordinates'] = [coordinates.float()]
        out['batch_indices'] = [batch_indices]
        graph_data = self.constructor.construct_batched_graphs_with_labels(out, labels)
        gnn_out = self.gnn(graph_data.x,
                           graph_data.edge_index,
                           graph_data.edge_attr,
                           graph_data.batch,
                           graph_data.pos)
        out['node_pred'] = gnn_out['node_pred']
        out['edge_pred'] = gnn_out['edge_pred']
        out['edge_truth'] = [graph_data.edge_truth]
        out['graph'] = [graph_data]
        return out

    def forward(self, input):
        if self.training:
            out = self._forward(input)
        else:
            # TODO
            print(input)
            assert len(input) == 1
            coordinates = input[0][:, :3]
            batch_indices = input[0][:, 3].int()
            out = self.sparse_occuseg(input)
            out['coordinates'] = [coordinates.float()]
            out['batch_indices'] = [batch_indices]
            graph_data = self.constructor.construct_batched_graphs(out)
            gnn_out = self.gnn(graph_data.x,
                               graph_data.edge_index,
                               graph_data.edge_attr,
                               graph_data.batch,
                               graph_data.pos)
            out['node_pred'] = gnn_out['node_pred']
            out['edge_pred'] = gnn_out['edge_pred']
            out['graph'] = [graph_data]
        return out


class SparseOccuSegGNNLoss(nn.Module):

    def __init__(self, cfg, name='occuseg_gnn_loss'):
        super(SparseOccuSegGNNLoss, self).__init__()
        self.loss_fn = SparseOccuSegLoss(cfg)
        self.edge_loss = WeightedEdgeLoss()
        self.eval = cfg['occuseg_gnn_loss'].get('eval', False)

    def forward(self, result, label):
        res = self.loss_fn(result, label)
        # Add GNN Loss
        node_pred = result['node_pred'][0]
        edge_pred = result['edge_pred'][0]
        if not self.eval:
            edge_truth = result['edge_truth'][0]
            edge_loss = self.edge_loss(edge_pred.squeeze(), edge_truth.float())
            edge_loss = edge_loss.mean()
            with torch.no_grad():
                true_negatives = float(torch.sum(( (edge_pred.squeeze() < 0) & ~edge_truth.bool() ).int()))
                precision = true_negatives / (float(torch.sum( (edge_truth == 0).int() )) + 1e-6)
                recall = true_negatives / (float(torch.sum( (edge_pred < 0).int() )) + 1e-6)
                f1 = precision * recall / (precision + recall + 1e-6)

            res['edge_accuracy'] = f1
            res['loss'] += edge_loss
            res['edge_loss'] = float(edge_loss)
        else:
            return res

        edge_pred = edge_pred.squeeze()
        return res
