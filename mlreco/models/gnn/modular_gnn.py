import numpy as np
import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d, LeakyReLU
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_std
from torch_geometric.nn import MetaLayer, NNConv

from mlreco.models.gnn.node_pred import NodeModel
from mlreco.models.gnn.edge_pred import EdgeModel
from mlreco.models.gnn.normalizations import BatchNorm, InstanceNorm

class FullGNN(nn.Module):
    '''
    Full GNN Module for extracting node/edge/global features
    '''
    def __init__(self, cfg, name='full_gnn'):
        super(FullGNN, self).__init__()
        self.model_config = cfg[name]
        self.nodeInput = self.model_config.get('node_features', 16)
        self.nodeOutput = self.model_config.get('node_output_features', 32)
        self.edgeInput = self.model_config.get('edge_features', 16)
        self.aggr = self.model_config.get('aggr', 'add')
        self.leakiness = self.model_config.get('leakiness', 0.0)

        self.edge_mlps = torch.nn.ModuleList()
        self.nnConvs = torch.nn.ModuleList()
        self.edge_updates = torch.nn.ModuleList()
        self.edge_features = self.model_config.get('edge_features', 16)

        # perform batch normalization
        self.bn_node = torch.nn.ModuleList()
        self.bn_edge = BatchNorm1d(self.edgeInput)

        self.num_mp = self.model_config.get('num_mp', 3)

        nInput = self.nodeInput
        nOutput = self.nodeOutput
        for i in range(self.num_mp):
            self.edge_mlps.append(
                Seq(
                    BatchNorm1d(self.edge_features),
                    Lin(self.edge_features, nInput),
                    LeakyReLU(self.leakiness),
                    BatchNorm1d(nInput),
                    Lin(nInput, nInput),
                    LeakyReLU(self.leakiness),
                    BatchNorm1d(nInput),
                    Lin(nInput, nInput*nOutput)
                )
            )
            self.bn_node.append(BatchNorm(nInput))
            self.nnConvs.append(
                NNConv(nInput, nOutput, self.edge_mlps[i], aggr=self.aggr))
            # self.bn_node.append(BatchNorm(nOutput))
            # print(nInput, nOutput)
            self.edge_updates.append(
                MetaLayer(EdgeLayer(nOutput, self.edge_features, self.edge_features,
                                    leakiness=self.leakiness))
            )
            nInput = nOutput

        self.nodeClasses = self.model_config.get('node_classes', 4)
        self.edgeClasses = self.model_config.get('edge_classes', 2)

        self.node_predictor = Seq(
            BatchNorm1d(nOutput),
            nn.Linear(nOutput, self.nodeClasses))
        self.edge_predictor = nn.Linear(self.edge_features, self.edgeClasses)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, node_features, edge_indices, edge_features, xbatch):
        x = node_features.view(-1, self.nodeInput)
        e = edge_features.view(-1, self.edgeInput)

        for i in range(self.num_mp):
            x = self.bn_node[i](x)
            x = self.nnConvs[i](x, edge_indices, e)
            # x = self.bn_node(x)
            x = F.leaky_relu(x, negative_slope=self.leakiness)
            _, e, _ = self.edge_updates[i](x, edge_indices, e)
        # print(edge_indices.shape)
        x_pred = self.node_predictor(x)
        x_pred[:, :3] = self.tanh(x_pred[:, :3])
        x_pred[:, 3:] = 2 * self.sigmoid(x_pred[:, 3:])
        e_pred = self.edge_predictor(e)

        res = {
            'node_predictions': [x_pred],
            'edge_predictions': [e_pred]
            }

        return res


class EdgeLayer(nn.Module):
    '''
    An EdgeModel for predicting edge features.

    Example: Parent-Child Edge prediction and EM primary assignment prediction.

    INPUTS:

        DEFINITIONS:
            E: number of edges
            F_x: number of node features
            F_e: number of edge features
            F_u: number of global features
            F_o: number of output edge features
            B: number of graphs (same as batch size)

        If an entry i->j is an edge, then we have source node feature
        F^i_x, target node feature F^j_x, and edge features F_e.

        - source: [E, F_x] Tensor, where E is the number of edges

        - target: [E, F_x] Tensor, where E is the number of edges

        - edge_attr: [E, F_e] Tensor, indicating input edge features.

        - global_features: [B, F_u] Tensor, where B is the number of graphs
        (equivalent to number of batches).

        - batch: [E] Tensor containing batch indices for each edge from 0 to B-1.

    RETURNS:

        - output: [E, F_o] Tensor with F_o output edge features.
    '''
    def __init__(self, node_in, edge_in, edge_out, leakiness=0.0):
        super(EdgeLayer, self).__init__()
        # TODO: Construct Edge MLP
        self.edge_mlp = nn.Sequential(
            BatchNorm1d(2 * node_in + edge_in),
            nn.Linear(2 * node_in + edge_in, edge_out),
            nn.LeakyReLU(negative_slope=leakiness),
            BatchNorm1d(edge_out),
            nn.Linear(edge_out, edge_out),
            nn.LeakyReLU(negative_slope=leakiness),
            BatchNorm1d(edge_out),
            nn.Linear(edge_out, edge_out)
        )

    def forward(self, src, dest, edge_attr, u=None, batch=None):
        out = torch.cat([src, dest, edge_attr], dim=1)
        return self.edge_mlp(out)


# class NodeModel(nn.Module):
#     '''
#     NodeModel for node feature prediction.

#     Example: Particle Classification using node-level features.
#     '''
#     def __init__(self, node_in, node_out, edge_in, leakiness=0.0):
#         super(NodeModel, self).__init__()


# class GlobalModel(nn.Module):
#     '''
#     Global Model for global feature prediction.

#     Example: event classification (graph classification) over the whole image
#     within a batch.

#     Do Hierarchical Pooling to reduce features
#     '''
#     pass
