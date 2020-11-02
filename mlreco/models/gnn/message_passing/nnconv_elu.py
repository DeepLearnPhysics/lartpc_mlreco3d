import numpy as np
import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, ELU, BatchNorm1d
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_std
from torch_geometric.nn import MetaLayer, NNConv
from mlreco.models.gnn.normalizations import BatchNorm, InstanceNorm

class NNConvModel(nn.Module):
    '''
    NNConv GNN Module for extracting node/edge/global features
    '''
    def __init__(self, cfg):
        super(NNConvModel, self).__init__()
        self.model_config = cfg
        self.node_input     = self.model_config.get('node_feats', 16)
        self.edge_input     = self.model_config.get('edge_feats', 19)
        self.global_input   = self.model_config.get('global_feats', 16)
        self.node_output    = self.model_config.get('node_output_feats', 32)
        self.edge_output    = self.model_config.get('edge_output_feats', self.edge_input)
        self.global_output  = self.model_config.get('global_output_feats', 32)
        self.aggr           = self.model_config.get('aggr', 'add')
        self.leakiness      = self.model_config.get('leakiness', 0.1)

        self.edge_mlps = torch.nn.ModuleList()
        self.nnConvs = torch.nn.ModuleList()
        self.edge_updates = torch.nn.ModuleList()

        # perform batch normalization
        self.bn_node = torch.nn.ModuleList()
        self.bn_edge = BatchNorm1d(self.edge_input)

        self.num_mp = self.model_config.get('num_mp', 3)

        node_input  = self.node_input
        node_output = self.node_output
        edge_input  = self.edge_input
        edge_output = self.edge_output
        for i in range(self.num_mp):
            self.edge_mlps.append(
                Seq(
                    BatchNorm1d(edge_input),
                    Lin(edge_input, node_input),
                    ELU(),
                    BatchNorm1d(node_input),
                    Lin(node_input, node_input),
                    ELU(),
                    BatchNorm1d(node_input),
                    Lin(node_input, node_input*node_output)
                )
            )
            self.bn_node.append(BatchNorm(node_input))
            self.nnConvs.append(
                NNConv(node_input, node_output, self.edge_mlps[i], aggr=self.aggr))
            # self.bn_node.append(BatchNorm(node_output))
            # print(node_input, node_output)
            self.edge_updates.append(
                MetaLayer(edge_model=EdgeLayer(node_output, edge_input, edge_output,
                                    leakiness=self.leakiness)#,
                          #node_model=NodeLayer(node_output, node_output, self.edge_input,
                                                #leakiness=self.leakiness)
                          #global_model=GlobalModel(node_output, 1, 32)
                         )
            )
            node_input = node_output
            edge_input = edge_output

        self.node_classes = self.model_config.get('node_classes', 2)
        self.edge_classes = self.model_config.get('edge_classes', 2)

        self.node_predictor = nn.Linear(node_output, self.node_classes)
        self.edge_predictor = nn.Linear(edge_output, self.edge_classes)

    def forward(self, node_features, edge_indices, edge_features, xbatch):
        x = node_features.view(-1, self.node_input)
        e = edge_features.view(-1, self.edge_input)

        for i in range(self.num_mp):
            x = self.bn_node[i](x)
            x = self.nnConvs[i](x, edge_indices, e)
            # x = self.bn_node(x)
            x = F.elu(x)
            # add u and batch arguments for not having error in some old version
            _, e, _ = self.edge_updates[i](x, edge_indices, e, u=None, batch=xbatch)
        # print(edge_indices.shape)
        x_pred = self.node_predictor(x)
        e_pred = self.edge_predictor(e)

        res = {
            'node_pred': [x_pred],
            'edge_pred': [e_pred],
            'node_features': [x],
            'edge_features': [e]
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
            nn.ELU(),
            BatchNorm1d(edge_out),
            nn.Linear(edge_out, edge_out),
            nn.ELU(),
            BatchNorm1d(edge_out),
            nn.Linear(edge_out, edge_out)
        )

    def forward(self, src, dest, edge_attr, u=None, batch=None):
        out = torch.cat([src, dest, edge_attr], dim=1)
        return self.edge_mlp(out)


class NodeLayer(nn.Module):
    '''
    NodeModel for node feature prediction.

    Example: Particle Classification using node-level features.

    INPUTS:

        DEFINITIONS:
            N: number of nodes
            F_x: number of node features
            F_e: number of edge features
            F_u: number of global features
            F_o: number of output node features
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

        - output: [C, F_o] Tensor with F_o output node feature
    '''
    def __init__(self, node_in, node_out, edge_in, leakiness=0.0):
        super(NodeLayer, self).__init__()

        self.node_mlp_1 = nn.Sequential(
            BatchNorm1d(node_in + edge_in),
            nn.Linear(node_in + edge_in, node_out),
            nn.ELU(),
            BatchNorm1d(node_out),
            nn.Linear(node_out, node_out),
            nn.ELU(),
            BatchNorm1d(node_out),
            nn.Linear(node_out, node_out)
        )

        self.node_mlp_2 = nn.Sequential(
            BatchNorm1d(node_in + node_out),
            nn.Linear(node_in + node_out, node_out),
            nn.ELU(),
            BatchNorm1d(node_out),
            nn.Linear(node_out, node_out),
            nn.ELU(),
            BatchNorm1d(node_out),
            nn.Linear(node_out, node_out)
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out], dim=1)
        return self.node_mlp_2(out)


class GlobalModel(nn.Module):
    '''
    Global Model for global feature prediction.

    Example: event classification (graph classification) over the whole image
    within a batch.

    Do Hierarchical Pooling to reduce features

    INPUTS:

        DEFINITIONS:
            N: number of nodes
            F_x: number of node features
            F_e: number of edge features
            F_u: number of global features
            F_o: number of output node features
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

        - output: [C, F_o] Tensor with F_o output node feature
    '''
    def __init__(self, node_in, batch_size, global_out, leakiness=0.0):
        super(GlobalModel, self).__init__()

        self.global_mlp = nn.Sequential(
            BatchNorm1d(node_in + batch_size),
            nn.Linear(node_in + batch_size, global_out),
            nn.ELU(),
            BatchNorm1d(global_out),
            nn.Linear(global_out, global_out),
            nn.ELU(),
            BatchNorm1d(global_out),
            nn.Linear(global_out, global_out)
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        out = torch.cat([u, scatter_mean(x, batch, dim=0)], dim=1)
        return self.global_mlp(out)
