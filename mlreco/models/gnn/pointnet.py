import numpy as np
import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d, LeakyReLU
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_std
from torch_geometric.nn import MetaLayer, PPFConv, XConv, PointConv
from mlreco.models.gnn.normalizations import BatchNorm, InstanceNorm

from .modular_nnconv_old import EdgeLayer

class PointConvModel(nn.Module):
    '''
    PPFConv GNN Module for Graph-SPICE
    '''
    def __init__(self, cfg, name='pointconv'):
        super(PointConvModel, self).__init__()
        self.model_config = cfg
        self.node_input     = self.model_config.get('node_feats', 16)
        self.edge_input     = self.model_config.get('edge_feats', 1)
        self.node_output    = self.model_config.get('node_output_feats', self.node_input)
        self.edge_output    = self.model_config.get('edge_output_feats', self.edge_input)
        self.aggr           = self.model_config.get('aggr', 'add')
        self.leakiness      = self.model_config.get('leakiness', 0.1)

        self.edge_mlps = torch.nn.ModuleList()
        self.messagePassing = torch.nn.ModuleList()
        self.edge_updates = torch.nn.ModuleList()

        # perform batch normalization
        self.bn_node = torch.nn.ModuleList()

        self.num_mp = self.model_config.get('num_mp', 3)

        self.gnn_type = self.model_config.get('gnn_type', 'pointconv')
        gnn_type_dict = {
            'pointconv': PointConv,
            'ppfconv': PPFConv,
            'xconv': XConv
        }
        self.mp = gnn_type_dict[self.gnn_type]

        node_input  = self.node_input
        offset = 0
        if self.gnn_type == 'pointconv':
            offset = 3

        print(node_input)

        node_output = self.node_output
        edge_input  = self.edge_input
        edge_output = self.edge_output

        self.local_nn = torch.nn.ModuleList()
        self.global_nn = torch.nn.ModuleList()

        for i in range(self.num_mp):
            self.local_nn.append(
                Seq(
                    BatchNorm1d(node_input + offset),
                    Lin(node_input + offset, node_output),
                    LeakyReLU(self.leakiness),
                    BatchNorm1d(node_output),
                    Lin(node_output, node_output)
                )
            )
            self.global_nn.append(
                Seq(
                    BatchNorm1d(node_output),
                    Lin(node_output, node_output),
                    LeakyReLU(self.leakiness),
                    BatchNorm1d(node_output),
                    Lin(node_output, node_output)
                )
            )
            self.bn_node.append(BatchNorm(node_input))
            self.messagePassing.append(self.mp(self.local_nn[i], self.global_nn[i]))
            self.edge_updates.append(
                MetaLayer(edge_model=EdgeLayer(node_output, edge_input, edge_output,
                                    leakiness=self.leakiness)#,
                          #node_model=NodeLayer(node_output, node_output, self.edge_input,
                                                #leakiness=self.leakiness)
                          #global_model=GlobalModel(node_output, 1, 32)
                         )
            )
            node_input = node_output

        self.edge_classes = self.model_config.get('edge_classes', 1)

        self.edge_predictor = nn.Linear(edge_output, self.edge_classes)


    def forward(self, node_features, edge_indices, edge_attr, xbatch, pos):

        x = node_features.view(-1, self.node_input)
        e = edge_attr.view(-1, self.edge_input)

        for i in range(self.num_mp):
            x = self.bn_node[i](x)
            x = self.messagePassing[i](x, pos, edge_indices)
            x = F.leaky_relu(x, negative_slope=self.leakiness)
            _, e, _ = self.edge_updates[i](x, edge_indices, e, u=None, batch=xbatch)
        e_pred = self.edge_predictor(e)
        
        res = {
            'node_pred': [x],
            'edge_pred': [e_pred]
            }


        return res