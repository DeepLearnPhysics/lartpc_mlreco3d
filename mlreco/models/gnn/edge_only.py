# GNN that attempts to match clusters to groups
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import numpy as np
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, LeakyReLU, Dropout, BatchNorm1d
from torch_geometric.nn import MetaLayer, GATConv

class EdgeOnlyModel(torch.nn.Module):
    """
    Model that runs edge weights through a MLP for predictions
    
    """
    def __init__(self, cfg):
        super(EdgeOnlyModel, self).__init__()
        
        if 'modules' in cfg:
                self.model_config = cfg['modules']['edge_only']
        else:
            self.model_config = cfg

        self.leak = self.model_config.get('leak', 0.1)

        self.bn_edge = BatchNorm1d(10)
        self.edge_pred_mlp = Seq(
            Lin(10, 16),
            LeakyReLU(self.leak),
            Lin(16, 32),
            LeakyReLU(self.leak),
            Lin(32, 64),
            LeakyReLU(self.leak),
            Lin(64,32),
            LeakyReLU(self.leak),
            Lin(32,16),
            LeakyReLU(self.leak),
            Lin(16,8),
            LeakyReLU(self.leak),
            Lin(8,2)
        )
    
    def forward(self, x, edge_index, e, xbatch):
        
        e = self.bn_edge(e)
        
        e = self.edge_pred_mlp(e)
        
        return {'edge_pred':[e]}
