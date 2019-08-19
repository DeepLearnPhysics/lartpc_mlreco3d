# GNN that attempts to match clusters to groups
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import numpy as np
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, LeakyReLU, Dropout, BatchNorm1d
from torch_geometric.nn import MetaLayer, GATConv

from .edge_pred import EdgeModel, BilinEdgeModel

class EdgeNodeOnlyModel(torch.nn.Module):
    """
    Model that runs edge weights + node weights through a MLP for predictions
    
    """
    def __init__(self, cfg):
        super(EdgeNodeOnlyModel, self).__init__()
        
        if 'modules' in cfg:
                self.model_config = cfg['modules']['edge_only']
        else:
            self.model_config = cfg
            
        self.node_in = self.model_config.get('node_feats', 16)
        self.edge_in = self.model_config.get('edge_feats', 10)

        self.leak = self.model_config.get('leak', 0.1)


        self.bn_node = BatchNorm1d(self.node_in)
        self.bn_edge = BatchNorm1d(self.edge_in)
  
        # final prediction layer
        pred_cfg = self.model_config.get('pred_model', 'basic')
        if pred_cfg == 'basic':
            self.edge_predictor = MetaLayer(EdgeModel(self.node_in, self.edge_in, self.leak))
        elif pred_cfg == 'bilin':
            self.edge_predictor = MetaLayer(BilinEdgeModel(self.node_in, self.edge_in, self.leak))
        else:
            raise Exception('unrecognized prediction model: ' + pred_cfg)
            
    
    def forward(self, x, edge_index, e, xbatch):
        
        x = x.view(-1,self.node_in)
        e = e.view(-1,self.edge_in)
        if self.edge_in > 1:
            e = self.bn_edge(e)
        if self.node_in > 1:
            x = self.bn_node(x)
        
        x, e, u = self.edge_predictor(x, edge_index, e, u=None, batch=xbatch)
        
        return {'edge_pred':[e]}
