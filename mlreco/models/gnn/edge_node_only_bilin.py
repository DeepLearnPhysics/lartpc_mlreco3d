# GNN that attempts to match clusters to groups
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import numpy as np
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, LeakyReLU, Dropout, BatchNorm1d, Bilinear
from torch_geometric.nn import MetaLayer, GATConv

class BilinEdgePredMLP(torch.nn.Module):
    """
    Model that uses bilinear layers to produce edge predictions
    """
    def __init__(self, node_in, edge_in, leak=0.0):
        super(BilinEdgePredMLP, self).__init__()
        self.bse = Bilinear(node_in, edge_in, 16, bias=True)
        self.bte = Bilinear(node_in, edge_in, 16, bias=True)
        self.bst = Bilinear(node_in, node_in, edge_in, bias=False)
        self.bee = Bilinear(edge_in, edge_in, 16, bias=True)
        
        self.mlp = Seq(
            Lin(3*16, 64),
            LeakyReLU(leak),
            Lin(64, 64),
            LeakyReLU(leak),
            Lin(64,32),
            LeakyReLU(leak),
            Lin(32,16),
            LeakyReLU(leak),
            Lin(16,2)
        )
        
    def forward(self, source, target, edge_attr):
        # two bilinear forms
        x = self.bse(source, edge_attr)
        y = self.bte(target, edge_attr)
        
        # trilinear form
        z = self.bst(source, target)
        z = self.bee(z, edge_attr)
        
        out = torch.cat([x, y, z], dim=1)
        out = self.mlp(out)
        return out
         

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
        
        self.edge_pred_layer = BilinEdgePredMLP(
            self.node_in,
            self.edge_in,
            leak = self.leak
        )
        
        def edge_pred_model(source, target, edge_attr, u, batch):
            out = self.edge_pred_layer(source, target, edge_attr)
            return out
        
        self.edge_predictor = MetaLayer(edge_pred_model, None, None)
    
    def forward(self, x, edge_index, e, xbatch):
        
        x = x.view(-1,self.node_in)
        e = e.view(-1,self.edge_in)
        if self.edge_in > 1:
            e = self.bn_edge(e)
        if self.node_in > 1:
            x = self.bn_node(x)
        
        x, e, u = self.edge_predictor(x, edge_index, e, u=None, batch=xbatch)
        
        return {
            'edge_pred': e
        }