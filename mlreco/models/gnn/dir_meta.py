# GNN that attempts to match clusters to groups
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import numpy as np
import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, LeakyReLU, Dropout, BatchNorm1d, Bilinear
from torch_geometric.nn import MetaLayer, GATConv, AGNNConv, NNConv

from .edge_pred import EdgeModel as EdgePredModel, BilinEdgeModel as BilinEdgePredModel
      
        
        
class BilinDirModel(torch.nn.Module):
    def __init__(self, radlength):
        super(BilinDirModel, self).__init__()
        
        self.radlength = radlength
        
    def forward(self, source, target, edge_attr, u, batch):
        
        edge_A = edge_attr[:,:-1] # edge orientation: M x 9
        edge_l = edge_attr[:,-1] # edge length: M x 1
        
        
        wl = torch.exp(-edge_l / self.radlength)
        # edge_A * source * exp(-edge_l/radlength)
        w1 = torch.sum(torch.mul(edge_A, source), dim=1)
        # edge_A * target * exp(-edge_l/radlength)
        w2 = torch.sum(torch.mul(edge_A, target), dim=1)
        # take minimum
        w, _  = torch.min(torch.stack([w1, w2], dim=1), dim=1)
        w = torch.mul(w, wl)
        
        return w


class EdgeMetaModel(torch.nn.Module):
    """
    Simple GNN with several MetaLayers, followed by MetaLayer for edge prediction
    
    for use in config
    model:
        modules:
            attention_gnn:
                nheads: <number of heads for attention>
    """
    def __init__(self, cfg):
        super(EdgeMetaModel, self).__init__()
        
        
        if 'modules' in cfg:
            self.model_config = cfg['modules']['attention_gnn']
        else:
            self.model_config = cfg

        self.radlength = self.model_config.get('radlength', 87)
        
        self.edgep = MetaLayer(BilinDirModel(self.radlength))
        self.lin = Lin(2, 2)
        
        
    def forward(self, x, edge_index, e, xbatch):
        """
        inputs data:
            x - vertex features
            edge_index - graph edge list
            e - edge features
            xbatch - node batchid
        """
        
        # note that this is essentially deterministic
        x, e, u = self.edgep(x, edge_index, e, u=None, batch=xbatch)
        e = torch.stack([1-e, e], dim=1)
        e = self.lin(e)

        return [[e]]
