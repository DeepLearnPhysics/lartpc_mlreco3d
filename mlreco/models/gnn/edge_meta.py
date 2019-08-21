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
      
        
        
class BilinEdgeModel(torch.nn.Module):
    def __init__(self, node_in, edge_in, edge_out, leak):
        super(BilinEdgeModel, self).__init__()

        self.bse = Bilinear(node_in, edge_in, edge_out, bias=True)
        self.bte = Bilinear(node_in, edge_in, edge_out, bias=True)
        self.bst = Bilinear(node_in, node_in, edge_out, bias=False)
        self.bee = Bilinear(edge_out, edge_in, edge_out, bias=True)
        
        self.relu = LeakyReLU(leak)
        
    def forward(self, source, target, edge_attr, u, batch):
        # two bilinear forms
        x = self.bse(source, edge_attr)
        y = self.bte(target, edge_attr)
        
        # trilinear form
        z = self.bst(source, target)
        z = self.bee(z, edge_attr)
        
        return self.relu(z)


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

        self.leak = self.model_config.get('leak', 0.1)
        
        self.node_in = self.model_config.get('node_feats', 16)
        self.edge_in = self.model_config.get('edge_feats', 10)
        
        self.aggr = self.model_config.get('aggr', 'add')
        
        self.bn_node = BatchNorm1d(self.node_in)
        self.bn_edge = BatchNorm1d(self.edge_in)
        
        self.num_mp = self.model_config.get('num_mp', 3)
        
        self.nn = torch.nn.ModuleList()
        self.en = torch.nn.ModuleList()
        self.layer = torch.nn.ModuleList()
        einput = self.edge_in
        eoutput = max(self.edge_in, 32)
        ninput = self.node_in
        noutput = max(2*self.node_in, 32)
        for i in range(self.num_mp):
            self.en.append(
                MetaLayer(BilinEdgeModel(ninput, einput, eoutput, self.leak))
            )
            self.nn.append(
                Seq(
                    Lin(eoutput, ninput),
                    LeakyReLU(self.leak),
                    Lin(ninput, ninput*noutput),
                    LeakyReLU(self.leak)
                )
            )
            self.layer.append(
                NNConv(ninput, noutput, self.nn[i], aggr=self.aggr)
            )
            ninput = noutput
            einput = eoutput
    
        # final prediction layer
        pred_cfg = self.model_config.get('pred_model', 'basic')
        if pred_cfg == 'basic':
            self.edge_predictor = MetaLayer(EdgePredModel(noutput, eoutput, self.leak))
        elif pred_cfg == 'bilin':
            self.edge_predictor = MetaLayer(BilinEdgePredModel(noutput, eoutput, self.leak))
        else:
            raise Exception('unrecognized prediction model: ' + pred_cfg)
        
        
    def forward(self, x, edge_index, e, xbatch):
        """
        inputs data:
            x - vertex features
            edge_index - graph edge list
            e - edge features
            xbatch - node batchid
        """
        
        # batch normalization of node features
        x = self.bn_node(x)
        
        # batch normalization of edge features
        e = self.bn_edge(e)
        
        # go through layers
        for i in range(self.num_mp):
            x, e, u = self.en[i](x, edge_index, e, u=None, batch=xbatch)
            x = self.layer[i](x, edge_index, e)
        
        x, e, u = self.edge_predictor(x, edge_index, e, u=None, batch=xbatch)

        return {'edge_pred':[e]}
