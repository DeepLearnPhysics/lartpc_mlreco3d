# GNN that attempts to match clusters to groups
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import numpy as np
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, LeakyReLU, Dropout, BatchNorm1d
from torch_geometric.nn import MetaLayer, NNConv

from .edge_pred import EdgeModel, BilinEdgeModel

class NNConvModel(torch.nn.Module):
    """
    Simple GNN with several edge convolutions, followed by MetLayer for edge prediction

    for use in config
    model:
        modules:
            edge_model:
              name: nnconv
    """
    def __init__(self, cfg):
        super(NNConvModel, self).__init__()


        if 'modules' in cfg:
            self.model_config = cfg['modules']['edge_nnconv']
        else:
            self.model_config = cfg


        self.node_in = self.model_config.get('node_feats', 16)
        self.edge_in = self.model_config.get('edge_feats', 19)

        self.aggr = self.model_config.get('aggr', 'add')
        self.leak = self.model_config.get('leak', 0.1)

        # extra flags
        self.batchnorm_layer = self.model_config.get('batchnorm_layer', False) # whether to apply batchnorm everywhere
        self.mlp_depth = self.model_config.get("mlp_depth", 2) # depth of mlp

        # perform batch normalization
        self.bn_node = BatchNorm1d(self.node_in)
        self.bn_edge = BatchNorm1d(self.edge_in)

        self.num_mp = self.model_config.get('num_mp', 3)

        self.nn = torch.nn.ModuleList()
        self.layer = torch.nn.ModuleList()
        ninput = self.node_in
        noutput = max(2*self.node_in, 32)
        # construct the mlp nodes number in each layer
        # need two because after first message passing
        # the layer structure changed
        self.mlp_node_numbers = [self.edge_in]
        mlp_node_numbers2 = [self.edge_in]
        step  = int((ninput*noutput-ninput)/(self.mlp_depth-1))
        step2 = int((noutput**2 - noutput)/(self.mlp_depth-1))
        for j in range(self.mlp_depth):
            # weird layer configuration just for being compatible with previously trained nn which has (ninput, ninput*noutput) layers
            if j==0:
                self.mlp_node_numbers.append(ninput)
                mlp_node_numbers2.append(noutput)
            elif j!=self.mlp_depth-1:
                self.mlp_node_numbers.append(self.edge_in+int(j*step))
                mlp_node_numbers2.append(self.edge_in+int(j*step2))
            else:
                self.mlp_node_numbers.append(ninput*noutput)
                mlp_node_numbers2.append(noutput**2)
        for i in range(self.num_mp):
            modules = []
            for j in range(self.mlp_depth):
                if self.batchnorm_layer:
                    modules.append(
                        BatchNorm1d(self.mlp_node_numbers[j])
                    )
                modules.append(
                    Lin(self.mlp_node_numbers[j], self.mlp_node_numbers[j+1])
                )
                modules.append(
                    LeakyReLU(self.leak)
                )
            self.nn.append(Seq(*modules))
            self.layer.append(
                NNConv(ninput, noutput, self.nn[i], aggr=self.aggr)
            )
            ninput = noutput
            self.mlp_node_numbers = mlp_node_numbers2

        # final prediction layer
        pred_cfg = self.model_config.get('pred_model', 'basic')
        if pred_cfg == 'basic':
            self.edge_predictor = MetaLayer(EdgeModel(noutput, self.edge_in, self.leak))
        elif pred_cfg == 'bilin':
            self.edge_predictor = MetaLayer(BilinEdgeModel(noutput, self.edge_in, self.leak))
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

        x = x.view(-1,self.node_in)
        e = e.view(-1,self.edge_in)
        if self.edge_in > 1:
            e = self.bn_edge(e)
        if self.node_in > 1:
            x = self.bn_node(x)

        # go through layers
        for i in range(self.num_mp):
            if self.batchnorm_layer:
                x = self.bn_node(x)
            x = self.layer[i](x, edge_index, e)

        x, e, u = self.edge_predictor(x, edge_index, e, u=None, batch=xbatch)

        return {'edge_pred':[e]}
