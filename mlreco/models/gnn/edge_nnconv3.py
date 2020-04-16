# GNN that attempts to match clusters to groups
# debug model
# change the MLP layer configuration for node updating
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
        self.update_edge = self.model_config.get('udpate_edge', False) # whether to update edge in each message passing loop

        # perform batch normalization
        self.bn_node = torch.nn.ModuleList()
        self.bn_edge = BatchNorm1d(self.edge_in)

        self.num_mp = self.model_config.get('num_mp', 3)

        self.nn = torch.nn.ModuleList()
        self.edge_updates = torch.nn.ModuleList()
        self.layer = torch.nn.ModuleList()
        ninput = self.node_in
        noutput = max(2*self.node_in, 32)
        # construct the mlp nodes number in each layer
        # need two because after first message passing
        # the layer structure changed
        self.mlp_node_numbers = [self.edge_in]
        mlp_node_numbers2 = [self.edge_in]
        for j in range(self.mlp_depth):
            # weird layer configuration just for being compatible with previously trained nn which has (ninput, ninput*noutput) layers
            if j!=self.mlp_depth-1:
                self.mlp_node_numbers.append(ninput)
                mlp_node_numbers2.append(noutput)
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
            self.bn_node.append(BatchNorm1d(ninput))
            self.edge_updates.append(
                MetaLayer(EdgeLayer(noutput, self.edge_in, self.edge_in,leakiness=self.leak))
            )
            ninput = noutput
            self.mlp_node_numbers = mlp_node_numbers2

        # final prediction layer
        pred_cfg = self.model_config.get('pred_model', 'basic')
        if pred_cfg == 'basic':
            self.edge_predictor = torch.nn.Linear(self.edge_in, 2)
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


        # go through layers
        for i in range(self.num_mp):
            x = self.bn_node[i](x)
            x = self.layer[i](x, edge_index, e)
            if self.update_edge:
                # it also activate x as well.
                # to-do: this structure is just for making previous training still usable.
                #        we can abandon in future
                x = F.leaky_relu(x, negative_slope=self.leakiness)
                _, e, _ = self.edge_updates[i](x, edge_index, e)

        e = self.edge_predictor(e)

        return {'edge_pred':[e]}



class EdgeLayer(torch.nn.Module):
    '''
    borrowed from: https://github.com/dkoh0207/lartpc_mlreco3d/blob/443bdd8ba10b4e80f937fa637f28cf931e65e57c/mlreco/models/gnn/modular_gnn.py

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
        self.edge_mlp = Seq(
            BatchNorm1d(2 * node_in + edge_in),
            Lin(2 * node_in + edge_in, edge_out),
            LeakyReLU(negative_slope=leakiness),
            BatchNorm1d(edge_out),
            Lin(edge_out, edge_out),
            LeakyReLU(negative_slope=leakiness),
            BatchNorm1d(edge_out),
            Lin(edge_out, edge_out)
        )

    def forward(self, src, dest, edge_attr, u=None, batch=None):
        out = torch.cat([src, dest, edge_attr], dim=1)
        return self.edge_mlp(out)