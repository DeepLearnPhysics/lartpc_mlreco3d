# GNN that attempts to identify primary clusters using simple edge convolutions
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, LeakyReLU, Dropout
import torch.nn.functional as F
from torch_geometric.nn import MetaLayer, GATConv, AGNNConv
from mlreco.utils.gnn.cluster import get_cluster_batch, get_cluster_label, form_clusters_new
from mlreco.utils.gnn.primary import assign_primaries
from mlreco.utils.gnn.network import complete_graph
from mlreco.utils.gnn.compton import filter_compton
from mlreco.utils.gnn.data import cluster_vtx_features, cluster_vtx_features_old
from mlreco.utils.gnn.evaluation import primary_assign_vox_efficiency
from torch_geometric.nn import MetaLayer, EdgeConv
from torch_scatter import scatter_mean
import numpy as np

class NodeEConvModel(torch.nn.Module):
    """
    Simple GNN with several edge convolutions, followed by MetaLayer for node prediction
    """
    def __init__(self, cfg):
        super(NodeEConvModel, self).__init__()

        if 'modules' in cfg:
            self.model_config = cfg['modules']['node_econv']
        else:
            self.model_config = cfg

        self.node_in = self.model_config.get('node_feats', 16)
        self.edge_in = self.model_config.get('edge_feats', 10)
        
        # first layer increases number of features from 4 to 16
        self.econv_mlp1 = Seq(Lin(2*self.node_in,32), LeakyReLU(0.1), Lin(32,16), LeakyReLU(0.1))
        self.econv1 = EdgeConv(self.econv_mlp1, aggr='max')
        
        # second layer increases number of features from 16 to 32
        self.econv_mlp2 = Seq(Lin(32,64), LeakyReLU(0.1), Lin(64,32), LeakyReLU(0.1))
        self.econv2 = EdgeConv(self.econv_mlp2, aggr='max')
        
        # third layer increases number of features from 32 to 64
        self.econv_mlp3 = Seq(Lin(64,128), LeakyReLU(0.1), Lin(128,64), LeakyReLU(0.1))
        self.econv3 = EdgeConv(self.econv_mlp3, aggr='max')
        
        # final prediction layer
        class EdgeModel(torch.nn.Module):
            def __init__(self):
                super(EdgeModel, self).__init__()

                self.edge_mlp = Seq(Lin(128, 64), LeakyReLU(0.12), Lin(64, 16))

            def forward(self, src, dest, edge_attr, u, batch):
                return self.edge_mlp(torch.cat([src, dest], dim=1))
            
        class NodeModel(torch.nn.Module):
            def __init__(self):
                super(NodeModel, self).__init__()

                self.node_mlp_1 = Seq(Lin(80, 64), LeakyReLU(0.12), Lin(64, 32))
                self.node_mlp_2 = Seq(Lin(32, 16), LeakyReLU(0.12), Lin(16, 2))
                #self.node_mlp = Seq(Lin(64, 32), LeakyReLU(0.12), Lin(32, 16), LeakyReLU(0.12), Lin(32, 2))

            def forward(self, x, edge_index, edge_attr, u, batch):
                row, col = edge_index
                out = torch.cat([x[col], edge_attr], dim=1)
                out = self.node_mlp_1(out)
                out = scatter_mean(out, row, dim=0, dim_size=x.size(0))
                return self.node_mlp_2(out)
        
        self.predictor = MetaLayer(EdgeModel(), NodeModel())
        
    def forward(self, x, edge_index, e, xbatch):
        """
        inputs data:
            data[0] - dbscan data
        """
        
        # go through layers
        x = self.econv1(x, edge_index)
        x = self.econv2(x, edge_index)
        x = self.econv3(x, edge_index)

        x, e, u = self.predictor(x, edge_index, e, u=None, batch=xbatch)
        x = F.log_softmax(x, dim=1)
        return {'node_pred':[x]}

