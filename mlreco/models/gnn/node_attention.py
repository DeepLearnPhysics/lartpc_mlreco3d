# GNN that attempts to identify primary clusters
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
import numpy as np

class NodeAttentionModel(torch.nn.Module):
    """
    Simple GNN with several edge convolutions, followed by MetaLayer for node prediction
    """
    def __init__(self, cfg):
        super(NodeAttentionModel, self).__init__()

        print(cfg)
        if 'modules' in cfg:
            self.model_config = cfg['modules']['node_attention']
        else:
            self.model_config = cfg

        self.node_in = self.model_config.get('node_feats', 16)
        self.edge_in = self.model_config.get('edge_feats', 10)
        self.nheads = self.model_config.get('nheads', 1)

        # first layer increases the number of features to 16
        self.lin1 = torch.nn.Linear(self.node_in, 16)
        #self.lin1 = torch.nn.Linear(15, 16)

        # second and third layer are the torch implementation of AGNN
        self.prop1 = AGNNConv(requires_grad=False)
        self.prop2 = AGNNConv(requires_grad=True)

        # last layer predicts a single number per node
        self.lin2 = torch.nn.Linear(16, 2)

        '''
        # first layer increases number of features from 15 to 16
        self.attn1 = GATConv(15, 16, heads=self.nheads, concat=False)
        # second layer increases number of features from 32 to 64
        self.attn2 = GATConv(16, 32, heads=self.nheads, concat=False)
        # third layer stays at 64 features
        self.attn3 = GATConv(32, 64, heads=self.nheads, concat=False)

        # final node prediction
        self.node_pred_mlp = Seq(Lin(64, 32), LeakyReLU(0.12), Lin(32, 16), LeakyReLU(0.12), Lin(16,1))
        # note that output is not rounded with Sigmoid.
        '''


    def forward(self, x, edge_index, e, xbatch):
        """
        inputs data:
            data[0] - dbscan data
        """

        # go through layers
        x = F.dropout(x, training=True)
        x = F.relu(self.lin1(x))
        x = self.prop1(x, edge_index)
        x = self.prop2(x, edge_index)
        x = F.dropout(x, training=True)
        x = self.lin2(x)
        x = F.log_softmax(x, dim=1)

        '''
        x = self.attn1(x, edge_index)
        x = self.attn2(x, edge_index)
        x = self.attn3(x, edge_index)

        x = self.node_pred_mlp(x)
        '''

        # return vertex features
        return {'node_pred':[x]}

