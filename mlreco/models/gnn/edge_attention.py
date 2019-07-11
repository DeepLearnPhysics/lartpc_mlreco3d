# GNN that attempts to match clusters to groups
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import numpy as np
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, LeakyReLU, Dropout, BatchNorm1d
from torch_geometric.nn import MetaLayer, GATConv
from mlreco.utils.gnn.cluster import get_cluster_batch, get_cluster_label, form_clusters_new
from mlreco.utils.gnn.primary import assign_primaries, analyze_primaries
from mlreco.utils.gnn.network import primary_bipartite_incidence
from mlreco.utils.gnn.compton import filter_compton
from mlreco.utils.gnn.data import cluster_vtx_features, cluster_edge_features, edge_assignment, cluster_vtx_features_old
from mlreco.utils.gnn.evaluation import secondary_matching_vox_efficiency
from mlreco.utils.groups import process_group_data

class BasicAttentionModel(torch.nn.Module):
    """
    Simple GNN with several edge convolutions, followed by MetLayer for edge prediction
    
    for use in config
    model:
        modules:
            attention_gnn:
                nheads: <number of heads for attention>
    """
    def __init__(self, cfg):
        super(BasicAttentionModel, self).__init__()
        
        
        if 'modules' in cfg:
            self.model_config = cfg['modules']['attention_gnn']
        else:
            self.model_config = cfg
            
        self.nheads = self.model_config['nheads']
        
        # perform batch normalization at each step
        self.bn_node = BatchNorm1d(16)
        
        # first layer increases number of features from 4 to 16
        # self.attn1 = GATConv(4, 16, heads=self.nheads, concat=False)
        # first layer increases number of features from 15 to 16
        self.attn1 = GATConv(16, 16, heads=self.nheads, concat=False)
        
        # second layer increases number of features from 16 to 32
        self.attn2 = GATConv(16, 32, heads=self.nheads, concat=False)
        
        # third layer increases number of features from 32 to 64
        self.attn3 = GATConv(32, 64, heads=self.nheads, concat=False)
        
        self.bn_edge = BatchNorm1d(10)
    
        # final prediction layer
        self.edge_pred_mlp = Seq(Lin(138, 64), Dropout(p=0.2), LeakyReLU(0.12), Dropout(p=0.2), Lin(64, 16), LeakyReLU(0.12), Lin(16,1), Sigmoid())
        
        def edge_pred_model(source, target, edge_attr, u, batch):
            out = torch.cat([source, target, edge_attr], dim=1)
            out = self.edge_pred_mlp(out)
            return out
        
        self.edge_predictor = MetaLayer(edge_pred_model, None, None)
        
        
    def forward(self, data):
        """
        inputs data:
            data[0] - dbscan data
            data[1] - primary data
        """
        # need to form graph, then pass through GNN
        clusts = form_clusters_new(data[0])
        
        # remove track-like particles
        #types = get_cluster_label(data[0], clusts)
        #selection = types > 1 # 0 or 1 are track-like
        #clusts = clusts[selection]
        
        # remove compton clusters
        # if no cluster fits this condition, return
        selection = filter_compton(clusts) # non-compton looking clusters
        if not len(selection):
            e = torch.tensor([], requires_grad=True)
            if data[0].is_cuda:
                e.cuda()
            return e
        
        clusts = clusts[selection]
        
        # process group data
        # data_grp = process_group_data(data[1], data[0])
        # data_grp = data[1]
        
        # form primary/secondary bipartite graph
        primaries = assign_primaries(data[1], clusts, data[0])
        batch = get_cluster_batch(data[0], clusts)
        edge_index = primary_bipartite_incidence(batch, primaries, cuda=True)
        
        # obtain vertex features
        x = cluster_vtx_features(data[0], clusts, cuda=True)
        # batch normalization
        x = self.bn_node(x)
        # x = cluster_vtx_features_old(data[0], clusts, cuda=True)
        #print("max input: ", torch.max(x.view(-1)))
        #print("min input: ", torch.min(x.view(-1)))
        # obtain edge features
        e = cluster_edge_features(data[0], clusts, edge_index, cuda=True)
        # batch normalization
        e = self.bn_edge(e)
        
        # go through layers
        x = self.attn1(x, edge_index)

        x = self.attn2(x, edge_index)

        x = self.attn3(x, edge_index)
        
        xbatch = torch.tensor(batch).cuda()
        x, e, u = self.edge_predictor(x, edge_index, e, u=None, batch=xbatch)

        return {
            'edge_pred': e
        }
