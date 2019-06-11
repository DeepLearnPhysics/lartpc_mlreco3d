# GNN that attempts to match clusters to groups
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, LeakyReLU
from torch_geometric.nn import MetaLayer, GATConv
from mlreco.utils.gnn.cluster import form_clusters, get_cluster_batch, get_cluster_label
from mlreco.utils.gnn.primary import assign_primaries
from mlreco.utils.gnn.network import primary_bipartite_incidence
from mlreco.utils.gnn.compton import filter_compton
from mlreco.utils.gnn.data import cluster_vtx_features, cluster_edge_features, edge_assignment

class BasicAttentionModel(torch.nn.Module):
    """
    Simple GNN with several edge convolutions, followed by MetLayer for edge prediction
    """
    def __init__(self, cfg):
        super(BasicAttentionModel, self).__init__()
        
        self.model_config = cfg['modules']['attention_gnn']
        self.nheads = self.model_config['nheads']
        
        # first layer increases number of features from 3 to 16
        self.attn1 = GATConv(3, 16, heads=self.nheads, concat=False)
        
        # second layer increases number of features from 16 to 32
        self.attn2 = GATConv(16, 32, heads=self.nheads, concat=False)
        
        # third layer increases number of features from 32 to 64
        self.attn3 = GATConv(32, 64, heads=self.nheads, concat=False)
    
        # final prediction layer
        self.edge_pred_mlp = Seq(Lin(138, 64), LeakyReLU(0.12), Lin(64, 16), LeakyReLU(0.12), Lin(16,1), Sigmoid())
        
        def edge_pred_model(source, target, edge_attr, u, batch):
            out = torch.cat([source, target, edge_attr], dim=1)
            out = self.edge_pred_mlp(out)
            return out
        
        self.edge_predictor = MetaLayer(edge_pred_model, None, None)
        
        
    def forward(self, data):
        """
        inputs data:
            data[0] - 5 types data
            data[1] - groups data
            data[2] - primary data
        """
        # need to form graph, then pass through GNN
        clusts = form_clusters(data[0])
        
        # remove compton clusters
        selection = filter_compton(clusts) # non-compton looking clusters
        clusts = clusts[selection]
        
        # form primary/secondary bipartite graph
        primaries = assign_primaries(data[2], clusts, data[1])
        batch = get_cluster_batch(data[0], clusts)
        edge_index = primary_bipartite_incidence(batch, primaries)
        
        # obtain vertex features
        x = cluster_vtx_features(data[0], clusts)
        # obtain edge features
        e = cluster_edge_features(data[0], clusts, edge_index)
        
        # go through layers
        x = self.attn1(x, edge_index)
        x = self.attn2(x, edge_index)
        x = self.attn3(x, edge_index)
        
        xbatch = torch.tensor(batch)
        x, e, u = self.edge_predictor(x, edge_index, e, u=None, batch=xbatch)
        return e
    
    
class EdgeLabelLoss(torch.nn.Module):
    def __init__(self, cfg, lossfn=nn.MSELoss()):
        super(EdgeLabelLoss, self).__init__()
        self.lossfn = lossfn
        
    def forward(self, edge_pred, data):
        """
        edge_pred:
            predicted edge weights from model forward
        data:
            data[0] - 5 types data
            data[1] - groups data
            data[2] - primary data
        """
        # first decide what true edges should be
        # need to form graph, then pass through GNN
        clusts = form_clusters(data[0])
        
        # remove compton clusters
        selection = filter_compton(clusts) # non-compton looking clusters
        clusts = clusts[selection]
        
        # form primary/secondary bipartite graph
        primaries = assign_primaries(data[2], clusts, data[1])
        batch = get_cluster_batch(data[0], clusts)
        edge_index = primary_bipartite_incidence(batch, primaries)
        group = get_cluster_label(data[1], clusts)
        
        # determine true assignments
        edge_assn = edge_assignment(edge_index, batch, group)
        
        return self.lossfn(edge_assn, edge_pred)