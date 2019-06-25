# GNN that attempts to match clusters to groups
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import numpy as np
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, LeakyReLU, Dropout
from torch_geometric.nn import MetaLayer, GATConv
from mlreco.utils.gnn.cluster import get_cluster_batch, get_cluster_label, form_clusters_new
from mlreco.utils.gnn.primary import assign_primaries
from mlreco.utils.gnn.network import primary_bipartite_incidence
from mlreco.utils.gnn.compton import filter_compton
from mlreco.utils.gnn.data import cluster_vtx_features, cluster_edge_features, edge_assignment, cluster_vtx_features_old
from mlreco.utils.gnn.evaluation import secondary_matching_vox_efficiency
from mlreco.utils.groups import process_group_data

class BasicAttentionModel(torch.nn.Module):
    """
    Simple GNN with several edge convolutions, followed by MetLayer for edge prediction
    """
    def __init__(self, cfg):
        super(BasicAttentionModel, self).__init__()
        
        
        if 'modules' in cfg:
            self.model_config = cfg['modules']['attention_gnn']
        else:
            self.model_config = cfg
            
        self.nheads = self.model_config['nheads']
        
        # first layer increases number of features from 4 to 16
        # self.attn1 = GATConv(4, 16, heads=self.nheads, concat=False)
        # first layer increases number of features from 15 to 16
        self.attn1 = GATConv(16, 16, heads=self.nheads, concat=False)
        
        # second layer increases number of features from 16 to 32
        self.attn2 = GATConv(16, 32, heads=self.nheads, concat=False)
        
        # third layer increases number of features from 32 to 64
        self.attn3 = GATConv(32, 64, heads=self.nheads, concat=False)
    
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
        # x = cluster_vtx_features_old(data[0], clusts, cuda=True)
        #print("max input: ", torch.max(x.view(-1)))
        #print("min input: ", torch.min(x.view(-1)))
        # obtain edge features
        e = cluster_edge_features(data[0], clusts, edge_index, cuda=True)
        
        # go through layers
        x = self.attn1(x, edge_index)
        #print("max x: ", torch.max(x.view(-1)))
        #print("min x: ", torch.min(x.view(-1)))
        x = self.attn2(x, edge_index)
        #print("max x: ", torch.max(x.view(-1)))
        #print("min x: ", torch.min(x.view(-1)))
        x = self.attn3(x, edge_index)
        #print("max x: ", torch.max(x.view(-1)))
        #print("min x: ", torch.min(x.view(-1)))
        
        xbatch = torch.tensor(batch).cuda()
        x, e, u = self.edge_predictor(x, edge_index, e, u=None, batch=xbatch)
        print("max edge weight: ", torch.max(e.view(-1)))
        print("min edge weight: ", torch.min(e.view(-1)))
        return e
    
    
class EdgeLabelLoss(torch.nn.Module):
    def __init__(self, cfg):
        # torch.nn.MSELoss(reduction='sum')
        # torch.nn.L1Loss(reduction='sum')
        super(EdgeLabelLoss, self).__init__()
        self.model_config = cfg['modules']['attention_gnn']
        
        if 'loss' in self.model_config:
            if self.model_config['loss'] == 'L1':
                self.lossfn = torch.nn.L1Loss(reduction='sum')
            elif self.model_config['loss'] == 'L2':
                self.lossfn = torch.nn.MSELoss(reduction='sum')
        else:
            self.lossfn = torch.nn.L1Loss(reduction='sum')
        
        if 'balance_classes' in self.model_config:
            self.balance = self.model_config['balance_classes']
        else:
            # default behavior
            self.balance = True
        
    def forward(self, edge_pred, data0, data1, data2):
        """
        edge_pred:
            predicted edge weights from model forward
        data:
            data[0] - 5 types data
            data[1] - groups data
            data[2] - primary data
        """
        data0 = data0[0]
        data1 = data1[0]
        data2 = data2[0]
        # first decide what true edges should be
        # need to form graph, then pass through GNN
        # clusts = form_clusters(data0)
        clusts = form_clusters_new(data0)
        
        # remove track-like particles
        # types = get_cluster_label(data0, clusts)
        # selection = types > 1 # 0 or 1 are track-like
        # clusts = clusts[selection]
        
        # remove compton clusters
        # if no cluster fits this condition, return
        selection = filter_compton(clusts) # non-compton looking clusters
        if not len(selection):
            total_loss = self.lossfn(edge_pred, edge_pred)
            return {
                'accuracy': 1.,
                'loss_seg': total_loss
            }
        
        clusts = clusts[selection]
        
        # process group data
        # data_grp = process_group_data(data1, data0)
        data_grp = data1
        
        # form primary/secondary bipartite graph
        primaries = assign_primaries(data2, clusts, data0)
        batch = get_cluster_batch(data0, clusts)
        edge_index = primary_bipartite_incidence(batch, primaries)
        group = get_cluster_label(data_grp, clusts)
        
        primaries_true = assign_primaries(data2, clusts, data1, use_labels=True)
        print("primaries (est):  ", primaries)
        print("primaries (true): ", primaries_true)
        
        # determine true assignments
        edge_assn = edge_assignment(edge_index, batch, group, cuda=True)
        
        edge_assn = edge_assn.view(-1)
        edge_pred = edge_pred.view(-1)
        
        if self.balance:
            # weight edges so that 0/1 labels appear equally often
            ind0 = edge_assn == 0
            ind1 = edge_assn == 1
            # number in each class
            n0 = torch.sum(ind0).float()
            n1 = torch.sum(ind1).float()
            print("n0 = ", n0, " n1 = ", n1)
            # weights to balance classes
            w0 = n1 / (n0 + n1)
            w1 = n0 / (n0 + n1)
            print("w0 = ", w0, " w1 = ", w1)
            edge_assn[ind0] = w0 * edge_assn[ind0]
            edge_assn[ind1] = w1 * edge_assn[ind1]
            edge_pred = edge_pred.clone()
            edge_pred[ind0] = w0 * edge_pred[ind0]
            edge_pred[ind1] = w1 * edge_pred[ind1]
            
            
        
        total_loss = self.lossfn(edge_pred, edge_assn)
        
        # compute accuracy of assignment
        # need to multiply by batch size to be accurate
        total_acc = (np.max(batch) + 1) * torch.tensor(secondary_matching_vox_efficiency(edge_index, edge_assn, edge_pred, primaries, clusts, len(clusts)))
        
        return {
            'accuracy': total_acc,
            'loss_seg': total_loss
        }
