# GNN that attempts to identify primary clusters
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, LeakyReLU, Dropout
import torch.nn.functional as F
from torch_geometric.nn import MetaLayer, GATConv, AGNNConv
from mlreco.utils.gnn.cluster import form_clusters, get_cluster_batch, get_cluster_label, form_clusters_new
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
        
        self.model_config = cfg['modules']['node_attention_gnn']
        self.nheads = self.model_config['nheads']
        
        # first layer increases the number of features to 16
        self.lin1 = torch.nn.Linear(4, 16)
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
        
        
    def forward(self, data):
        """
        inputs data:
            data[0] - dbscan data
        """
        # need to form graph, then pass through GNN
        clusts = form_clusters_new(data[0])
        
        # remove compton clusters (should we?)
        # if no cluster fits this condition, return
        selection = filter_compton(clusts) # non-compton looking clusters
        if not len(selection):
            x = torch.tensor([], requires_grad=True)
            if data[0].is_cuda:
                x.cuda()
            return x
        
        clusts = clusts[selection]
        
        # form complete graph
        batch = get_cluster_batch(data[0], clusts)
        edge_index = complete_graph(batch, cuda=True)
        
        # obtain vertex features
        #x = cluster_vtx_features(data[0], clusts, cuda=True)
        x = cluster_vtx_features_old(data[0], clusts, cuda=True)
        
        # go through layers
        x = F.dropout(x, training=True)
        x = F.relu(self.lin1(x))
        x = self.prop1(x, edge_index)
        x = self.prop2(x, edge_index)
        x = F.dropout(x, training=True)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)

        '''
        x = self.attn1(x, edge_index)
        x = self.attn2(x, edge_index)
        x = self.attn3(x, edge_index)
        
        x = self.node_pred_mlp(x)
        
        # return vertex features
        return x
        '''
    
class NodeLabelLoss(torch.nn.Module):
    def __init__(self, cfg):
        # torch.nn.MSELoss(reduction='sum')
        # torch.nn.L1Loss(reduction='sum')
        super(NodeLabelLoss, self).__init__()
        self.model_config = cfg['modules']['node_attention_gnn']
        
        self.lossfn = torch.nn.HingeEmbeddingLoss(reduction='sum')
        
        if 'balance_classes' in self.model_config:
            self.balance = self.model_config['balance_classes']
        else:
            # default behavior
            self.balance = True
        
    def forward(self, node_pred, data0, data1):
        """
        node_pred:
            predicted node type from model forward
        data:
            data[0] - 5 types data
            data[1] - primary data
        """
        data0 = data0[0]
        data1 = data1[0]
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
            total_loss = self.lossfn(node_pred, node_pred)
            return {
                'accuracy': 1.,
                'loss_seg': total_loss
            }
        
        clusts = clusts[selection]
        
        # get the true node labels
        primaries = assign_primaries(data1, clusts, data0)
        #node_assn = torch.tensor([2*float(i in primaries)-1. for i in range(len(clusts))]) # must return -1 or 1
        node_assn = torch.tensor([int(i in primaries) for i in range(len(clusts))]) # must return 0 or 1
        if node_pred.is_cuda:
            node_assn = node_assn.cuda()
        
        node_assn = node_assn.view(-1)
        #node_pred = node_pred.view(-1)
        
        weights = torch.tensor([1., 1.])
        if node_pred.is_cuda:
            weights = weights.cuda()
            
        if self.balance:
            ind0 = node_assn == 0
            ind1 = node_assn == 1
            # number in each class
            n0 = torch.sum(ind0).float()
            n1 = torch.sum(ind1).float()
            weights[0] = n1/(n0+n1)
            weights[1] = n0/(n0+n1)
            print('class sizes', n0, n1)
        
        #total_loss = self.lossfn(node_pred, node_assn)
        print('weights', weights)
        total_loss = F.nll_loss(node_pred, node_assn, weight=weights)
        print(total_loss)
        
        # compute accuracy of assignment
        preds = torch.argmin(node_pred, dim=1)
        print(node_pred)
        print(preds)
        tot_vox = np.sum([len(c) for c in clusts])
        int_vox = np.sum([len(clusts[i]) for i in range(len(clusts)) if node_assn[i] == preds[i]])
        total_acc = int_vox * 1.0 / tot_vox
        #total_acc = torch.tensor(primary_assign_vox_efficiency(node_assn, node_pred, clusts))
        
        return {
            'accuracy': total_acc,
            'loss_seg': total_loss
        }