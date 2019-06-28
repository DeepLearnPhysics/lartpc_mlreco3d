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
from torch_geometric.nn import MetaLayer, EdgeConv
from torch_scatter import scatter_mean
import numpy as np

class NodeEConvModel(torch.nn.Module):
    """
    Simple GNN with several edge convolutions, followed by MetaLayer for node prediction
    """
    def __init__(self, cfg):
        super(NodeEConvModel, self).__init__()
        
        # first layer increases number of features from 4 to 16
        #self.econv_mlp1 = Seq(Lin(32,32), LeakyReLU(0.1), Lin(32,16), LeakyReLU(0.1))
        self.econv_mlp1 = Seq(Lin(8,32), LeakyReLU(0.1), Lin(32,16), LeakyReLU(0.1))
        self.econv1 = EdgeConv(self.econv_mlp1, aggr='max')
        
        # second layer increases number of features from 16 to 32
        self.econv_mlp2 = Seq(Lin(32,64), LeakyReLU(0.1), Lin(64,32), LeakyReLU(0.1))
        self.econv2 = EdgeConv(self.econv_mlp2, aggr='max')
        
        # third layer increases number of features from 32 to 64
        self.econv_mlp3 = Seq(Lin(64,128), LeakyReLU(0.1), Lin(128,64), LeakyReLU(0.1))
        self.econv3 = EdgeConv(self.econv_mlp3, aggr='max')
        
        # final prediction layer
        self.edge_mlp = Seq(Lin(128, 64), LeakyReLU(0.12), Lin(64, 16))
        self.node_mlp_1 = Seq(Lin(80, 64), LeakyReLU(0.12), Lin(64, 32))
        self.node_mlp_2 = Seq(Lin(32, 16), LeakyReLU(0.12), Lin(16, 2))
        #self.node_mlp = Seq(Lin(64, 32), LeakyReLU(0.12), Lin(32, 16), LeakyReLU(0.12), Lin(32, 2))

        def edge_model(src, target, edge_attr, u, batch):
            # source, target: [E, F_x], where E is the number of edges.
            # edge_attr: [E, F_e]
            # u: [B, F_u], where B is the number of graphs.
            # batch: [E] with max entry B - 1.
            out = torch.cat([src, target], 1)
            return self.edge_mlp(out)

        def node_model(x, edge_index, edge_attr, u, batch):
            # x: [N, F_x], where N is the number of nodes.
            # edge_index: [2, E] with max entry N - 1.
            # edge_attr: [E, F_e]
            # u: [B, F_u]
            # batch: [N] with max entry B - 1.
            row, col = edge_index
            out = torch.cat([x[col], edge_attr], dim=1)
            out = self.node_mlp_1(out)
            out = scatter_mean(out, row, dim=0, dim_size=x.size(0))
            return self.node_mlp_2(out)

        self.predictor = MetaLayer(edge_model, node_model, None)
        
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
        if not len(edge_index):
            x = torch.tensor([], requires_grad=True)
            if data[0].is_cuda:
                x.cuda()
            return x
        
        batch = torch.tensor(batch)
        if data[0].is_cuda:
            batch = batch.cuda()
        
        # obtain vertex features
        #x = cluster_vtx_features(data[0], clusts, cuda=True)
        x = cluster_vtx_features_old(data[0], clusts, cuda=True)
        
        # go through layers
        x = self.econv1(x, edge_index)
        x = self.econv2(x, edge_index)
        x = self.econv3(x, edge_index)

        x, e, u = self.predictor(x, edge_index, edge_attr=None, u=None, batch=batch)
        return F.log_softmax(x, dim=1)

class NodeLabelLoss(torch.nn.Module):
    def __init__(self, cfg):
        # torch.nn.MSELoss(reduction='sum')
        # torch.nn.L1Loss(reduction='sum')
        super(NodeLabelLoss, self).__init__()
        self.model_config = cfg['modules']['node_econv_gnn']
        
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
        preds = torch.argmax(node_pred, dim=1)
        print('node_assn', node_assn)
        print('preds', preds)
        tot_c = len(clusts)
        int_c = torch.sum(node_assn == preds).item()
        total_acc = int_c * 1.0 / tot_c
        print(tot_c, int_c, total_acc)
        #total_acc = torch.tensor(primary_assign_vox_efficiency(node_assn, node_pred, clusts))
        
        return {
            'accuracy': total_acc,
            'loss_seg': total_loss
        }