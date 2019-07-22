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
from mlreco.utils.gnn.features.utils import edge_labels_to_node_labels
from mlreco.utils.groups import process_group_data
from mlreco.utils.metrics import SBD
from .gnn import edge_model_construct

from mlreco.utils.gnn.features.core import generate_graph


class FullEdgeModel(torch.nn.Module):
    """
    Driver for edge prediction, assumed to be with PyTorch GNN model.
    This class mostly acts as a wrapper that will hand the graph data to another model
    
    for use in config
    model:
        modules:
            edge_model:
                name: <name of edge model>
                model_cfg:
                    <dictionary of arguments to pass to model>
                remove_compton: <True/False to remove compton clusters> (default True)
                balance_classes: <True/False for loss computation> (default True)
                loss: 'L1' or 'L2' (default 'L1')
    """
    def __init__(self, cfg):
        super(FullEdgeModel, self).__init__()
        
        if 'modules' in cfg:
            self.model_config = cfg['modules']['edge_model']
        else:
            self.model_config = cfg
            
        
        self.remove_compton = self.model_config.get('remove_compton', True)
            
        # extract the model to use
        model = edge_model_construct(self.model_config.get('name', 'edge_only'))
                     
        # construct the model
        self.edge_predictor = model(self.model_config.get('model_cfg', {}))
      
        # check if primaries assignment should be thresholded
        self.pmd = self.model_config.get('primary_max_dist', None)
        
        
    def forward(self, data):
        """
        inputs data:
            - data[0] vertex_features
            - data[1] edge_features
            - data[2] edge_index
            - data[3] batch
        output:
        dictionary, with
            'edge_pred': torch.tensor with edge prediction weights
        """
        # get output
        outdict = self.edge_predictor(data[0], data[2], data[1], data[3])
        
        return outdict
    
    
class FullEdgeChannelLoss(torch.nn.Module):
    """
    Edge loss based on two channel output
    """
    def __init__(self, cfg):
        # torch.nn.MSELoss(reduction='sum')
        # torch.nn.L1Loss(reduction='sum')
        super(FullEdgeChannelLoss, self).__init__()
        self.model_config = cfg['modules']['edge_model']
            
        self.remove_compton = self.model_config.get('remove_compton', True)
        self.pmd = self.model_config.get('primary_max_dist')
        
        self.reduction = self.model_config.get('reduction', 'mean')
        self.loss = self.model_config.get('loss', 'CE')
        
        if self.loss == 'CE':
            self.lossfn = torch.nn.CrossEntropyLoss(reduction=self.reduction)
        elif self.loss == 'MM':
            p = self.model_config.get('p', 1)
            margin = self.model_config.get('margin', 1.0)
            self.lossfn = torch.nn.MultiMarginLoss(p=p, margin=margin, reduction=self.reduction)
        else:
            raise Exception('unrecognized loss: ' + self.loss)
        
        
    def forward(self, out, edge_assn, edge_index, batch):
        """
        out:
            dictionary output from GNN Model
            keys:
                'edge_pred': predicted edge weights from model forward
        """
        edge_pred = out['edge_pred']
        edge_assn = edge_assn[0]
        total_loss = self.lossfn(edge_pred, edge_assn)
        
        # compute accuracy of assignment
        # need to multiply by batch size to be accurate
        _, pred_inds = torch.max(edge_pred, 1)
        total_acc = (torch.max(batch[0]) + 1) * (pred_inds == edge_assn).sum().float()/len(edge_assn)
        
        print('edge_assn shape', edge_assn.cpu().numpy().flatten().shape)
        print('pred_inds shape', pred_inds.cpu().detach().numpy().flatten().shape)
        max_node = int(torch.max(edge_index[0]))
        node_truth = edge_labels_to_node_labels(None, edge_index[0].cpu().numpy().T, edge_assn.cpu().numpy().flatten(), node_len=max_node)
        node_preds = edge_labels_to_node_labels(None, edge_index[0].cpu().numpy().T, pred_inds.cpu().detach().numpy().flatten(), node_len=max_node)
        sbd = SBD(node_preds, node_truth)
        
        return {
            'sbd': sbd,
            'accuracy': total_acc,
            'loss_seg': total_loss
        }