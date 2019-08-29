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
from mlreco.utils.gnn.evaluation import secondary_matching_vox_efficiency, secondary_matching_vox_efficiency3
from mlreco.utils.gnn.evaluation import DBSCAN_cluster_metrics
from mlreco.utils.groups import process_group_data
from .gnn import edge_model_construct

class EdgeModel(torch.nn.Module):
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
        super(EdgeModel, self).__init__()
        
        if 'modules' in cfg:
            self.model_config = cfg['modules']['edge_model']
        else:
            self.model_config = cfg
        
        self.remove_compton = self.model_config.get('remove_compton', True)
        self.compton_thresh = self.model_config.get('compton_thresh', 30)
            
        # extract the model to use
        model = edge_model_construct(self.model_config.get('name', 'edge_only'))
                     
        # construct the model
        self.edge_predictor = model(self.model_config.get('model_cfg', {}))
      
        # check if primaries assignment should be thresholded
        self.pmd = self.model_config.get('primary_max_dist', None)
        
        
    def forward(self, data):
        """
        inputs data:
            data[0] - dbscan data
            data[1] - primary data
        output:
        dictionary, with
            'edge_pred': torch.tensor with edge prediction weights
        """
        # get device
        device = data[0].device
        
        # need to form graph, then pass through GNN
        clusts = form_clusters_new(data[0])
        
        # remove compton clusters
        # if no cluster fits this condition, return
        if self.remove_compton:
            selection = filter_compton(clusts, self.compton_thresh) # non-compton looking clusters
            if not len(selection):
                e = torch.tensor([], requires_grad=True)
                if data[0].is_cuda:
                    e.cuda()
                return e

            clusts = clusts[selection]
        
        # form primary/secondary bipartite graph
        primaries = assign_primaries(data[1], clusts, data[0], max_dist=self.pmd)
        batch = get_cluster_batch(data[0], clusts)
        edge_index = primary_bipartite_incidence(batch, primaries, device=device)
        if not edge_index.shape[0]:
            e = torch.tensor([], requires_grad=True)
            if data[0].is_cuda:
                e.cuda()
            return e

        # print(primaries)
        
        # obtain vertex features
        x = cluster_vtx_features(data[0], clusts, device=device)
        # obtain edge features
        e = cluster_edge_features(data[0], clusts, edge_index, device=device)
        # get x batch
        xbatch = torch.tensor(batch).to(device)
        
        # print(x.shape)
        # print(torch.max(edge_index))
        
        # get output
        out = self.edge_predictor(x, edge_index, e, xbatch)
        
        return out
    
    
    
class EdgeChannelLoss(torch.nn.Module):
    """
    Edge loss based on two channel output
    """
    def __init__(self, cfg):
        # torch.nn.MSELoss(reduction='sum')
        # torch.nn.L1Loss(reduction='sum')
        super(EdgeChannelLoss, self).__init__()
        self.model_config = cfg['modules']['edge_model']
        
        self.remove_compton = self.model_config.get('remove_compton', True)
        self.compton_thresh = self.model_config.get('compton_thresh', 30)
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
        
        
    def forward(self, out, clusters, groups, primary):
        """
        out:
            array output from the DataParallel gather function
            out['edge_pred'] - n_gpus tensors of predicted edge weights from model forward
        data:
            cluster_labels - n_gpus Nx5 tensors of (x, y, z, batch_id, cluster_id)
            group_labels - n_gpus Nx5 tensors of (x, y, z, batch_id, group_id) 
            em_primaries - n_gpus tensor of (x, y, z) coordinates of origins of EM primaries
        """
        edge_ct = 0
        total_loss, total_acc, total_primary_fdr, total_primary_acc = 0., 0., 0., 0.
        ari, ami, sbd, pur, eff = 0., 0., 0., 0., 0.
        ngpus = len(clusters)
        for i in range(ngpus):
            edge_pred = out['edge_pred'][i]
            data0 = clusters[i]
            data1 = groups[i]
            data2 = primary[i]
            
            device = data0.device

            # first decide what true edges should be
            # need to form graph, then pass through GNN
            # clusts = form_clusters(data0)
            clusts = form_clusters_new(data0)

            # remove compton clusters
            # if no cluster fits this condition, return
            if self.remove_compton:
                selection = filter_compton(clusts) # non-compton looking clusters
                if not len(selection):
                    total_loss += self.lossfn(edge_pred, edge_pred)
                    total_acc += 1.
                    continue

                clusts = clusts[selection]

            # process group data
            # data_grp = process_group_data(data1, data0)
            data_grp = data1

            # form primary/secondary bipartite graph
            primaries = assign_primaries(data2, clusts, data0, max_dist=self.pmd)
            batch = get_cluster_batch(data0, clusts)
            edge_index = primary_bipartite_incidence(batch, primaries)
            if not edge_index.shape[0]:
                total_loss += self.lossfn(edge_pred, edge_pred)
                total_acc += 1.
                continue

            group = get_cluster_label(data_grp, clusts)

            primaries_true = assign_primaries(data2, clusts, data1, use_labels=True)
            primary_fdr, primary_tdr, primary_acc = analyze_primaries(primaries, primaries_true)
            total_primary_fdr += primary_fdr
            total_primary_acc += primary_acc

            # determine true assignments
            edge_assn = edge_assignment(edge_index, batch, group, device=device, dtype=torch.long)

            edge_assn = edge_assn.view(-1)

            total_loss += self.lossfn(edge_pred, edge_assn)

            # compute accuracy of assignment
            # need to multiply by batch size to be accurate
            total_acc += torch.tensor(
                secondary_matching_vox_efficiency3(
                    edge_index,
                    edge_assn,
                    edge_pred,
                    primaries,
                    clusts,
                    len(clusts)
                )
            )
            
            ari0, ami0, sbd0, pur0, eff0 = DBSCAN_cluster_metrics(
                edge_index,
                edge_assn,
                edge_pred,
                primaries,
                clusts,
                len(clusts)
            )
            ari += ari0
            ami += ami0
            sbd += sbd0
            pur += pur0
            eff += eff0
            
            edge_ct += edge_index.shape[1]

        return {
            'primary_fdr': total_primary_fdr/ngpus,
            'primary_acc': total_primary_acc/ngpus,
            'ARI': ari/ngpus,
            'AMI': ami/ngpus,
            'SBD': sbd/ngpus,
            'purity': pur/ngpus,
            'efficiency': eff/ngpus,
            'accuracy': total_acc/ngpus,
            'loss': total_loss/ngpus,
            'edge_count': edge_ct
        }
    