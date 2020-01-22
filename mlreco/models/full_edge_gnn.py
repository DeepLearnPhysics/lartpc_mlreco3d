# GNN that attempts to match clusters to groups
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import numpy as np
from mlreco.utils.gnn.cluster import get_cluster_batch, get_cluster_label, get_cluster_group, form_clusters_new
from mlreco.utils.gnn.network import complete_graph, delaunay_graph, mst_graph, get_fragment_edges
from mlreco.utils.gnn.compton import filter_compton
from mlreco.utils.gnn.data import cluster_vtx_features, cluster_edge_features, edge_assignment
from mlreco.utils.gnn.evaluation import DBSCAN_cluster_metrics3
from .gnn import edge_model_construct

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
                compton_threshold: Minimum number of voxels
                balance_classes: <True/False for loss computation> (default False)
                loss: 'CE' or 'MM' (default 'CE')
    """
    def __init__(self, cfg):
        super(FullEdgeModel, self).__init__()

        # Get the model input parameters 
        if 'modules' in cfg:
            self.model_config = cfg['modules']['edge_model']
        else:
            self.model_config = cfg

        self.remove_compton = self.model_config.get('remove_compton', True)
        self.compton_thresh = self.model_config.get('compton_thresh', 30)

        # Choose what type of network to use
        self.network = self.model_config.get('network', 'complete')
        self.edge_max_dist = self.model_config.get('edge_max_dist', -1)
        self.edge_dist_metric = self.model_config.get('edge_dist_metric','set')
            
        # Extract the model to use
        model = edge_model_construct(self.model_config.get('name', 'edge_only'))

        # Construct the model
        self.edge_predictor = model(self.model_config.get('model_cfg', {}))

    @staticmethod
    def default_return(device):
        """
        Default forward return if the graph is empty (no node)
        """
        xg = torch.tensor([], requires_grad=True)
        x  = torch.tensor([])
        x.to(device)
        return {'edge_pred':[xg], 'clust_ids':[x], 'group_ids':[x], 'batch_ids':[x], 'edge_index':[x]}

    def forward(self, data):
        """
        inputs data:
            data[0]: (Nx8) Cluster tensor with row (x, y, z, batch_id, voxel_val, cluster_id, group_id, sem_type)
        output data:
            dictionary, with
                'edge_pred': torch.tensor with edge prediction weights
                'clust_ids': torch.tensor with cluster ids
                'group_ids': torch.tensor with cluster group ids
                'batch_ids': torch.tensor with cluster batch ids
                'edge_index': 2xn tensor of edges in the bipartite graph
        """
        # Get device
        cluster_label = data[0]
        device = cluster_label.device

        # Mask out the energy depositions that are not EM
        em_mask = np.where(cluster_label[:,-1] == 0)[0]

        # Find index of points that belong to the same EM clusters
        clusts = form_clusters_new(cluster_label[em_mask])
        clusts = np.array([em_mask[c] for c in clusts])

        # If requested, remove clusters below a certain size threshold
        if self.remove_compton:
            selection = np.where(filter_compton(clusts, self.compton_thresh))[0]
            if not len(selection):
                return self.default_return(device)
            clusts = clusts[selection]

        # Get the cluster id of each cluster
        clust_ids = get_cluster_label(cluster_label, clusts)

        # Get the group id of each cluster
        group_ids = get_cluster_group(cluster_label, clusts)

        # Get the batch id of each cluster
        batch_ids = get_cluster_batch(cluster_label, clusts)

        # Form the requested network 
        dist_mat = None
        if self.edge_max_dist > 0:
            dist_mat = inter_cluster_distance(cluster_label[:,:3], clusts, self.edge_dist_metric) 
        if self.network == 'complete':
            edge_index = complete_graph(batch_ids, dist_mat, self.edge_max_dist, device)
        elif self.network == 'delaunay':
            mask = np.hstack(clusts)
            labels = np.hstack([np.full(len(c), i) for i, c in enumerate(clusts)])
            edge_index = delaunay_graph(cluster_label[mask], labels, dist_mat, self.edge_max_dist, device)
        elif self.network == 'mst':
            if dist_mat is None:
                dist_mat = inter_cluster_distance(cluster_label[:,:3], clusts, self.edge_dist_metric) 
            edge_index = mst_graph(batch_ids, dist_mat, self.edge_max_dist, device)

        # Skip if there is no edges (Is this necessary ? TODO)
        if not edge_index.shape[1]:
            return self.default_return(device)

        # Obtain vertex features
        x = cluster_vtx_features(cluster_label, clusts, device=device)

        # Obtain edge features
        e = cluster_edge_features(cluster_label, clusts, edge_index, device=device)

        # Convert the the batch IDs to a torch tensor to pass to torch
        batch_ids = torch.tensor(batch_ids).to(device)
        
        # Pass through the model, get output
        out = self.edge_predictor(x, edge_index, e, batch_ids)

        return {**out,
                'clust_ids':[torch.tensor(clust_ids).to(device)],
                'group_ids':[torch.tensor(group_ids).to(device)],
                'batch_ids':[batch_ids],
                'edge_index':[edge_index]}


class FullEdgeChannelLoss(torch.nn.Module):
    """
    Edge loss based on two channel output
    """
    def __init__(self, cfg):
        super(FullEdgeChannelLoss, self).__init__()

        # Get the model loss parameters
        if 'modules' in cfg:
            self.model_config = cfg['modules']['edge_model']
        else:
            self.model_config = cfg

        self.reduction = self.model_config.get('reduction', 'mean')
        self.loss = self.model_config.get('loss', 'CE')
        self.balance_classes = self.model_config.get('balance_classes', False)

        if self.loss == 'CE':
            self.lossfn = torch.nn.CrossEntropyLoss(reduction=self.reduction)
        elif self.loss == 'MM':
            p = self.model_config.get('p', 1)
            margin = self.model_config.get('margin', 1.0)
            self.lossfn = torch.nn.MultiMarginLoss(p=p, margin=margin, reduction=self.reduction)
        else:
            raise Exception('unrecognized loss: ' + self.loss)

    def forward(self, out, clusters, graph):
        """
        out:
            dictionary output from the DataParallel gather function
            out['edge_pred'] - n_gpus tensors of predicted edge weights from model forward
        data:
            clusters: n_gpus (Nx8) Cluster tensor with row (x, y, z, batch_id, voxel_val, cluster_id, group_id, sem_type)
        """
        edge_ct = 0
        total_loss, total_acc = 0., 0.
        total_ari, total_ami, total_sbd, total_pur, total_eff = 0., 0., 0., 0., 0.
        ngpus = len(clusters)
        for i in range(len(clusters)):

            # Get the necessary data products
            clust_label = clusters[i]
            edge_pred = out['edge_pred'][i]
            clust_ids = out['clust_ids'][i]
            group_ids = out['group_ids'][i]
            batch_ids = out['batch_ids'][i]
            edge_index = out['edge_index'][i]
            device = edge_pred.device
            if not len(clust_ids):
                if ngpus > 1:
                    ngpus -= 1
                continue
            
            # Get list of IDs of points contained in each cluster
            clusts = np.array([torch.nonzero((clust_label[:,3] == batch_ids[j]) & (clust_label[:,5] == clust_ids[j])).reshape(-1).cpu().numpy() for j in range(len(batch_ids))])

            # Get the list of true edges
            true_edge_index = get_fragment_edges(graph[i], clust_ids, batch_ids)

            # Use group information to determine the true edge assigment 
            edge_assn = edge_assignment(edge_index, batch_ids, group_ids, device=device, dtype=torch.long)
            #edge_assn = torch.tensor([np.any([(e == pair).all() for pair in true_edge_index]) for e in edge_index.transpose(0,1).cpu().numpy()], dtype=torch.long)
            edge_assn = edge_assn.view(-1)

            # Increment the loss, balance classes if requested
            if self.balance_classes:
                counts = np.unique(edge_assn, return_counts=True)[1]
                weights = np.array([float(counts[k])/len(edge_assn) for k in range(2)])
                for k in range(2):
                    total_loss += (1./weights[k])*self.lossfn(edge_pred[edge_assn==k], edge_assn[edge_assn==k])
            else:
                total_loss += self.lossfn(edge_pred, edge_assn)

            # Compute accuracy of assignment
            total_acc += torch.sum(torch.argmax(edge_pred, dim=1) == edge_assn).float()/edge_assn.shape[0]

            ari, ami, sbd, pur, eff = DBSCAN_cluster_metrics3(
                edge_index,
                edge_assn,
                torch.argmax(edge_pred, dim=1),
                clusts
            )
            total_ari += ari
            total_ami += ami
            total_sbd += sbd
            total_pur += pur
            total_eff += eff

            edge_ct += edge_index.shape[1]

        return {
            'ARI': total_ari/ngpus,
            'AMI': total_ami/ngpus,
            'SBD': total_sbd/ngpus,
            'purity': total_pur/ngpus,
            'efficiency': total_eff/ngpus,
            'accuracy': total_acc/ngpus,
            'loss': total_loss/ngpus,
            'edge_count': edge_ct
        }

