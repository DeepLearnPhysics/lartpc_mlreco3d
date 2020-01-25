# GNN that attempts to put clusters together into groups
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import numpy as np
from .gnn import edge_model_construct
from mlreco.utils.gnn.cluster import form_clusters, reform_clusters, get_cluster_batch, get_cluster_label, get_cluster_group, get_cluster_primary
from mlreco.utils.gnn.network import complete_graph, delaunay_graph, mst_graph, bipartite_graph, inter_cluster_distance, get_fragment_edges
from mlreco.utils.gnn.data import cluster_vtx_features, cluster_edge_features
from mlreco.utils.gnn.evaluation import edge_assignment, edge_assignment_from_graph, node_assignment, node_assignment_group, clustering_metrics

class EdgeModel(torch.nn.Module):
    """
    Driver class for edge prediction, assumed to be a GNN model.
    This class mostly acts as a wrapper that will hand the graph data to another model.

    For use in config:
    model:
      name: cluster_gnn
      modules:
        edge_model:
          name: <name of the edge model>
          model_cfg:
            <dictionary of arguments to pass to the model>
          node_type       : <semantic class to group (all classes if -1, default 0, i.e. EM)>
          node_min_size   : <minimum number of voxels inside a cluster to be considered (default -1)>
          node_encoder    : <node feature encoding: 'basic' or 'cnn' (default 'basic')>
          network         : <type of network: 'complete', 'delaunay', 'mst' or 'bipartite' (default 'complete')>
          edge_max_dist   : <maximal edge Euclidean length (default -1)>
          edge_dist_method: <edge length evaluation method: 'centroid' or 'set' (default 'set')>
          model_path      : <path to the model weights>
    """
    def __init__(self, cfg):
        super(EdgeModel, self).__init__()

        # Get the model input parameters 
        if 'modules' in cfg:
            self.model_config = cfg['modules']['edge_model']
        else:
            self.model_config = cfg

        # Choose what type of node to use
        self.node_type = self.model_config.get('node_type', 0)
        self.node_min_size = self.model_config.get('node_min_size', -1)
        self.node_encoder = self.model_config.get('node_encoder', 'basic')

        # Choose what type of network to use
        self.network = self.model_config.get('network', 'complete')
        self.edge_max_dist = self.model_config.get('edge_max_dist', -1)
        self.edge_dist_metric = self.model_config.get('edge_dist_metric','set')
            
        # Extract the model to use
        edge_model = edge_model_construct(self.model_config.get('name'))

        # Construct the model
        self.edge_predictor = edge_model(self.model_config.get('model_cfg'))

    @staticmethod
    def default_return(device):
        """
        Default return when no valid node is found in the input data.

        Args:
            device (torch.device): Device on which the input is stored
        Returns:
            dict:
                'edge_pred' (torch.tensor): (0,2) Empty two-channel edge predictions
                'clust_ids' (np.ndarray)  : (0) Empty cluster ids
                'batch_ids' (np.ndarray)  : (0) Empty cluster batch ids
                'edge_index' (np.ndarray) : (2,0) Empty incidence matrix
        """
        xg = torch.empty((0,2), requires_grad=True, device=device)
        x  = np.empty(0)
        e  = np.empty((2,0))
        return {'edge_pred':[xg], 'clust_ids':[x], 'batch_ids':[x], 'edge_index':[e]}

    def forward(self, data):
        """
        Prepares particle clusters and feed them to the GNN model.

        Args:
            data ([torch.tensor]): (N,8) [x, y, z, batchid, value, id, groupid, shape]
        Returns:
            dict:
                'edge_pred' (torch.tensor): (E,2) Two-channel edge predictions
                'clust_ids' (np.ndarray)  : (C) Cluster ids
                'batch_ids' (np.ndarray)  : (C) Cluster batch ids
                'edge_index' (np.ndarray) : (2,E) Incidence matrix
        """
        # Get original device, bring data to CPU (for data preparation)
        device = data[0].device
        cluster_label = data[0].detach().cpu().numpy()

        # Find index of points that belong to the same clusters
        # If a specific semantic class is required, apply mask
        # Here the specified size selection is applied
        if self.node_type > -1:
            mask = np.where(cluster_label[:,-1] == 0)[0]
            clusts = form_clusters(cluster_label[mask], self.node_min_size)
            clusts = np.array([mask[c] for c in clusts])
        else:
            clusts = form_clusters(cluster_label, self.node_min_size)

        # Get the batch, cluster and group id of each cluster
        batch_ids = get_cluster_batch(cluster_label, clusts)
        clust_ids = get_cluster_label(cluster_label, clusts)

        # Form the requested network
        dist_mat = None
        if self.edge_max_dist > 0 or self.network == 'mst':
            dist_mat = inter_cluster_distance(cluster_label[:,:3], clusts, self.edge_dist_metric)
        if self.network == 'complete':
            edge_index = complete_graph(batch_ids, dist_mat, self.edge_max_dist)
        elif self.network == 'delaunay':
            edge_index = delaunay_graph(cluster_label, clusts, dist_mat, self.edge_max_dist)
        elif self.network == 'mst':
            edge_index = mst_graph(batch_ids, dist_mat, self.edge_max_dist)
        elif self.network == 'bipartite':
            group_ids = get_cluster_group(cluster_label, clusts)
            primary_ids = get_cluster_primary(clust_ids, group_ids)
            edge_index = bipartite_graph(batch_ids, primary_ids, dist_mat, self.edge_max_dist)
        else:
            raise ValueError('Network type not recognized: '+self.network)

        # Skip if there is no edges (Is this necessary ? TODO)
        if not edge_index.shape[1]:
            return self.default_return(device)

        # Obtain node and edge features
        if self.node_encoder == 'basic':
            x = torch.tensor(cluster_vtx_features(cluster_label, clusts), device=device, dtype=torch.float)
        elif self.node_encoder == 'cnn':
            raise NotImplementedError('CNN encoder not yet implemented...')
        else:
            raise ValueError('Node encoder not recognized: '+self.node_encoding) 

        e = torch.tensor(cluster_edge_features(cluster_label, clusts, edge_index), device=device, dtype=torch.float)

        # Bring edge_index and batch_ids to device
        index = torch.tensor(edge_index, device=device, dtype=torch.long)
        xbatch = torch.tensor(batch_ids, device=device)

        # Pass through the model, get output (long edge_index)
        out = self.edge_predictor(x, index, e, xbatch) 

        return {**out,
                'clust_ids':[clust_ids],
                'batch_ids':[batch_ids],
                'edge_index':[edge_index]}


class EdgeChannelLoss(torch.nn.Module):
    """
    Takes the output of EdgeModel and computes the channel-loss.

    For use in config:
    model:
      name: cluster_gnn
      modules:
        edge_model:
          loss            : <loss function: 'CE' or 'MM' (default 'CE')>
          reduction       : <loss reduction method: 'mean' or 'sum' (default 'sum')>
          balance_classes : <balance loss per class: True or False (default False)>
          target_photons  : <use true photon connections as basis for loss (default False)>
    """
    def __init__(self, cfg):
        super(EdgeChannelLoss, self).__init__()

        # Get the model loss parameters
        if 'modules' in cfg:
            self.model_config = cfg['modules']['edge_model']
        else:
            self.model_config = cfg

        # Set the loss
        self.loss = self.model_config.get('loss', 'CE')
        self.reduction = self.model_config.get('reduction', 'mean')
        self.balance_classes = self.model_config.get('balance_classes', False)
        self.target_photons = self.model_config.get('target_photons', False)

        if self.loss == 'CE':
            self.lossfn = torch.nn.CrossEntropyLoss(reduction=self.reduction)
        elif self.loss == 'MM':
            p = self.model_config.get('p', 1)
            margin = self.model_config.get('margin', 1.0)
            self.lossfn = torch.nn.MultiMarginLoss(p=p, margin=margin, reduction=self.reduction)
        else:
            raise Exception('Loss not recognized: ' + self.loss)

    def forward(self, out, clusters, graph):
        """
        Applies the requested loss on the edge prediction. 

        Args:
            out (dict):
                'edge_pred' (torch.tensor): (E,2) Two-channel edge predictions
                'clust_ids' (np.ndarray)  : (C) Cluster ids
                'batch_ids' (np.ndarray)  : (C) Cluster batch ids
                'edge_index' (np.ndarray) : (2,E) Incidence matrix
            clusters ([torch.tensor])     : (N,8) [x, y, z, batchid, value, id, groupid, shape]
            graph ([torch.tensor])        : (N,3) True edges 
        Returns:
            double: loss, accuracy, clustering metrics
        """
        total_loss, total_acc = 0., 0.
        total_ari, total_ami, total_sbd, total_pur, total_eff = 0., 0., 0., 0., 0.
        ngpus = len(clusters)
        out['group_ids'] = []
        for i in range(len(clusters)):

            # If the input did not have any node, proceed
            if not len(out['clust_ids'][i]):
                if ngpus == 1:
                    total_loss = torch.tensor(0., requires_grad=True, device=edge_pred.device)
                ngpus = max(1, ngpus-1)
                continue

            # Get list of IDs of points contained in each cluster
            cluster_label = clusters[i].detach().cpu().numpy()
            clust_ids = out['clust_ids'][i]
            batch_ids = out['batch_ids'][i]
            clusts = reform_clusters(cluster_label, clust_ids, batch_ids)

            # Use group information or particle tree to determine the true edge assigment
            edge_pred = out['edge_pred'][i]
            edge_index = out['edge_index'][i]
            group_ids = get_cluster_group(cluster_label, clusts)
            out['group_ids'].append(group_ids)
            if not self.target_photons:
                edge_assn = edge_assignment(edge_index, batch_ids, group_ids)
            else:
                graph = graph[i].detach().cpu().numpy()
                true_edge_index = get_fragment_edges(graph, clust_ids, batch_ids)
                edge_assn = edge_assignment_from_graph(edge_index, true_edge_undex)

            edge_assn = torch.tensor(edge_assn, device=edge_pred.device, dtype=torch.long, requires_grad=False).view(-1)

            # Increment the loss, balance classes if requested
            if self.balance_classes:
                counts = torch.unique(edge_assn, return_counts=True)[1]
                weights = np.array([float(counts[k])/len(edge_assn) for k in range(2)])
                for k in range(2):
                    total_loss += (1./weights[k])*self.lossfn(edge_pred[edge_assn==k], edge_assn[edge_assn==k])
            else:
                total_loss += self.lossfn(edge_pred, edge_assn)

            # Compute accuracy of assignment (fraction of correctly assigned edges)
            total_acc += torch.sum(torch.argmax(edge_pred, dim=1) == edge_assn).float()/edge_assn.shape[0]

            # Compute clustering accuracy metrics
            node_assn = node_assignment_group(group_ids, batch_ids)
            node_pred = node_assignment(edge_index, torch.argmax(edge_pred, dim=1).detach().cpu().numpy(), len(clust_ids))
            ari, ami, sbd, pur, eff = clustering_metrics(clusts, node_assn, node_pred)
            total_ari += ari
            total_ami += ami
            total_sbd += sbd
            total_pur += pur
            total_eff += eff

        return {
            'ARI': total_ari/ngpus,
            'AMI': total_ami/ngpus,
            'SBD': total_sbd/ngpus,
            'purity': total_pur/ngpus,
            'efficiency': total_eff/ngpus,
            'accuracy': total_acc/ngpus,
            'loss': total_loss/ngpus,
        }

