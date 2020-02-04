# GNN that attempts to predict primary clusters
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import numpy as np
from .gnn import node_model_construct, node_encoder_construct, edge_encoder_construct
from mlreco.utils.gnn.cluster import form_clusters, reform_clusters, get_cluster_batch, get_cluster_label, get_cluster_group, get_cluster_primary
from mlreco.utils.gnn.network import complete_graph, delaunay_graph, mst_graph, bipartite_graph, inter_cluster_distance
from mlreco.utils.gnn.data import cluster_vtx_features, cluster_edge_features

class ClustNodeGNN(torch.nn.Module):
    """
    Driver class for cluster node prediction, assumed to be a GNN model.
    This class mostly acts as a wrapper that will hand the graph data to another model.
    If DBSCAN is used, use the semantic label tensor as an input.

    For use in config:
    model:
      name: cluster_gnn
      modules:
        chain:
          node_type       : <semantic class to group (all classes if -1, default 0, i.e. EM)>
          node_min_size   : <minimum number of voxels inside a cluster to be considered (default -1)>
          network         : <type of network: 'complete', 'delaunay', 'mst' or 'bipartite' (default 'complete')>
          edge_max_dist   : <maximal edge Euclidean length (default -1)>
          edge_dist_method: <edge length evaluation method: 'centroid' or 'set' (default 'set')>
        dbscan:
          <dictionary of dbscan parameters>
        node_encoder:
          name: <name of the node encoder>
          <dictionary of arguments to pass to the encoder>
        edge_encoder:
          name: <name of the edge encoder>
          <dictionary of arguments to pass to the encoder>
        edge_model:
          name: <name of the edge model>
          <dictionary of arguments to pass to the model>
          model_path      : <path to the model weights>
    """
    def __init__(self, cfg):
        super(ClustNodeGNN, self).__init__()

        # Get the chain input parameters 
        chain_config = cfg['modules']['chain']

        # Choose what type of node to use
        self.node_type = chain_config.get('node_type', 0)
        self.node_min_size = chain_config.get('node_min_size', -1)

        # Choose what type of network to use
        self.network = chain_config.get('network', 'complete')
        self.edge_max_dist = chain_config.get('edge_max_dist', -1)
        self.edge_dist_metric = chain_config.get('edge_dist_metric','set')

        # If requested, use DBSCAN to form clusters from semantics
        self.do_dbscan = False
        if 'dbscan' in cfg['modules']:
            self.do_dbscan = True
            self.dbscan = DBScanClusts2(cfg)

        # Initialize encoders
        self.node_encoder = node_encoder_construct(cfg)
        self.edge_encoder = edge_encoder_construct(cfg)
            
        # Construct the model
        self.node_predictor = node_model_construct(cfg)

    @staticmethod
    def default_return(device):
        """
        Default return when no valid node is found in the input data.

        Args:
            device (torch.device): Device on which the input is stored
        Returns:
            dict:
                'node_pred' (torch.tennsor): (0,2) Empty two-channel node predictions
                'clusts' ([np.ndarray])    : (0) Empty cluster ids
                'batch_ids' (np.ndarray)   : (0) Empty cluster batch ids
                'edge_index' (np.ndarray)  : (2,0) Empty incidence matrix
        """
        xg = torch.empty((0,2), requires_grad=True, device=device)
        x  = np.empty(0)
        e  = np.empty((2,0))
        return {'node_pred':[xg], 'clusts':[x], 'batch_ids':[x], 'edge_index':[e]}

    def forward(self, data):
        """
        Prepares particle clusters and feed them to the GNN model.

        Args:
            data ([torch.tensor]): (N,8) [x, y, z, batchid, value, id, groupid, shape]
        Returns:
            dict:
                'node_pred' (torch.tensor): (E,2) Two-channel node predictions
                'clusts' ([np.ndarray])   : [(N_0), (N_1), ..., (N_C)] Cluster ids
                'batch_ids' (np.ndarray)  : (C) Cluster batch ids
                'edge_index' (np.ndarray)  : (2,0) Empty incidence matrix
        """
        # Find index of points that belong to the same clusters
        # If a specific semantic class is required, apply mask
        # Here the specified size selection is applied
        data = data[0]
        device = data.device
        if self.do_dbscan:
            clusts = self.dbscan(data, onehot=False)
            if self.node_type > -1:
                mask = np.where(data[:,-1] == 0)[0]
                clusts = clusts[self.node_type]
                clusts = [mask[c] for c in clusts]
            else:
                clusts = np.concatenate(clusts).tolist()
        else:
            if self.node_type > -1:
                mask = np.where(data[:,-1] == 0)[0]
                clusts = form_clusters(data[mask], self.node_min_size)
                clusts = [mask[c] for c in clusts]
            else:
                clusts = form_clusters(data, self.node_min_size)

        if not len(clusts):
            return self.default_return(device)

        # Get the batch id for each cluster
        batch_ids = get_cluster_batch(data, clusts)

        # Compute the cluster distance matrix, if necessary
        dist_mat = None
        if self.edge_max_dist > 0 or self.network == 'mst':
            dist_mat = inter_cluster_distance(data[:,:3], clusts, self.edge_dist_metric)

        # Form the requested network
        if len(clusts) == 1:
            edge_index = np.empty((2,0))
        elif self.network == 'complete':
            edge_index = complete_graph(batch_ids, dist_mat, self.edge_max_dist)
        elif self.network == 'delaunay':
            edge_index = delaunay_graph(data, clusts, dist_mat, self.edge_max_dist)
        elif self.network == 'mst':
            edge_index = mst_graph(batch_ids, dist_mat, self.edge_max_dist)
        elif self.network == 'bipartite':
            group_ids = get_cluster_group(data, clusts)
            primary_ids = get_cluster_primary(clust_ids, group_ids)
            edge_index = bipartite_graph(batch_ids, primary_ids, dist_mat, self.edge_max_dist)
        else:
            raise ValueError('Network type not recognized: '+self.network)

        # Skip if there is no edges
        if not edge_index.shape[1]:
            return self.default_return(device)

        # Obtain node and edge features
        x = self.node_encoder(data, clusts)
        e = self.edge_encoder(data, clusts, edge_index)

        # Bring edge_index and batch_ids to device
        index = torch.tensor(edge_index, device=device, dtype=torch.long)
        xbatch = torch.tensor(batch_ids, device=device)

        # Pass through the model, get output (long edge_index)
        out = self.node_predictor(x, index, e, xbatch) 

        return {**out,
                'clusts':[clusts],
                'batch_ids':[batch_ids],
                'edge_index':[edge_index]}


class NodeChannelLoss(torch.nn.Module):
    """
    Takes the output of EdgeModel and computes the channel-loss.

    For use in config:
    model:
      name: cluster_gnn
      modules:
        chain:
          loss            : <loss function: 'CE' or 'MM' (default 'CE')>
          reduction       : <loss reduction method: 'mean' or 'sum' (default 'sum')>
          balance_classes : <balance loss per class: True or False (default False)>
          target_photons  : <use true photon connections as basis for loss (default False)>
    """
    def __init__(self, cfg):
        super(NodeChannelLoss, self).__init__()

        # Get the chain input parameters 
        chain_config = cfg['modules']['chain']

        # Set the loss
        self.loss = chain_config.get('loss', 'CE')
        self.reduction = chain_config.get('reduction', 'mean')
        self.balance_classes = chain_config.get('balance_classes', False)

        if self.loss == 'CE':
            self.lossfn = torch.nn.CrossEntropyLoss(reduction=self.reduction)
        elif self.loss == 'MM':
            p = chain_config.get('p', 1)
            margin = chain_config.get('margin', 1.0)
            self.lossfn = torch.nn.MultiMarginLoss(p=p, margin=margin, reduction=self.reduction)
        else:
            raise Exception('Loss not recognized: ' + self.loss)

    def forward(self, out, clusters):
        """
        Applies the requested loss on the node prediction. 

        Args:
            out (dict):
                'node_pred' (torch.tensor): (C,2) Two-channel node predictions
                'clusts' ([np.ndarray])   : [(N_0), (N_1), ..., (N_C)] Cluster ids
            clusters ([torch.tensor])     : (N,8) [x, y, z, batchid, value, id, groupid, shape]
        Returns:
            double: loss, accuracy, clustering metrics
        """
        total_loss, total_acc = 0., 0.
        ngpus = len(clusters)
        for i in range(len(clusters)):

            # If the input did not have any node, proceed
            if 'node_pred' not in out:
                if ngpus == 1:
                    total_loss = torch.tensor(0., requires_grad=True, device=clusters[i].device)
                ngpus = max(1, ngpus-1)
                continue

            # Get list of IDs of points contained in each cluster
            clusts = out['clusts'][i]

            # Use the primary information to determine the true node assignment
            node_pred = out['node_pred'][i]
            primary_ids = []
            for c in clusts:
                primary = (clusters[i][c, -3]==clusters[i][c,-2]).any() # If any point is primary, assign cluster to primary
                primary_ids.append(primary)

            node_assn = torch.tensor(primary_ids, dtype=torch.long, device=node_pred.device, requires_grad=False)

            # Increment the loss, balance classes if requested
            if self.balance_classes:
                counts = torch.unique(node_assn, return_counts=True)[1]
                weights = np.array([float(counts[k])/len(node_assn) for k in range(2)])
                for k in range(2):
                    total_loss += (1./weights[k])*self.lossfn(node_pred[node_assn==k], node_assn[node_assn==k])
            else:
                total_loss += self.lossfn(node_pred, node_assn)

            # Compute accuracy of assignment (fraction of correctly assigned nodes)
            total_acc += torch.sum(torch.argmax(node_pred, dim=1) == node_assn).float()/len(clusts)

        return {
            'accuracy': total_acc/ngpus,
            'loss': total_loss/ngpus
        }

