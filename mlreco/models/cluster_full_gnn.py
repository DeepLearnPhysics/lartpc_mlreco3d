# GNN that attempts to put clusters together into groups
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import numpy as np
from .gnn import node_model_construct, edge_model_construct, node_encoder_construct, edge_encoder_construct
from .layers.dbscan import DBScanClusts2
from mlreco.utils.gnn.cluster import form_clusters, get_cluster_batch, get_cluster_label, relabel_groups, get_cluster_points_label, get_cluster_directions
from mlreco.utils.gnn.network import complete_graph, delaunay_graph, mst_graph, bipartite_graph, inter_cluster_distance, get_fragment_edges
from mlreco.utils.gnn.evaluation import edge_assignment, edge_assignment_from_graph, node_assignment, node_assignment_score
from mlreco.utils import local_cdist
from mlreco.models.cluster_node_gnn import NodeChannelLoss
from mlreco.models.cluster_gnn import EdgeChannelLoss

class ClustFullGNN(torch.nn.Module):
    """
    Driver class for cluster node+edge prediction, assumed to be a GNN model.
    This class mostly acts as a wrapper that will hand the graph data to another model.
    If DBSCAN is used, use the semantic label tensor as an input.

    For use in config:
    model:
      name: cluster_hierachy_gnn
      modules:
        chain:
          node_type         : <semantic class to group (all classes if -1, default 0, i.e. EM)>
          node_min_size     : <minimum number of voxels inside a cluster to be considered (default -1)>
          add_start_point   : <add label start point to the node features (default False)
          add_start_dir     : <add predicted start direction to the node features (default False)
          start_dir_max_dist: <maximium distance between start point and cluster voxels to be used to estimate direction (default -1, i.e no limit)>
          network           : <type of node prediction network: 'complete', 'delaunay' or 'mst' (default 'complete')>
          edge_max_dist     : <maximal edge Euclidean length (default -1)>
          edge_dist_method  : <edge length evaluation method: 'centroid' or 'set' (default 'set')>
          edge_dist_numpy   : <use numpy to compute inter cluster distance (default False)>
        dbscan:
          <dictionary of dbscan parameters>
        node_encoder:
          name: <name of the node encoder>
          <dictionary of arguments to pass to the encoder>
          model_path      : <path to the encoder weights>
        edge_encoder:
          name: <name of the edge encoder>
          <dictionary of arguments to pass to the encoder>
          model_path      : <path to the encoder weights>
        node_model:
          name: <name of the node model>
          <dictionary of arguments to pass to the model>
          model_path      : <path to the model weights>
        edge_model:
          name: <name of the edge model>
          <dictionary of arguments to pass to the model>
          model_path      : <path to the model weights>
    """

    MODULES = ['chain', 'dbscan', 'node_encoder', 'edge_encoder', 'node_model', 'edge_model']

    def __init__(self, cfg):
        super(ClustFullGNN, self).__init__()

        # Get the chain input parameters
        chain_config = cfg['chain']

        # Choose what type of node to use
        self.node_type = chain_config.get('node_type', 0)
        self.node_min_size = chain_config.get('node_min_size', -1)
        self.add_start_point = chain_config.get('add_start_point', False)
        self.add_start_dir = chain_config.get('add_start_dir', False)
        self.start_dir_max_dist = chain_config.get('start_dir_max_dist', -1)

        # Choose what type of network to use
        self.network = chain_config.get('network', 'complete')
        self.edge_max_dist = chain_config.get('edge_max_dist', -1)
        self.edge_dist_metric = chain_config.get('edge_dist_metric', 'set')
        self.edge_dist_numpy = chain_config.get('edge_dist_numpy',False)
        self.group_pred = chain_config.get('group_pred','score')

        # If requested, use DBSCAN to form clusters from semantics
        self.do_dbscan = False
        if 'dbscan' in cfg:
            self.do_dbscan = True
            self.dbscan = DBScanClusts2(cfg)

        # Initialize encoders
        self.node_encoder = node_encoder_construct(cfg)
        self.edge_encoder = edge_encoder_construct(cfg)

        # Construct the models
        self.edge_predictor = edge_model_construct(cfg)

    def forward(self, data):
        """
        Prepares particle clusters and feed them to the GNN model.

        Args:
            data ([torch.tensor]): (N,5-6) [x, y, z, batchid, (value,) id]
        Returns:
            dict:
                'node_pred' (torch.tensor): (N,2) Two-channel node predictions
                'edge_pred' (torch.tensor): (E,2) Two-channel edge predictions
                'clusts' ([np.ndarray])   : [(N_0), (N_1), ..., (N_C)] Cluster ids
                'edge_index' (np.ndarray) : (E,2) Incidence matrix
        """
        # Find index of points that belong to the same clusters
        # If a specific semantic class is required, apply mask
        # Here the specified size selection is applied
        if len(data) > 1:
            particles = data[1]
        data = data[0]
        device = data.device
        result = {}
        if self.do_dbscan:
            clusts = self.dbscan(data, onehot=False)
            if self.node_type > -1:
                clusts = clusts[self.node_type]
            else:
                clusts = np.concatenate(clusts).tolist()
        else:
            if self.node_type > -1:
                mask = torch.nonzero(data[:,-1] == self.node_type).flatten()
                clusts = form_clusters(data[mask], self.node_min_size)
                clusts = [mask[c].cpu().numpy() for c in clusts]
            else:
                clusts = form_clusters(data, self.node_min_size)
                clusts = [c.cpu().numpy() for c in clusts]

        if not len(clusts):
            return result

        # Get the batch id for each cluster
        batch_ids = get_cluster_batch(data, clusts)

        # Compute the cluster distance matrix, if necessary
        dist_mat = None
        if self.edge_max_dist > 0 or self.network == 'mst':
            dist_mat = inter_cluster_distance(data[:,:3], clusts, batch_ids, self.edge_dist_metric, self.edge_dist_numpy)

        # Form the requested network
        if len(clusts) == 1:
            edge_index = np.empty((2,0))
        elif self.network == 'complete':
            edge_index = complete_graph(batch_ids, dist_mat, self.edge_max_dist)
        elif self.network == 'delaunay':
            edge_index = delaunay_graph(data, clusts, dist_mat, self.edge_max_dist)
        elif self.network == 'mst':
            edge_index = mst_graph(batch_ids, dist_mat, self.edge_max_dist)
        else:
            raise ValueError('Network type not recognized: '+self.network)

        # Skip if there is no edges
        if not edge_index.shape[1]:
            return result

        # Obtain node and edge features
        x = self.node_encoder(data, clusts)
        e = self.edge_encoder(data, clusts, edge_index)

        # Add start point and/or start direction to node features if requested
        if self.add_start_point:
            points = get_cluster_points_label(data, particles, clusts, groupwise=False)
            from mlreco.utils import local_cdist
            for i, c in enumerate(clusts):
                dist_mat = local_cdist(points[i,:3].reshape(1,-1), data[c,:3])
                points[i] = data[c][torch.argmin(dist_mat,dim=1),:3]
            x = torch.cat([x, points.float()], dim=1)
            if self.add_start_dir:
                dirs = get_cluster_directions(data, points[:,:3], clusts, self.start_dir_max_dist)
                x = torch.cat([x, dirs.float()], dim=1)

        # Bring edge_index and batch_ids to device
        index = torch.tensor(edge_index, device=device, dtype=torch.long)
        xbatch = torch.tensor(batch_ids, device=device)

        # Pass through the model, get output (long edge_index)
        out = self.edge_predictor(x, index, e, xbatch)
        node_pred = out['node_pred'][0]
        edge_pred = out['edge_pred'][0]

        # Divide the output out into different arrays (one per batch)
        _, counts = torch.unique(data[:,3], return_counts=True)
        vids = np.concatenate([np.arange(n.item()) for n in counts])
        cids = np.concatenate([np.arange(n) for n in np.unique(batch_ids, return_counts=True)[1]])
        bcids = [np.where(batch_ids == b)[0] for b in range(len(counts))]
        beids = [np.where(batch_ids[edge_index[0]] == b)[0] for b in range(len(counts))]

        node_pred = [node_pred[b] for b in bcids]
        edge_pred = [edge_pred[b] for b in beids]
        edge_index = [cids[edge_index[:,b]].T for b in beids]
        clusts = [np.array([vids[c] for c in np.array(clusts)[b]]) for b in bcids]

        # Figure out the group ids of each of the clusters (batch-wise groups)
        group_pred = []
        if self.group_pred == 'threshold':
            for b in range(len(counts)):
                group_pred.append(node_assignment(edge_index[b], np.argmax(edge_pred[b].detach().cpu().numpy(), axis=1), len(clusts[b])))
        elif self.group_pred == 'score':
            for b in range(len(counts)):
                if len(clusts[b]):
                    group_pred.append(node_assignment_score(edge_index[b], edge_pred[b].detach().cpu().numpy(), len(clusts[b])))
                else:
                    group_pred.append(np.array([], dtype = np.int64))

        return {'node_pred': [node_pred],
                'edge_pred': [edge_pred],
                'group_pred': [group_pred],
                'edge_index': [edge_index],
                'clusts': [clusts]}


class ChainLoss(torch.nn.modules.loss._Loss):
    """
    Takes the output of ClustHierarchyGNN and computes the total loss
    coming from the edge model and the node model.

    For use in config:
    model:
      name: cluster_hierachy_gnn
      modules:
        chain:
          loss            : <loss function: 'CE' or 'MM' (default 'CE')>
          reduction       : <loss reduction method: 'mean' or 'sum' (default 'sum')>
          balance_classes : <balance loss per class: True or False (default False)>
          target_photons  : <use true photon connections as basis for loss (default False)>
    """
    def __init__(self, cfg, name='chain'):
        super(ChainLoss, self).__init__()
        self.node_loss = NodeChannelLoss(cfg, name)
        self.edge_loss = EdgeChannelLoss(cfg, name)

    def forward(self, result, clust_label):
        # Apply edge loss
        loss = {}
        edge_loss = self.edge_loss(result, clust_label, None)
        loss.update(edge_loss)

        # Apply node loss
        # Override group IDs with those predicted. Determine the primary target by using the GT
        # primaries for each predicted group, iif the predicted group contains only one primary.
        # high_purity MUST be set to true in the configuration file for this to have an effect.
        clust_label_new = clust_label
        if 'node_pred' in result:
            clust_label_new = relabel_groups(clust_label, result['clusts'], result['group_pred'], new_array=True)
        node_loss = self.node_loss(result, clust_label_new)
        loss.update(node_loss)

        # Combine losses
        loss['node_loss'] = node_loss['loss']
        loss['edge_loss'] = edge_loss['loss']
        loss['loss'] = node_loss['loss'] + edge_loss['loss']
        loss['node_accuracy'] = node_loss['accuracy']
        loss['edge_accuracy'] = edge_loss['accuracy']
        loss['accuracy'] = (node_loss['accuracy'] + edge_loss['accuracy'])/2
        return loss
