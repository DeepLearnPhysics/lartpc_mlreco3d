# GNN that attempts to put clusters together into groups
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import copy
import numpy as np
from .gnn import node_model_construct, edge_model_construct, node_encoder_construct, edge_encoder_construct
from .layers.dbscan import DBScanClusts2
from mlreco.utils.gnn.cluster import form_clusters, get_cluster_label, get_cluster_batch, get_cluster_group
from mlreco.utils.gnn.network import complete_graph, delaunay_graph, mst_graph, bipartite_graph, inter_cluster_distance, get_fragment_edges
from mlreco.utils.gnn.evaluation import edge_assignment, edge_assignment_from_graph, node_assignment, node_assignment_score
from mlreco.utils import local_cdist
from mlreco.models.cluster_node_gnn import NodeChannelLoss
from mlreco.models.cluster_gnn import EdgeChannelLoss

class ClustGroupPriorGNN(torch.nn.Module):
    """
    Driver class for cluster node+edge prediction, assumed to be a GNN model.
    This class mostly acts as a wrapper that will hand the graph data to another model.
    If DBSCAN is used, use the semantic label tensor as an input.

    For use in config:
    model:
      name: cluster_hierachy_gnn
      modules:
        chain:
          node_type       : <semantic class to group (all classes if -1, default 0, i.e. EM)>
          node_min_size   : <minimum number of voxels inside a cluster to be considered (default -1)>
          network         : <type of node prediction network: 'complete', 'delaunay' or 'mst' (default 'complete')>
          edge_max_dist   : <maximal edge Euclidean length (default -1)>
          edge_dist_method: <edge length evaluation method: 'centroid' or 'set' (default 'set')>
          edge_dist_numpy : <use numpy to compute inter cluster distance (default False)>
          group_pred      : <group prediction method: 'threshold', 'score' (default 'threshold')>
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
    def __init__(self, cfg):
        super(ClustGroupPriorGNN, self).__init__()

        # Get the chain input parameters
        chain_config = cfg['chain']

        # Choose what type of node to use
        self.node_type = chain_config.get('node_type', 0)
        self.node_min_size = chain_config.get('node_min_size', -1)

        # Choose what type of network to use
        self.network = chain_config.get('network', 'complete')
        self.edge_max_dist = chain_config.get('edge_max_dist', -1)
        self.edge_dist_metric = chain_config.get('edge_dist_metric', 'set')
        self.edge_dist_numpy = chain_config.get('edge_dist_numpy',False)
        self.group_pred = chain_config.get('group_pred','threshold')

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
        self.node_predictor = node_model_construct(cfg)

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
                mask = torch.nonzero(data[:,7] == self.node_type).flatten()
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

        # Bring edge_index and batch_ids to device
        index = torch.tensor(edge_index, device=device, dtype=torch.long)
        xbatch = torch.tensor(batch_ids, device=device)

        # Pass through the edge model, get edge predictions
        out = self.edge_predictor(x, index, e, xbatch)
        edge_pred = out['edge_pred'][0]

        # Divide the output out into different arrays (one per batch)
        _, counts = torch.unique(data[:,3], return_counts=True)
        vids = np.concatenate([np.arange(n.item()) for n in counts])
        cids = np.concatenate([np.arange(n) for n in np.unique(batch_ids, return_counts=True)[1]])
        bcids = [np.where(batch_ids == b)[0] for b in range(len(counts))]
        beids = [np.where(batch_ids[edge_index[0]] == b)[0] for b in range(len(counts))]

        edge_pred = [edge_pred[b] for b in beids]
        edge_index = [cids[edge_index[:,b]].T for b in beids]
        div_clusts = [np.array([vids[c] for c in np.array(clusts)[b]]) for b in bcids]

        # Get the group_ids of each of the clusters (batch-wise groups)
        div_group_ids = []
        if self.group_pred == 'threshold':
            for b in range(len(counts)):
                div_group_ids.append(node_assignment(edge_index[b], np.argmax(edge_pred[b].detach().cpu().numpy(), axis=1), len(div_clusts[b])))
        elif self.group_pred == 'score':
            for b in range(len(counts)):
                if len(div_clusts[b]):
                    div_group_ids.append(node_assignment_score(edge_index[b], edge_pred[b].detach().cpu().numpy(), len(div_clusts[b])))
                else:
                    div_group_ids.append(np.array([], dtype = np.int64))

        group_ids = np.concatenate(div_group_ids)
        pairs = np.vstack((group_ids, batch_ids)).T
        pairs = np.ascontiguousarray(pairs).view(np.dtype((np.void, pairs.dtype.itemsize * pairs.shape[1])))
        _, group_ids = np.unique(pairs, return_inverse=True)

        # Build a graph that only connects nodes within the same group
        node_edge_index = complete_graph(group_ids)
        if node_edge_index.shape[1] < 2:
            return {'edge_pred': [edge_pred],
                    'edge_index': [edge_index],
                    'clusts': [div_clusts],
                    'group_ids':[div_group_ids]}

        # Bring node_edge_index to device
        index = torch.tensor(node_edge_index, device=device, dtype=torch.long)

        # Obtain node and edge features
        #x = self.node_encoder(data, clusts)
        x = torch.cat([out['node_pred'][0], self.node_encoder(data, clusts)], dim=1)
        e = self.edge_encoder(data, clusts, node_edge_index)

        # Pass through the node model, get node predictions (long edge_index)
        out = self.node_predictor(x, index, e, xbatch)
        node_pred = out['node_pred'][0]

        # Divide the output out into different arrays (one per batch)
        bneids = [np.where(batch_ids[node_edge_index[0]] == b)[0] for b in range(len(counts))]

        node_pred = [node_pred[b] for b in bcids]
        node_edge_index = [cids[node_edge_index[:,b]].T for b in bneids]

        return {'node_pred': [node_pred],
                'edge_pred': [edge_pred],
                'edge_index': [edge_index],
                'node_edge_index': [node_edge_index],
                'clusts': [div_clusts],
                'group_ids':[div_group_ids]}


class GroupPriorLoss(torch.nn.modules.loss._Loss):
    """
    Takes the output of ClustGroupPriorGNN and computes the total loss
    coming from the edge model and the node model. Node predictions
    are attempted within a grouping prior.

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
    def __init__(self, cfg):
        super(GroupPriorLoss, self).__init__()
        self.edge_loss = EdgeChannelLoss(cfg)
        self.node_loss = NodeChannelLoss(cfg)

    def forward(self, result, clusters, particles):
        loss = {}
        # Apply the regular channel edge loss
        edge_loss = self.edge_loss(result, clusters, None)
        loss.update(edge_loss)
        loss['edge_loss'] = edge_loss['loss']
        loss['edge_accuracy'] = edge_loss['accuracy']

        # Override group IDs with those predicted. Determine the primary
        # by either using the GT primaries or by taking the earliest cluster in time
        # for each predicted group.
        clusters_new = clusters
        #clusters_new = copy.deepcopy(clusters)
        if 'node_pred' in result:
            device = clusters[0].device
            dtype  = clusters[0].dtype
            for i in range(len(clusters)):
                batches = clusters[i][:,3]
                for b in batches.unique():
                    batch_mask = torch.nonzero(batches == b).flatten()
                    labels = clusters[i][batch_mask]
                    clusts = result['clusts'][i][b.int().item()]
                    clust_ids = get_cluster_label(labels, clusts)
                    part_info = particles[i][particles[i][:,-1] == b.int().item()][clust_ids]
                    times = part_info[:,3]
                    group_ids = result['group_ids'][i][b.int().item()]
                    groups = [np.where(group_ids==g)[0] for g in np.unique(group_ids)]
                    first_ids = [g[np.argmin(times[g])] for g in groups]
                    # This sets the group id to the cluster that is first in time
                    #for j, c in enumerate(clusts):
                    #    batch_mask[c]
                    #    clust_id = group_ids[j] if j in first_ids else -1
                    #    new_labels = torch.full([len(c)], clust_id, dtype=dtype).to(device)
                    #    new_groups = torch.full([len(c)], group_ids[j], dtype=dtype).to(device)
                    #    clusters_new[i][batch_mask[c], 5] = new_labels
                    #    clusters_new[i][batch_mask[c], 6] = new_groups

                    # This sets the group ID to the cluster that is a GT primary in that group
                    # if there is only one GT primary, and to some crap otherwise
                    true_group_ids = get_cluster_group(labels, clusts)
                    primary_mask   = clust_ids == true_group_ids
                    new_id = len(particles[i][particles[i][:,-1] == b.int().item()])
                    for g in np.unique(group_ids):
                        group_mask     = group_ids == g
                        primary_labels = np.where(primary_mask & group_mask)[0]
                        group_id = -1
                        if len(primary_labels) != 1:
                            group_id = new_id
                            new_id += 1
                        else:
                            group_id = clust_ids[primary_labels[0]]
                        for c in clusts[group_mask]:
                            new_groups = torch.full([len(c)], group_id, dtype=dtype).to(device)
                            clusters_new[i][batch_mask[c], 6] = new_groups

        node_loss = self.node_loss(result, clusters_new)
        loss.update(node_loss)
        loss['node_loss'] = node_loss['loss']
        loss['node_accuracy'] = node_loss['accuracy']

        # Compute total loss and total accuracy
        loss['loss'] = node_loss['loss'] + edge_loss['loss']
        loss['accuracy'] = (node_loss['accuracy'] + edge_loss['accuracy'])/2
        return loss
