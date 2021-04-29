# GNN that attempts to put clusters together into groups
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import numpy as np
from .gnn import node_model_construct, edge_model_construct, node_encoder_construct, edge_encoder_construct
from .layers.dbscan import DBScanClusts2
from mlreco.utils.gnn.cluster import form_clusters, get_cluster_label, get_cluster_batch
from mlreco.utils.gnn.network import complete_graph, delaunay_graph, mst_graph, bipartite_graph, inter_cluster_distance, get_fragment_edges
from mlreco.utils.gnn.evaluation import edge_assignment, edge_assignment_from_graph
from mlreco.utils import local_cdist
from mlreco.models.cluster_node_gnn import NodeChannelLoss
from mlreco.models.cluster_gnn import EdgeChannelLoss

class ClustBipartiteGNN(torch.nn.Module):
    """
    Driver class for cluster node+edge prediction, assumed to be a GNN model.
    This class mostly acts as a wrapper that will hand the graph data to another model.
    If DBSCAN is used, use the semantic label tensor as an input.

    For use in config:
    model:
      name: cluster_bipartite_gnn
      modules:
        chain:
          node_type       : <semantic class to group (all classes if -1, default 0, i.e. EM)>
          node_min_size   : <minimum number of voxels inside a cluster to be considered (default -1)>
          network         : <type of node prediction network: 'complete', 'delaunay' or 'mst' (default 'complete')>
          edge_max_dist   : <maximal edge Euclidean length (default -1)>
          edge_dist_method: <edge length evaluation method: 'centroid' or 'set' (default 'set')>
          edge_dist_numpy : <use numpy to compute inter cluster distance (default False)>
          directed:       : <True if the edge bipartite network is directed (default True)>
          directed_to     : <nodes in the edge bipartite graph the messages are passed to (default 'secondary')>
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
        super(ClustBipartiteGNN, self).__init__()

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
        self.directed = chain_config.get('directed', True)
        self.directed_to = chain_config.get('directed_to', 'secondary')

        # If requested, use DBSCAN to form clusters from semantics
        self.do_dbscan = False
        if 'dbscan' in cfg:
            self.do_dbscan = True
            self.dbscan = DBScanClusts2(cfg)

        # Initialize encoders
        self.node_encoder = node_encoder_construct(cfg)
        self.edge_encoder = edge_encoder_construct(cfg)

        # Construct the models
        self.node_predictor = node_model_construct(cfg)
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

        # Pass through the node model, get node predictions
        index = torch.tensor(edge_index, device=device, dtype=torch.long)
        xbatch = torch.tensor(batch_ids, device=device, dtype=torch.long)
        out = self.node_predictor(x, index, e, xbatch)
        node_pred = out['node_pred'][0]

        # Split the node prediction output, append result
        _, counts = torch.unique(data[:,3], return_counts=True)
        cids = np.concatenate([np.arange(n) for n in np.unique(batch_ids, return_counts=True)[1]])
        vids = np.concatenate([np.arange(n.item()) for n in counts])
        bcids = [np.where(batch_ids == b)[0] for b in range(len(counts))]
        beids = [np.where(batch_ids[edge_index[0]] == b)[0] for b in range(len(counts))]

        split_clusts = [np.array([vids[c] for c in np.array(clusts)[b]]) for b in bcids]
        node_pred = [node_pred[b] for b in bcids]
        edge_index = [cids[edge_index[:,b]].T for b in beids]

        result.update(dict(
            node_pred = [node_pred],
            clusts = [split_clusts],
            node_edge_index = [edge_index]
        ))

        # Convert the node output to a list of primaries
        primary_ids = torch.argmax(out['node_pred'][0], dim=1)
        primaries = torch.nonzero(primary_ids).flatten()

        # Initialize the network for edge prediction, get edge features
        edge_index = bipartite_graph(batch_ids, primaries, dist_mat, self.edge_max_dist, self.directed, self.directed_to)
        if edge_index.shape[1] < 2: # Batch norm 1D does not handle batch_size < 2
            return result
        e = self.edge_encoder(data, clusts, edge_index)

        # Pass through the node model, get edge predictions
        index = torch.tensor(edge_index, device=device, dtype=torch.long)
        xbatch = torch.tensor(batch_ids, device=device, dtype=torch.long)
        out = self.edge_predictor(x, index, e, xbatch)
        edge_pred = out['edge_pred'][0]

        # Split the edge prediction output, append result
        beids = [np.where(batch_ids[edge_index[0]] == b)[0] for b in range(len(counts))]

        edge_pred = [edge_pred[b] for b in beids]
        edge_index = [cids[edge_index[:,b]].T for b in beids]

        result.update(dict(
            edge_pred = [edge_pred],
            edge_index = [edge_index]
        ))

        return result


class ChainLoss(torch.nn.modules.loss._Loss):
    """
    Takes the output of ClustBipartiteGNN and computes the total loss
    coming from the edge model and the node model.

    For use in config:
    model:
      name: cluster_bipartite_gnn
      modules:
        chain:
          loss            : <loss function: 'CE' or 'MM' (default 'CE')>
          reduction       : <loss reduction method: 'mean' or 'sum' (default 'sum')>
          balance_classes : <balance loss per class: True or False (default False)>
          target_photons  : <use true photon connections as basis for loss (default False)>
    """
    def __init__(self, cfg):
        super(ChainLoss, self).__init__()
        self.node_loss = NodeChannelLoss(cfg)
        self.edge_loss = EdgeChannelLoss(cfg)

    def forward(self, result, clust_label):
        loss = {}
        node_loss = self.node_loss(result, clust_label)
        edge_loss = self.edge_loss(result, clust_label, None)
        loss.update(node_loss)
        loss.update(edge_loss)
        loss['node_loss'] = node_loss['loss']
        loss['edge_loss'] = edge_loss['loss']
        loss['loss'] = node_loss['loss'] + edge_loss['loss']
        loss['node_accuracy'] = node_loss['accuracy']
        loss['edge_accuracy'] = edge_loss['accuracy']
        loss['accuracy'] = (node_loss['accuracy'] + edge_loss['accuracy'])/2
        return loss
