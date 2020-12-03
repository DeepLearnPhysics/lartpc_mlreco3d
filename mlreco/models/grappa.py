# GNN that attempts to put clusters together into groups
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import numpy as np
from .layers.dbscan import DBScanClusts2 as DBSCAN
from .layers.momentum import MomentumNet
from .gnn import gnn_model_construct, node_encoder_construct, edge_encoder_construct, node_loss_construct, edge_loss_construct
from mlreco.utils.gnn.data import merge_batch
from mlreco.utils.gnn.cluster import form_clusters, get_cluster_batch, get_cluster_label, relabel_groups, get_cluster_points_label, get_cluster_directions
from mlreco.utils.gnn.network import complete_graph, delaunay_graph, mst_graph, bipartite_graph, inter_cluster_distance, get_fragment_edges

class GNN(torch.nn.Module):
    """
    Driver class for cluster node+edge prediction, assumed to be a GNN model.
    This class mostly acts as a wrapper that will hand the graph data to another model.
    If DBSCAN is used, use the semantic label tensor as an input.

    For use in config:
    model:
      name: grappa
      modules:
        grappa:
          base:
            node_type         : <semantic class to group (all classes if -1, default 0, i.e. EM)>
            node_min_size     : <minimum number of voxels inside a cluster to be considered (default -1)>
            source_col        : <column in the input data that specifies the source node ids of each voxel (default 5)>
            target_col        : <column in the input data that specifies the target instance ids of each voxel (default 6)>
            use_dbscan        : <use DBSCAN to cluster the input instances of the class specified by node_type (default False)>
            add_start_point   : <add label start point to the node features (default False)>
            add_start_dir     : <add predicted start direction to the node features (default False)>
            start_dir_max_dist: <maximium distance between start point and cluster voxels to be used to estimate direction (default -1, i.e no limit)>
            start_dir_opt     : <optimize start direction by minimizing relative transverse spread of neighborhood (slow, default: False)>
            start_dir_cpu     : <optimize the start direction on CPU (default: False)>
            network           : <type of network: 'complete', 'delaunay', 'mst', 'knn' or 'bipartite' (default 'complete')>
            edge_max_dist     : <maximal edge Euclidean length (default -1)>
            edge_dist_method  : <edge length evaluation method: 'centroid' or 'set' (default 'set')>
            edge_dist_numpy   : <use numpy to compute inter cluster distance (default False)>
            merge_batch       : <flag for whether to merge batches (default False)>
            merge_batch_mode  : <mode of batch merging, 'const' or 'fluc'; 'const' use a fixed size of batch for merging, 'fluc' takes the input size a mean and sample based on it (default 'const')>
            merge_batch_size  : <size of batch merging (default 2)>
            shuffle_clusters  : <randomize cluster order (default False)>
            kinematics_mlp    : <applies type and momentum MLPs on the node features (default False)>
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

    MODULES = [('grappa', ['base', 'dbscan', 'node_encoder', 'edge_encoder', 'gnn_model']), 'grappa_loss']

    def __init__(self, cfg, name='grappa'):
        super(GNN, self).__init__()

        # Get the chain input parameters
        base_config = cfg[name].get('base', {})

        # Choose what type of node to use
        self.node_type = base_config.get('node_type', 0)
        self.node_min_size = base_config.get('node_min_size', -1)
        self.source_col = base_config.get('source_col', 5)
        self.target_col = base_config.get('target_col', 6)
        self.add_start_point = base_config.get('add_start_point', False)
        self.add_start_dir = base_config.get('add_start_dir', False)
        self.start_dir_max_dist = base_config.get('start_dir_max_dist', -1)
        self.start_dir_opt = base_config.get('start_dir_opt', False)
        self.start_dir_cpu = base_config.get('start_dir_cpu', False)
        self.shuffle_clusters = base_config.get('shuffle_clusters', False)

        # Choose what type of network to use
        self.network = base_config.get('network', 'complete')
        self.edge_max_dist = base_config.get('edge_max_dist', -1)
        self.edge_dist_metric = base_config.get('edge_dist_metric', 'set')
        self.edge_dist_numpy = base_config.get('edge_dist_numpy',False)
        self.group_pred = base_config.get('group_pred','score')

        # If requested, merge images together within the batch
        self.merge_batch = base_config.get('merge_batch', False)
        self.merge_batch_mode = base_config.get('merge_batch_mode', 'const')
        self.merge_batch_size = base_config.get('merge_batch_size', 2)

        # If requested, use DBSCAN to form clusters from semantics
        if 'dbscan' in cfg[name]:
            self.dbscan = DBSCAN(cfg[name])

        # If requested, initialize two MLPs for kinematics predictions
        self.kinematics_mlp = base_config.get('kinematics_mlp', False)
        if self.kinematics_mlp:
            node_output_feats = cfg[name]['gnn_model'].get('node_output_feats', 64)
            self.type_net = MomentumNet(node_output_feats, 5)
            self.momentum_net = MomentumNet(node_output_feats, 1)

        # Initialize encoders
        self.node_encoder = node_encoder_construct(cfg[name])
        self.edge_encoder = edge_encoder_construct(cfg[name])

        # Construct the GNN
        self.edge_predictor = gnn_model_construct(cfg[name])

    def forward(self, data):
        """
        Prepares particle clusters and feed them to the GNN model.

        Args:
            array:
                data[0] ([torch.tensor]): (N,5-10) [x, y, z, batch_id(, value), part_id(, group_id, int_id, nu_id, sem_type)]
                                       or (N,5) [x, y, z, batch_id, sem_type] (with DBSCAN)
                data[1] ([torch.tensor]): (N,8) [first_x, first_y, first_z, batch_id, last_x, last_y, last_z, first_step_t] (optional)
        Returns:
            dict:
                'node_pred' (torch.tensor): (N,2) Two-channel node predictions
                'edge_pred' (torch.tensor): (E,2) Two-channel edge predictions
                'clusts' ([np.ndarray])   : [(N_0), (N_1), ..., (N_C)] Cluster ids
                'edge_index' (np.ndarray) : (E,2) Incidence matrix
        """

        # Form list of list of voxel indices, one list per cluster in the requested class
        cluster_data = data[0]
        if len(data) > 1: particles = data[1]
        result = {}
        if hasattr(self, 'dbscan'):
            clusts = self.dbscan(cluster_data, onehot=False)
            clusts = clusts[self.node_type] if self.node_type > -1 else np.concatenate(clusts).tolist()
        else:
            if self.node_type > -1:
                mask = torch.nonzero(cluster_data[:,-1] == self.node_type, as_tuple=True)[0]
                clusts = form_clusters(cluster_data[mask], self.node_min_size, self.source_col)
                clusts = [mask[c].cpu().numpy() for c in clusts]
            else:
                clusts = form_clusters(cluster_data, self.node_min_size, self.source_col)
                clusts = [c.cpu().numpy() for c in clusts]

        # If requested, shuffle the order in which the clusters are listed (used for debugging)
        if self.shuffle_clusters:
            import random
            random.shuffle(clusts)

        # If requested, merge images together within the batch
        if self.merge_batch:
            cluster_data, particles, batch_list = merge_batch(cluster_data, particles, self.merge_batch_size, self.merge_batch_mode=='fluc')
            _, batch_counts = np.unique(batch_list, return_counts=True)
            result['batch_counts'] = [batch_counts]

        # Update result with a list of clusters for each batch id
        batches, bcounts = torch.unique(cluster_data[:,3], return_counts=True)
        if not len(clusts):
            return {**result, 'clusts': [[np.array([]) for _ in batches]]}

        batch_ids = get_cluster_batch(cluster_data, clusts)
        cvids = np.concatenate([np.arange(n.item()) for n in bcounts])
        cbids = [np.where(batch_ids == b.item())[0] for b in batches]
        same_length = np.all([len(c) == len(clusts[0]) for c in clusts])
        clusts_np = np.array([c for c in clusts if len(c)], dtype=object if not same_length else np.int64)
        same_length = [np.all([len(c) == len(clusts_np[b][0]) for c in clusts_np[b]]) for b in cbids]
        result['clusts'] = [[np.array([cvids[c].astype(np.int64) for c in clusts_np[b]], dtype=np.object if not same_length[idx] else np.int64) for idx, b in enumerate(cbids)]]

        # If necessary, compute the cluster distance matrix
        dist_mat = None
        if self.edge_max_dist > 0 or self.network == 'mst' or self.network == 'knn':
            dist_mat = inter_cluster_distance(cluster_data[:,:3], clusts, batch_ids, self.edge_dist_metric, self.edge_dist_numpy)

        # Form the requested network
        if len(clusts) == 1:
            edge_index = np.empty((2,0))
        elif self.network == 'complete':
            edge_index = complete_graph(batch_ids, dist_mat, self.edge_max_dist)
        elif self.network == 'delaunay':
            edge_index = delaunay_graph(cluster_data.cpu().numpy(), clusts, dist_mat, self.edge_max_dist)
        elif self.network == 'mst':
            edge_index = mst_graph(batch_ids, dist_mat, self.edge_max_dist)
        elif self.network == 'knn':
            edge_index = knn_graph(batch_ids, dist_mat, k=5, undirected=True)
        elif self.network == 'bipartite':
            primary_ids = [i for i, c in enumerate(clusts) if (cluster_data[c,self.source_col] == cluster_data[c,self.target_col]).any()]
            edge_index = bipartite_graph(batch_ids, primary_ids, dist_mat, self.edge_max_dist)
        else:
            raise ValueError('Network type not recognized: '+self.network)

        # Update result with a list of edges for each batch id
        if not edge_index.shape[1]:
            return {**result, 'edge_index':[np.empty((2,0)) for _ in batches]}

        ebids = [np.where(batch_ids[edge_index[0]] == b.item())[0] for b in batches]
        ecids = np.concatenate([np.arange(n) for n in np.unique(batch_ids, return_counts=True)[1]])
        result['edge_index'] = [[ecids[edge_index[:,b]].T for b in ebids]]

        # Obtain node and edge features
        x = self.node_encoder(cluster_data, clusts)
        e = self.edge_encoder(cluster_data, clusts, edge_index)

        # Add start point and/or start direction to node features if requested
        if self.add_start_point:
            points = get_cluster_points_label(cluster_data, particles, clusts, self.source_col==6)
            for i, c in enumerate(clusts):
                dist_mat = torch.cdist(points[i].reshape(-1,3), cluster_data[c,:3])
                points[i] = cluster_data[c][torch.argmin(dist_mat,dim=1),:3].reshape(-1)
            x = torch.cat([x, points.float()], dim=1)
            if self.add_start_dir:
                dirs = get_cluster_directions(cluster_data, points[:,:3], clusts, self.start_dir_max_dist, self.start_dir_opt, self.start_dir_cpu)
                x = torch.cat([x, dirs.float()], dim=1)

        # Bring edge_index and batch_ids to device
        device = cluster_data.device
        index = torch.tensor(edge_index, device=device, dtype=torch.long)
        xbatch = torch.tensor(batch_ids, device=device)

        # Pass through the model, update result
        out = self.edge_predictor(x, index, e, xbatch)
        result['node_pred'] = [[out['node_pred'][0][b] for b in cbids]]
        result['edge_pred'] = [[out['edge_pred'][0][b] for b in ebids]]

        # If requested, pass the node features through two MLPs for kinematics predictions
        if self.kinematics_mlp:
            node_pred_type = self.type_net(out['node_features'][0])
            node_pred_p = self.momentum_net(out['node_features'][0])
            result['node_pred_type'] = [[node_pred_type[b] for b in cbids]]
            result['node_pred_p'] = [[node_pred_p[b] for b in cbids]]

        return result


class GNNLoss(torch.nn.modules.loss._Loss):
    """
    Takes the output of the GNN and computes the total loss.

    For use in config:
    model:
      name: grappa
      modules:
        grappa_loss:
          node_loss:
            name: <name of the node loss>
            <dictionary of arguments to pass to the loss>
          edge_loss:
            name: <name of the edge loss>
            <dictionary of arguments to pass to the loss>
    """
    def __init__(self, cfg, name='grappa_loss'):
        super(GNNLoss, self).__init__()

        # Initialize the node and edge losses, if requested
        self.apply_node_loss, self.apply_edge_loss = False, False
        if 'node_loss' in cfg[name]:
            self.apply_node_loss = True
            self.node_loss = node_loss_construct(cfg[name])
        if 'edge_loss' in cfg[name]:
            self.apply_edge_loss = True
            self.edge_loss = edge_loss_construct(cfg[name])

    def forward(self, result, clust_label, graph=None):

        # Apply edge and node losses, if instantiated
        loss = {}
        if self.apply_node_loss:
            node_loss = self.node_loss(result, clust_label)
            loss.update(node_loss)
            loss['node_loss'] = node_loss['loss']
            loss['node_accuracy'] = node_loss['accuracy']
        if self.apply_edge_loss:
            edge_loss = self.edge_loss(result, clust_label, graph)
            loss.update(edge_loss)
            loss['edge_loss'] = edge_loss['loss']
            loss['edge_accuracy'] = edge_loss['accuracy']
        if self.apply_node_loss and self.apply_edge_loss:
            loss['loss'] = loss['node_loss'] + loss['edge_loss']
            loss['accuracy'] = (loss['node_accuracy'] + loss['edge_accuracy'])/2

        return loss
