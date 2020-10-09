# GNN that attempts to put clusters together into groups
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
from .gnn import edge_model_construct, node_encoder_construct, edge_encoder_construct
from .layers.dbscan import DBScanClusts2
from mlreco.utils.gnn.data import merge_batch
from mlreco.utils.gnn.cluster import form_clusters, get_cluster_batch, get_cluster_label, get_cluster_points_label, get_cluster_directions
from mlreco.utils.gnn.network import complete_graph, delaunay_graph, mst_graph, knn_graph, bipartite_graph, inter_cluster_distance, get_fragment_edges
from mlreco.utils.gnn.evaluation import edge_assignment, edge_assignment_from_graph
from mlreco.models.cluster_node_gnn import NodeKinematicsLoss
from mlreco.models.cluster_gnn import EdgeChannelLoss


class MomentumNet(nn.Module):
    '''
    Small MLP for extracting input edge features from two node features.

    USAGE:
        net = EdgeFeatureNet(16, 16)
        node_x = torch.randn(16, 5)
        node_y = torch.randn(16, 5)
        edge_feature_x2y = net(node_x, node_y) # (16, 5)
    '''
    def __init__(self, num_input, num_output=1, num_hidden=128):
        super(MomentumNet, self).__init__()
        self.linear1 = nn.Linear(num_input, num_hidden)
        self.norm1 = nn.BatchNorm1d(num_hidden)
        self.linear2 = nn.Linear(num_hidden, num_hidden)
        self.norm2 = nn.BatchNorm1d(num_hidden)
        self.linear3 = nn.Linear(num_hidden, num_output)

        self.elu = nn.LeakyReLU(negative_slope=0.33)
        self.softplus = nn.Softplus()

    def forward(self, x):
        if x.shape[0] > 1:
            self.norm1(x)
        x = self.linear1(x)
        x = self.elu(x)
        if x.shape[0] > 1:
            x = self.norm2(x)
        x = self.linear2(x)
        x = self.elu(x)
        x = self.linear3(x)
        out = self.softplus(x)
        return out


def get_edge_features(nodes, batch_idx, edge_net):
    '''
    Compile Fully Connected Edge Features from nodes and batch indices.

    INPUTS:
        - nodes (N x d Tensor): list of node features
        - batch_idx (N x 1 Tensor): list of batch indices for nodes
        - bilinear_net: nn.Module that taks two vectors and returns edge feature vector.

    RETURNS:
        - edge_features: list of edges features
        - edge_indices: list of edge indices (i->j)
        - edge_batch_indices: list of batch indices (0 to B)
    '''
    unique_batch = batch_idx.unique()
    edge_index = []
    edge_features = []
    for bidx in unique_batch:
        mask = bidx == batch_idx
        clust_ids = torch.nonzero(mask).flatten()
        nodes_batch = nodes[mask]
        subindex = torch.arange(nodes_batch.shape[0])
        N = nodes_batch.shape[0]
        for i, row in enumerate(nodes_batch):
            submask = subindex != i
            edge_idx = [[clust_ids[i].item(), clust_ids[j].item()] for j in subindex[submask]]
            edge_index.extend(edge_idx)
            others = nodes_batch[submask]
            ei2j = edge_net(row.expand_as(others), others)
            edge_features.extend(ei2j)

    edge_index = np.vstack(edge_index)
    edge_features = torch.stack(edge_features, dim=0)

    return edge_index, edge_features


# class EdgeFeatureNet(nn.Module):
#     '''
#     Small MLP for extracting input edge features from two node features.

#     USAGE:
#         net = EdgeFeatureNet(16, 16)
#         node_x = torch.randn(16, 5)
#         node_y = torch.randn(16, 5)
#         edge_feature_x2y = net(node_x, node_y) # (16, 5)
#     '''
#     def __init__(self, num_input, num_output):
#         super(EdgeFeatureNet, self).__init__()
#         self.linear1 = nn.Linear(num_input * 2, 64)
#         self.norm1 = nn.BatchNorm1d(64)
#         self.linear2 = nn.Linear(64, 64)
#         self.norm2 = nn.BatchNorm1d(64)
#         self.linear3 = nn.Linear(64, num_output)

#         self.elu = nn.ELU()

#     def forward(self, x1, x2):
#         x = torch.cat([x1, x2], dim=1)
#         x = self.linear1(x)
#         if x.shape[0] > 1:
#             x = self.elu(self.norm1(x))
#         x = self.linear2(x)
#         if x.shape[0] > 1:
#             x = self.elu(self.norm2(x))
#         x = self.linear3(x)
#         return x


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
        self.source_col = chain_config.get('source_col', 6)

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
        self.momentum_net = MomentumNet(cfg['edge_model']['node_output_feats'], 1)
        self.type_net = MomentumNet(cfg['edge_model']['node_output_feats'], 5)

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
        # print(data)
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
                clusts = form_clusters(data[mask], self.node_min_size, self.source_col)
                clusts = [mask[c].cpu().numpy() for c in clusts]
            else:
                clusts = form_clusters(data, self.node_min_size, self.source_col)
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
        # edge_index, e = get_edge_features(x, batch_ids, self.edge_mlp)
        e = self.edge_encoder(data, clusts, edge_index)

        # print(x, x.shape)
        # print(e, e.shape)

        # Add start point and/or start direction to node features if requested
        if self.add_start_point:
            points = get_cluster_points_label(data, particles, clusts, groupwise=False)
            x = torch.cat([x, points.float()], dim=1)
            if self.add_start_dir:
                dirs = get_cluster_directions(data, points[:,:3], clusts, self.start_dir_max_dist)
                x = torch.cat([x, dirs.float()], dim=1)

        # Bring edge_index and batch_ids to device
        index = torch.tensor(edge_index, device=device, dtype=torch.long)
        xbatch = torch.tensor(batch_ids, device=device)

        # Pass through the model, get output (long edge_index)
        out = self.edge_predictor(x, index, e, xbatch)
        node_F = out['node_features'][0]
        edge_pred = out['edge_pred'][0]

        # print("node_F = ", node_F)

        node_pred_type = self.type_net(node_F)
        node_pred_p = self.momentum_net(node_F)

        # print("node_pred_type = ", node_pred_type)
        # print("node_pred_p = ", node_pred_p)

        # Divide the output out into different arrays (one per batch)
        _, counts = torch.unique(data[:,3], return_counts=True)
        vids = np.concatenate([np.arange(n.item()) for n in counts])
        cids = np.concatenate([np.arange(n) for n in np.unique(batch_ids, return_counts=True)[1]])
        bcids = [np.where(batch_ids == b)[0] for b in range(len(counts))]
        beids = [np.where(batch_ids[edge_index[0]] == b)[0] for b in range(len(counts))]

        node_pred_type = [node_pred_type[b] for b in bcids]
        node_pred_p = [node_pred_p[b] for b in bcids]
        edge_pred = [edge_pred[b] for b in beids]
        edge_index = [cids[edge_index[:,b]].T for b in beids]
        clusts = [np.array([vids[c] for c in np.array(clusts)[b]]) for b in bcids]

        # # Figure out the group ids of each of the clusters (batch-wise groups)
        # group_pred = []
        # if self.group_pred == 'threshold':
        #     for b in range(len(counts)):
        #         group_pred.append(node_assignment(edge_index[b], np.argmax(edge_pred[b].detach().cpu().numpy(), axis=1), len(clusts[b])))
        # elif self.group_pred == 'score':
        #     for b in range(len(counts)):
        #         if len(clusts[b]):
        #             group_pred.append(node_assignment_score(edge_index[b], edge_pred[b].detach().cpu().numpy(), len(clusts[b])))
        #         else:
        #             group_pred.append(np.array([], dtype = np.int64))
        res = {'node_pred_type': [node_pred_type],
                'node_pred_p': [node_pred_p], 
                'edge_pred': [edge_pred],
                # 'group_pred': [group_pred],
                'edge_index': [edge_index],
                'clusts': [clusts]}

        # print(res)

        return res



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
        self.node_loss = NodeKinematicsLoss(cfg, name)
        print(self.node_loss)
        self.edge_loss = EdgeChannelLoss(cfg, name)

    def forward(self, result, clust_label, graph):
        # Apply edge loss
        loss = {}
        edge_loss = self.edge_loss(result, clust_label, graph)
        loss.update(edge_loss)

        # Apply node loss
        # Override group IDs with those predicted. Determine the primary target by using the GT
        # primaries for each predicted group, iif the predicted group contains only one primary.
        # high_purity MUST be set to true in the configuration file for this to have an effect.
        clust_label_new = clust_label
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
