import torch
import torch.nn as nn
import numpy as np
import random

from mlreco.models.layers.common.dbscan import DBSCANFragmenter
from mlreco.models.layers.common.momentum import EvidentialMomentumNet, MomentumNet
from mlreco.models.layers.gnn import gnn_model_construct, node_encoder_construct, edge_encoder_construct, node_loss_construct, edge_loss_construct

from mlreco.utils.gnn.data import merge_batch
from mlreco.utils.gnn.cluster import form_clusters, get_cluster_batch, get_cluster_label, get_cluster_points_label, get_cluster_directions
from mlreco.utils.gnn.network import complete_graph, delaunay_graph, mst_graph, bipartite_graph, inter_cluster_distance, knn_graph






class VisionTransformer(nn.Module):

    def __init__(self, cfg, name='particle_transformer', batch_col=0, coords_col=(1,4)):
        super(VisionTransformer, self).__init__()

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
        self.shuffle_clusters = base_config.get('shuffle_clusters', False)

        self.batch_index = batch_col
        self.coords_index = coords_col

        # Interpret node type as list of classes to cluster, -1 means all classes
        if isinstance(self.node_type, int): self.node_type = [self.node_type]

        # Choose what type of network to use
        self.network = base_config.get('network', 'complete')
        self.edge_max_dist = base_config.get('edge_max_dist', -1)
        self.edge_dist_metric = base_config.get('edge_dist_metric', 'set')
        self.edge_dist_numpy = base_config.get('edge_dist_numpy',False)
        self.edge_knn_k = base_config.get('edge_knn_k', 5)

        # If requested, merge images together within the batch
        self.merge_batch = base_config.get('merge_batch', False)
        self.merge_batch_mode = base_config.get('merge_batch_mode', 'const')
        self.merge_batch_size = base_config.get('merge_batch_size', 2)

        # Initialize encoders
        self.node_encoder = node_encoder_construct(cfg[name], batch_col=self.batch_index, coords_col=self.coords_index)

        # Construct Vision Transformer
        

    def forward(self, data, clusts=None, groups=None, points=None, extra_feats=None):

        cluster_data = data[0]
        if len(data) > 1: particles = data[1]
        result = {}

        # Form list of list of voxel indices, one list per cluster in the requested class
        if clusts is None:
            if hasattr(self, 'dbscan'):
                clusts = self.dbscan(cluster_data, points=particles.detach().cpu().numpy() if len(data) > 1 else None)
            else:
                clusts = form_clusters(cluster_data.detach().cpu().numpy(), self.node_min_size, self.source_col, cluster_classes=self.node_type)

        # If requested, shuffle the order in which the clusters are listed (used for debugging)
        if self.shuffle_clusters:
            random.shuffle(clusts)

        # If requested, merge images together within the batch
        if self.merge_batch:
            cluster_data, particles, batch_list = merge_batch(cluster_data, particles, self.merge_batch_size, self.merge_batch_mode=='fluc')
            _, batch_counts = np.unique(batch_list, return_counts=True)
            result['batch_counts'] = [batch_counts]

        # Update result with a list of clusters for each batch id
        batches, bcounts = torch.unique(cluster_data[:,self.batch_index], return_counts=True)
        if not len(clusts):
            return {**result, 'clusts': [[np.array([]) for _ in batches]]}

        batch_ids = get_cluster_batch(cluster_data, clusts, batch_index=self.batch_index)
        cvids = np.concatenate([np.arange(n.item()) for n in bcounts])
        cbids = [np.where(batch_ids == b.item())[0] for b in batches]
        same_length = np.all([len(c) == len(clusts[0]) for c in clusts])
        clusts_np = np.array([c for c in clusts if len(c)], dtype=object if not same_length else np.int64)
        same_length = [np.all([len(c) == len(clusts_np[b][0]) for c in clusts_np[b]]) for b in cbids]
        result['clusts'] = [[np.array([cvids[c].astype(np.int64) for c in clusts_np[b]], dtype=np.object if not same_length[idx] else np.int64) for idx, b in enumerate(cbids)]]

        x = self.node_encoder(cluster_data, clusts)

        print(x, x.shape)

        assert False

        result['node_pred'] = [[x[b] for b in cbids]]

        return result