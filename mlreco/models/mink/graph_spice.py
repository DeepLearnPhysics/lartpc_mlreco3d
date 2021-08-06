import torch
import numpy as np

from mlreco.models.scn.cluster_cnn.losses.gs_embeddings import *
from mlreco.models.scn.cluster_cnn import gs_kernel_construct

from mlreco.models.mink.cluster.graph_spice import GraphSPICEEmbedder

from pprint import pprint
from mlreco.utils.cluster.graph_spice import ClusterGraphConstructor


class MinkGraphSPICE(nn.Module):
    '''
    Neighbor-graph embedding based particle clustering.

    GraphSPICE has two components:
        1) Voxel Embedder: UNet-type CNN architecture used for feature
        extraction and feature embeddings.

        2) Edge Probability Kernel function: A kernel function (any callable
        that takes two node attribute vectors to give a edge proability score).

    Prediction is done in two steps:
        1) A neighbor graph (ex. KNN, Radius) is constructed to compute
        edge probabilities between neighboring edges.
        2) Edges with low probability scores are dropped.
        3) The voxels are clustered by counting connected components.

    Parameters:
        - skip_classes: semantic labels for which to skip voxel clustering
        (ex. Michel, Delta, and Low Es rarely require neural network clustering)

        - dimension: dimension of input dataset.
    '''

    def __init__(self, cfg, name='graph_spice'):
        super(MinkGraphSPICE, self).__init__()
        self.model_config = cfg[name]
        self.skip_classes = self.model_config.get('skip_classes', [2, 3, 4])
        self.dimension = self.model_config.get('dimension', 3)
        self.embedder_name = self.model_config.get('embedder', 'graph_spice_embedder')
        self.embedder = GraphSPICEEmbedder(self.model_config['embedder_cfg'])
        self.node_dim = self.model_config.get('node_dim', 16)

        self.kernel_cfg = self.model_config['kernel_cfg']
        self.kernel_fn = gs_kernel_construct(self.kernel_cfg)

        constructor_cfg = self.model_config['constructor_cfg']

        self.use_raw_features = self.model_config.get('use_raw_features', False)

        # Cluster Graph Manager
        self.gs_manager = ClusterGraphConstructor(constructor_cfg)
        self.gs_manager.training = self.training


    def filter_class(self, input):
        '''
        Filter classes according to segmentation label.
        '''
        point_cloud, label = input
        mask = ~np.isin(label[:, -1].detach().cpu().numpy(), self.skip_classes)
        x = [point_cloud[mask], label[mask]]
        return x


    def forward(self, input):
        '''

        '''
        point_cloud, labels = self.filter_class(input)
        res = self.embedder([point_cloud])

        coordinates = point_cloud[:, 1:4]
        batch_indices = point_cloud[:, 0].int()

        res['coordinates'] = [coordinates]
        res['batch_indices'] = [batch_indices]

        if self.use_raw_features:
            res['hypergraph_features'] = res['features']

        graph = self.gs_manager(res,
                                self.kernel_fn,
                                labels)
        res['graph'] = [graph]
        res['graph_info'] = [self.gs_manager.info]
        return res
