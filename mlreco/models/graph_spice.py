import torch
import numpy as np

from .cluster_cnn.losses.spatial_embeddings import *
from .cluster_cnn import (cluster_model_construct, 
                          spice_loss_construct,
                          gs_kernel_construct)

from pprint import pprint
from mlreco.utils.cluster.graph_spice import ClusterGraphConstructor

class GraphSPICE(nn.Module):
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
        super(GraphSPICE, self).__init__()
        print('--------------------CFG----------------------------')
        pprint(cfg)
        print('--------------------GraphSPICE---------------------')
        self.model_config = cfg[name]
        pprint(self.model_config)
        assert False
        self.skip_classes = self.model_config.get('skip_classes', [2, 3, 4])
        self.dimension = self.model_config.get('dimension', 3)
        self.embedder_name = self.model_config.get('embedder', 'graph_spice')
        self.embedder = cluster_model_construct(
            self.model_config, self.embedder_name)
        self.node_dim = self.model_config.get('node_dim', 16)

        self.kernel_cfg = self.model_conifg['kernel_cfg']
        self.kernel_fn = gs_kernel_construct(self.kernel_cfg)

        constructor_cfg = self.model_config['constructor_cfg']

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
        graph = self.cg_manager(res_embeddings, 
                                self.kernel_fn, 
                                [point_cloud, labels])
        res['graph'] = [graph]
        return res


class GraphSPICELoss(nn.Module):

    def __init__(self, cfg, name='spice_loss'):
        super(GraphSPICELoss, self).__init__()
        self.loss_config = cfg[name]
        print('--------------------GraphSPICELoss---------------------')
        pprint(self.loss_config)
        self.loss_name = self.loss_config['name']
        self.skip_classes = self.loss_config.get('skip_classes', [2, 3, 4])
        self.eval_mode = self.loss_config['eval_mode']
        self.loss_fn = spice_loss_construct(self.loss_name)(self.loss_config)

        constructor_cfg = cfg['constructor_cfg']
        self.gs_manager = ClusterGraphConstructor(constructor_cfg)
        self.gs_manager.training = ~self.eval_mode
        # print("LOSS FN = ", self.loss_fn)

    def filter_class(self, segment_label, cluster_label):
        '''
        Filter classes according to segmentation label. 
        '''
        mask = ~np.isin(segment_label[0][:, -1].detach().cpu().numpy(), self.skip_classes)
        slabel = [segment_label[0][mask]]
        clabel = [cluster_label[0][mask]]
        return slabel, clabel


    def forward(self, result, segment_label, cluster_label):
        '''

        '''
        slabel, clabel = self.filter_class(segment_label, cluster_label)

        graph = result['graph'][0]
        self.gs_manager.replace_state(graph)
        result['edge_score'] = [graph.edge_attr]
        result['edge_index'] = [graph.edge_index]
        if not self.eval_mode:
            result['edge_truth'] = [graph.edge_truth]

        res = self.loss_fn(result, slabel, clabel)

        # Evaluate Graph with respect to cluster_label
        self.gs_manager.evaluate_nodes(cluster_label[0], )
        return res
