import torch
import numpy as np
import MinkowskiEngine as ME

from mlreco.models.layers.cluster_cnn.losses.gs_embeddings import *
from mlreco.models.layers.cluster_cnn import gs_kernel_construct, spice_loss_construct

from mlreco.models.layers.cluster_cnn.graph_spice_embedder import GraphSPICEEmbedder

from pprint import pprint
from mlreco.utils.cluster.cluster_graph_constructor import ClusterGraphConstructor


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

    MODULES = ['constructor_cfg', 'embedder_cfg', 'kernel_cfg', 'gspice_fragment_manager']

    def __init__(self, cfg, name='graph_spice'):
        super(MinkGraphSPICE, self).__init__()
        self.model_config = cfg.get(name, {})
        self.skip_classes = self.model_config.get('skip_classes', [2, 3, 4])
        self.dimension = self.model_config.get('dimension', 3)
        self.embedder_name = self.model_config.get('embedder', 'graph_spice_embedder')
        self.embedder = GraphSPICEEmbedder(self.model_config.get('embedder_cfg', {}))
        self.node_dim = self.model_config.get('node_dim', 16)

        self.kernel_cfg = self.model_config.get('kernel_cfg', {})
        self.kernel_fn = gs_kernel_construct(self.kernel_cfg)

        constructor_cfg = self.model_config.get('constructor_cfg', {})

        self.use_raw_features = self.model_config.get('use_raw_features', False)

        # Cluster Graph Manager
        # `training` needs to be set at forward time.
        # Before that, self.training is always True.
        self.gs_manager = ClusterGraphConstructor(constructor_cfg,
                                                batch_col=0)


    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

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
        self.gs_manager.training = self.training
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


class GraphSPICELoss(nn.Module):

    def __init__(self, cfg, name='graph_spice_loss'):
        super(GraphSPICELoss, self).__init__()
        self.loss_config = cfg.get(name, {})
        self.loss_name = self.loss_config.get('name', 'se_lovasz_inter')
        self.skip_classes = self.loss_config.get('skip_classes', [2, 3, 4])
        # We use the semantic label -1 to account
        # for semantic prediction mistakes.
        # self.skip_classes += [-1]
        self.eval_mode = self.loss_config.get('eval', False)
        self.loss_fn = spice_loss_construct(self.loss_name)(self.loss_config)

        constructor_cfg = self.loss_config.get('constructor_cfg', {})
        self.gs_manager = ClusterGraphConstructor(constructor_cfg,
                                                batch_col=0,
                                                training=~self.eval_mode)

        self.invert = self.loss_config.get('invert', False)
        # print("LOSS FN = ", self.loss_fn)

    def filter_class(self, segment_label, cluster_label):
        '''
        Filter classes according to segmentation label.
        '''
        mask = ~np.isin(segment_label[0][:, -1].cpu().numpy(), self.skip_classes)
        slabel = [segment_label[0][mask]]
        clabel = [cluster_label[0][mask]]
        return slabel, clabel


    def forward(self, result, segment_label, cluster_label):
        '''

        '''
        slabel, clabel = self.filter_class(segment_label, cluster_label)

        graph = result['graph'][0]
        graph_info = result['graph_info'][0]
        self.gs_manager.replace_state(graph, graph_info)
        result['edge_score'] = [graph.edge_attr]
        result['edge_index'] = [graph.edge_index]
        if not self.eval_mode:
            result['edge_truth'] = [graph.edge_truth]

        if self.invert:
            pred_labels = result['edge_score'][0] < 0.0
        else:
            pred_labels = result['edge_score'][0] >= 0.0

        if not self.eval_mode:

            edge_diff = pred_labels != (result['edge_truth'][0] > 0.5)

            # print("Number of Wrong Edges = {} / {}".format(
            #     torch.sum(edge_diff).item(), edge_diff.shape[0]))
            #
            # print("Number of True Dropped Edges = {} / {}".format(
            #     torch.sum(result['edge_truth'][0] < 0.5).item(),
            #     edge_diff.shape[0]))

        res = self.loss_fn(result, slabel, clabel)
        return res
