import torch
import numpy as np
import MinkowskiEngine as ME

from mlreco.models.layers.cluster_cnn.losses.gs_embeddings import *
from mlreco.models.layers.cluster_cnn import gs_kernel_construct, spice_loss_construct

from mlreco.models.layers.cluster_cnn.graph_spice_embedder import GraphSPICEEmbedder

from pprint import pprint
from mlreco.utils.cluster.cluster_graph_constructor import ClusterGraphConstructor


class GraphSPICE(nn.Module):
    '''
    Neighbor-graph embedding based particle clustering.

    GraphSPICE has two components:

    1. Voxel Embedder: UNet-type CNN architecture used for feature
    extraction and feature embeddings.

    2. Edge Probability Kernel function: A kernel function (any callable
    that takes two node attribute vectors to give a edge proability score).

    Prediction is done in two steps:

    1. A neighbor graph (ex. KNN, Radius) is constructed to compute
    edge probabilities between neighboring edges.

    2. Edges with low probability scores are dropped.
    
    3. The voxels are clustered by counting connected components.

    Configuration
    -------------
    skip_classes: list, default [2, 3, 4]
        semantic labels for which to skip voxel clustering
        (ex. Michel, Delta, and Low Es rarely require neural network clustering)
    dimension: int, default 3
        Spatial dimension (2 or 3).
    min_points: int, default 0
        If a value > 0 is specified, this will enable the orphans assignment for
        any predicted cluster with voxel count < min_points.

        .. warning::

            ``min_points`` is set to 0 at training time.

    node_dim: int
    use_raw_features: bool
    constructor_cfg: dict
        Configuration for ClusterGraphConstructor instance. A typical configuration:

        .. code-block:: yaml

              constructor_cfg:
                mode: 'knn'
                seg_col: -1
                cluster_col: 5
                edge_mode: 'attributes'
                hyper_dimension: 22
                edge_cut_threshold: 0.1

        .. warning::

            ``edge_cut_threshold`` is set to 0. at training time.
            At inference time you want to set it to a value > 0.
            As a rule of thumb, 0.1 is a good place to start.
            Its exact value can be optimized.

    embedder_cfg: dict
        A typical configuration would look like:

        .. code-block:: yaml

              embedder_cfg:
                graph_spice_embedder:
                  segmentationLayer: False
                  feature_embedding_dim: 16
                  spatial_embedding_dim: 3
                  num_classes: 5
                  occupancy_mode: 'softplus'
                  covariance_mode: 'softplus'
                uresnet:
                  filters: 32
                  input_kernel: 5
                  depth: 5
                  reps: 2
                  spatial_size: 768
                  num_input: 4 # 1 feature + 3 normalized coords
                  allow_bias: False
                  activation:
                    name: lrelu
                    args:
                      negative_slope: 0.33
                  norm_layer:
                    name: batch_norm
                    args:
                      eps: 0.0001
                      momentum: 0.01

    kernel_cfg: dict
        A typical configuration:

        .. code-block:: yaml

              kernel_cfg:
                name: 'bilinear'
                num_features: 32

    .. warning::

        Train time and test time configurations are slightly different for GraphSpice.

    Output
    ------
    graph:
    graph_info:
    coordinates:
    hypergraph_features:

    See Also
    --------
    GraphSPICELoss
    '''

    MODULES = ['constructor_cfg', 'embedder_cfg', 'kernel_cfg', 'gspice_fragment_manager']

    RETURNS = {
        'image_id'     : ['tensor'],
        'coordinates'  : ['tensor'],
        'batch'        : ['tensor', 'image_id'],
        'x'            : ['tensor', 'image_id'],
        'pos'          : ['tensor', 'image_id'],
        'node_truth'   : ['tensor', 'image_id'],
        'voxel_id'     : ['tensor', 'image_id'],
        'graph_key'    : ['tensor'],
        'graph_id'     : ['tensor', 'graph_key'],
        'semantic_id'  : ['tensor', 'image_id'],
        'full_edge_index'   : ['edge_tensor', ['full_edge_index', 'image_id']],
        'edge_index'   : ['edge_tensor', ['full_edge_index', 'image_id']],
        'edge_batch'   : ['edge_tensor', ['full_edge_index', 'image_id']],
        'edge_image_id': ['edge_tensor', ['full_edge_index', 'image_id']],
        'edge_label'   : ['edge_tensor', ['full_edge_index', 'image_id']],
        'edge_attr'    : ['edge_tensor', ['full_edge_index', 'image_id']],
        'edge_pred'    : ['edge_tensor', ['full_edge_index', 'image_id']],
        'edge_prob'    : ['edge_tensor', ['full_edge_index', 'image_id']]
    }

    def __init__(self, cfg, name='graph_spice'):
        super(GraphSPICE, self).__init__()
        self.model_config = cfg.get(name, {})
        self.skip_classes = self.model_config.get('skip_classes', [2, 3, 4])
        self.dimension = self.model_config.get('dimension', 3)
        self.embedder_name = self.model_config.get('embedder', 'graph_spice_embedder')
        self.embedder = GraphSPICEEmbedder(self.model_config.get('embedder_cfg', {}))
        self.node_dim = self.model_config.get('node_dim', 16)

        self.kernel_cfg = self.model_config.get('kernel_cfg', {})
        self.kernel_fn = gs_kernel_construct(self.kernel_cfg)
        self.invert = self.model_config.get('invert', True)

        constructor_cfg = self.model_config.get('constructor_cfg', {})

        self.use_raw_features = self.model_config.get('use_raw_features', False)

        # Cluster Graph Manager
        # `training` needs to be set at forward time.
        # Before that, self.training is always True.
        self.gs_manager = ClusterGraphConstructor(constructor_cfg)

        self.RETURNS.update(self.embedder.RETURNS)


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
        # Pass input through the model
        self.gs_manager.training = self.training
        point_cloud, labels = self.filter_class(input)
        res = self.embedder([point_cloud])

        res['coordinates'] = [point_cloud[:, :4]]
        if self.use_raw_features:
            res['hypergraph_features'] = res['features']

        # Build the graph
        graph = self.gs_manager(res,
                                self.kernel_fn,
                                labels,
                                invert=self.invert)
        
        graph_state = self.gs_manager.save_state(unwrapped=False)
        res.update(graph_state)

        return res


class GraphSPICELoss(nn.Module):
    """
    Loss function for GraphSpice.

    Configuration
    -------------
    name: str, default 'se_lovasz_inter'
        Loss function to use.
    invert: bool, default True
        You want to leave this to True for statistical weighting purpose.
    kernel_lossfn: str
    edge_loss_cfg: dict
        For example

        .. code-block:: yaml

          edge_loss_cfg:
            loss_type: 'LogDice'

    eval: bool, default False
        Whether we are in inference mode or not.

        .. warning::

            Currently you need to manually switch ``eval`` to ``True``
            when you want to run the inference, as there is no way (?)
            to know from within the loss function whether we are training
            or not.

    Output
    ------
    To be completed.

    See Also
    --------
    GraphSPICE
    """

    RETURNS = {}

    def __init__(self, cfg, name='graph_spice_loss'):
        super(GraphSPICELoss, self).__init__()
        self.model_config = cfg.get('graph_spice', {})
        self.loss_config = cfg.get(name, {})

        self.loss_name = self.loss_config.get('name', 'se_lovasz_inter')
        self.skip_classes = self.model_config.get('skip_classes', [2, 3, 4])
        # We use the semantic label -1 to account
        # for semantic prediction mistakes.
        # self.skip_classes += [-1]
        # self.eval_mode = self.loss_config.get('eval', False)
        self.loss_fn = spice_loss_construct(self.loss_name)(self.loss_config)

        self.RETURNS.update(self.loss_fn.RETURNS)

        constructor_cfg = self.model_config.get('constructor_cfg', {})
        self.gs_manager = ClusterGraphConstructor(constructor_cfg)

        self.invert = self.loss_config.get('invert', True)
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
        # self.gs_manager.replace_state(result)
        self.gs_manager.load_state(result, unwrapped=False)

        # if self.invert:
        #     pred_labels = result['edge_score'][0] < 0.0
        # else:
        #     pred_labels = result['edge_score'][0] >= 0.0
        # edge_diff = pred_labels != (result['edge_truth'][0] > 0.5)

        # print("Number of Wrong Edges = {} / {}".format(
        #     torch.sum(edge_diff).item(), edge_diff.shape[0]))

        # print("Number of True Dropped Edges = {} / {}".format(
        #     torch.sum(result['edge_truth'][0] < 0.5).item(),
        #     edge_diff.shape[0]))


        slabel, clabel = self.filter_class(segment_label, cluster_label)
        res = self.loss_fn(result, slabel, clabel)
        return res
