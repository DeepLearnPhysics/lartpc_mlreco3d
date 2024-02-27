import numpy as np
import torch
import torch.nn as nn
import time

import MinkowskiEngine as ME
import MinkowskiFunctional as MF

from mlreco.models.layers.common.ppnplus import PPN, PPNLonelyLoss
from mlreco.models.uresnet import SegmentationLoss
from collections import defaultdict
from mlreco.models.uresnet_ppn_chain import UResNetPPN
from mlreco.utils.globals import SHAPE_COL

from mlreco.utils.cluster.cluster_graph_constructor import ClusterGraphConstructor
from mlreco.models.layers.cluster_cnn.graph_spice_embedder import GraphSPICEEmbedder
from mlreco.utils.cluster.fragmenter import GraphSPICEFragmentManager
from mlreco.models.layers.cluster_cnn.losses.gs_embeddings import *
from mlreco.models.layers.cluster_cnn import gs_kernel_construct, spice_loss_construct
from mlreco.models.graph_spice import GraphSPICE

class UResNetPPNGSPICE(nn.Module):
    """
    A model made of UResNet backbone and PPN layers, including GraphSPICE for clustering. 
    
    Typical configuration:

    .. code-block:: yaml

        model:
          name: uresnet_ppn_chain
          modules:
            uresnet_lonely:
              # Your uresnet config here
            ppn:
              # Your ppn config here

    Configuration
    -------------
    data_dim: int, default 3
    num_input: int, default 1
    allow_bias: bool, default False
    spatial_size: int, default 512
    leakiness: float, default 0.33
    activation: dict
        For activation function, defaults to `{'name': 'lrelu', 'args': {}}`
    norm_layer: dict
        For normalization function, defaults to `{'name': 'batch_norm', 'args': {}}`

    depth: int, default 5
        Depth of UResNet, also corresponds to how many times we down/upsample.
    filters: int, default 16
        Number of filters in the first convolution of UResNet.
        Will increase linearly with depth.
    reps: int, default 2
        Convolution block repetition factor
    input_kernel: int, default 3
        Receptive field size for very first convolution after input layer.

    num_classes: int, default 5
    score_threshold: float, default 0.5
    classify_endpoints: bool, default False
        Enable classification of points into start vs end points.
    ppn_resolution: float, default 1.0
    ghost: bool, default False
    downsample_ghost: bool, default True
    use_true_ghost_mask: bool, default False
    mask_loss_name: str, default 'BCE'
        Can be 'BCE' or 'LogDice'
    particles_label_seg_col: int, default -2
        Which column corresponds to particles' semantic label
    track_label: int, default 1

    See Also
    --------
    mlreco.models.uresnet.UResNet_Chain, mlreco.models.layers.common.ppnplus.PPN
    """
    MODULES = ['mink_uresnet', 'mink_uresnet_ppn_chain', 'mink_ppn']

    RETURNS = dict(UResNetPPN.RETURNS, **PPN.RETURNS)
    RETURNS.update(GraphSPICE.RETURNS)

    def __init__(self, cfg):
        super(UResNetPPNGSPICE, self).__init__()
        self.model_config = cfg

        # UResNet-PPN
        self.uresnet_ppn = UResNetPPN(cfg)
        
        # Graph-SPICE stuff
        
        self.kernel_cfg = self.model_config.get('kernel_cfg', {})
        self.kernel_fn = gs_kernel_construct(self.kernel_cfg)
        self.invert = self.model_config.get('invert', True)
        constructor_cfg = self.model_config.get('constructor_cfg', {})
        self.gs_manager = ClusterGraphConstructor(constructor_cfg)
        self.make_fragments = self.model_config.get('make_fragments', False)
        self.batch_col = 0
        if self.make_fragments:
            self._gspice_fragment_manager = GraphSPICEFragmentManager(
                cfg.get('graph_spice', {}).get('gspice_fragment_manager', {}),
                batch_col=self.batch_col)
            
            self.RETURNS.update(GraphSPICEEmbedder.RETURNS)
            
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
    
    def construct_fragments(self, input):
        
        frags = {}
        
        device = input[0].device
        semantic_labels = input[1][:, SHAPE_COL]
        filtered_semantic = ~(semantic_labels[..., None] == \
                                torch.tensor(self.skip_classes, device=device)).any(-1)
        graphs = self.gs_manager.fit_predict()
        perm = torch.argsort(graphs.voxel_id)
        cluster_predictions = graphs.node_pred[perm]
        filtered_input = torch.cat([input[0][filtered_semantic][:, :4],
                                    semantic_labels[filtered_semantic].view(-1, 1),
                                    cluster_predictions.view(-1, 1)], dim=1)

        fragments = self._gspice_fragment_manager(filtered_input, input[0], filtered_semantic)
        frags['filtered_input'] = [filtered_input]
        frags['fragment_batch_ids'] = [np.array(fragments[1])]
        frags['fragment_clusts'] = [np.array(fragments[0])]
        frags['fragment_seg'] = [np.array(fragments[2]).astype(int)]
        
        return frags
    
    def forward_gspice(self, input):
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
        
        if self.make_fragments:
            frags = self.construct_fragments(input)
            res.update(frags)
        
        graph_state = self.gs_manager.save_state(unwrapped=False)
        res.update(graph_state)

        return res

    def forward(self, input):
        
        raise NotImplementedError("This model is not yet implemented.")
        
        out = {}
        res_unet_ppn = self.uresnet_ppn(input)

        res_gspice = self.forward_gspice(input)

        return out


class UResNetPPNGSPICELoss(nn.Module):
    """
    See Also
    --------
    mlreco.models.uresnet.SegmentationLoss, mlreco.models.layers.common.ppnplus.PPNLonelyLoss
    """

    RETURNS = {
        'loss': ['scalar'],
        'accuracy': ['scalar']
    }

    def __init__(self, cfg):
        super(UResNetPPNGSPICELoss, self).__init__()
        self.ppn_loss = PPNLonelyLoss(cfg)
        self.segmentation_loss = SegmentationLoss(cfg)
        
        self.gspice_loss_config     = cfg['graph_spice_loss']
        self.gspice_loss_name       = self.gspice_loss_config.get('name', 'graph_spice_edge_only_loss')
        self.gspice_loss            = spice_loss_construct(self.gspice_loss_name)(self.gspice_loss_config)
        self.gspice_config          = cfg.get('graph_spice', {})
        constructor_cfg             = self.gspice_config.get('constructor_cfg', {})
        self.gs_manager             = ClusterGraphConstructor(constructor_cfg)
        self.invert                 = self.gspice_loss_config.get('invert', True)
        self.evaluate_true_accuracy = self.gspice_loss_config.get('evaluate_true_accuracy', False)

        self.RETURNS.update({'segmentation_'+k:v for k, v in self.segmentation_loss.RETURNS.items()})
        self.RETURNS.update({'ppn_'+k:v for k, v in self.ppn_loss.RETURNS.items()})
        self.RETURNS.update(self.gspice_loss.RETURNS)
        
    def filter_class(self, segment_label, cluster_label):
        '''
        Filter classes according to segmentation label.
        '''
        mask = ~np.isin(segment_label[0][:, -1].cpu().numpy(), self.skip_classes)
        slabel = [segment_label[0][mask]]
        clabel = [cluster_label[0][mask]]
        return slabel, clabel

    def forward(self, outputs, segment_label, particles_label, cluster_label, weights=None):

        res_segmentation = self.segmentation_loss(
            outputs, segment_label, weights=weights)

        res_ppn = self.ppn_loss(
            outputs, segment_label, particles_label)
        
        # Clustering
        slabel, clabel = self.filter_class(segment_label, cluster_label)
        self.gs_manager.load_state(outputs, unwrapped=False)
        res_gspice = self.loss_fn(outputs, slabel, clabel)
        
        if self.evaluate_true_accuracy:
            self.gs_manager.fit_predict()
            acc_out = self.gs_manager.evaluate()
            for key, val in acc_out.items():
                res_gspice[key] = val
        

        res = {
            'loss': res_segmentation['loss'] + res_ppn['loss'],
            'accuracy': (res_segmentation['accuracy'] + res_ppn['accuracy'])/2
        }

        res.update({'segmentation_'+k:v for k, v in res_segmentation.items()})
        res.update({'ppn_'+k:v for k, v in res_ppn.items()})
        res.update({'gspice_'+k:v for k, v in res_gspice.items()})
        
        print(res['segmentation_loss'])
        print(res['ppn_loss'])
        print(res['gspice_loss'])


        return res
