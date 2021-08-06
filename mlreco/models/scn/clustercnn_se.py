import torch
import torch.nn as nn

from .cluster_cnn.spatial_embeddings import SpatialEmbeddings
from .cluster_cnn.losses.spatial_embeddings import *
from .cluster_cnn import cluster_model_construct, backbone_construct, spice_loss_construct

class ClusterCNN(SpatialEmbeddings):
    '''
    UResNet with coordinate convolution block in final layer for clustering.

    Configurations:

    - coordConv: Option to concat coordinates to input features at
    final linear layer.
    - embedding_dim: dimension of final embedding space for clustering.

    A typical configuration of Spice would look like this (`spice` goes
    directly under the `modules` section of the configuration file):

    ..  code-block:: yaml

        spice:
          network_base:
            spatial_size: 768
            data_dim: 3
            features: 4
            leakiness: 0.33
          spatial_embeddings:
            seediness_dim: 1
            sigma_dim: 1
            embedding_dim: 3
            coordConv: True
            # model_path: 'your_weight.ckpt'
          uresnet:
            filters: 64
            input_kernel_size: 7
            num_strides: 7
            reps: 2
          fragment_clustering:
            s_thresholds: [0., 0., 0., 0.35]
            p_thresholds: [0.95, 0.95, 0.95, 0.95]
            cluster_all: False
            cluster_classes: [1]
            min_frag_size: 10
            min_voxels: 2
        spice_loss:
          name: se_vectorized_inter
          seediness_weight: 1.0
          embedding_weight: 1.0
          smoothing_weight: 1.0
          min_voxels: 2
          mask_loss_fn: lovasz_hinge
    '''

    MODULES = ['network_base', 'uresnet', 'spice_loss', 'spatial_embeddings']

    def __init__(self, cfg):
        super(ClusterCNN, self).__init__(cfg)
        #print(self)
        #print('Total Number of Trainable Parameters = {}'.format(
        #            sum(p.numel() for p in self.parameters() if p.requires_grad)))


class ClusteringLoss(nn.Module):
    '''
    Loss function for Proposal-Free Mask Generators.
    '''
    def __init__(self, cfg, name='spice_loss'):
        super(ClusteringLoss, self).__init__()

        self.loss_config = cfg[name]

        self.loss_func_name = self.loss_config.get('name', 'se_lovasz_inter')
        self.loss_func = spice_loss_construct(self.loss_func_name)
        self.loss_func = self.loss_func(cfg)

    def forward(self, result, cluster_label):
        segment_label = [cluster_label[0][:, [0, 1, 2, 3, -1]]]
        group_label = [cluster_label[0][:, [0, 1, 2, 3, 5]]]
        return self.loss_func(result, segment_label, group_label)
