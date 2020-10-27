import torch
import torch.nn as nn
import numpy as np
import sparseconvnet as scn

from .cluster_cnn.spatial_embeddings import SpatialEmbeddings1, SpatialEmbeddingsLite
from .cluster_cnn.losses.spatial_embeddings import *
from .cluster_cnn import cluster_model_construct, backbone_construct, spice_loss_construct

class ClusterCNN(SpatialEmbeddings1):
    '''
    UResNet with coordinate convolution block in final layer for clustering.

    Congifurations:
        - coordConv: Option to concat coordinates to input features at
        final linear layer.
        - embedding_dim: dimension of final embedding space for clustering.
    '''

    MODULES = ['network_base', 'uresnet', 'spice_loss', 'spatial_embeddings']

    def __init__(self, cfg):
        super(ClusterCNN, self).__init__(cfg)
        #print(self)


class ClusterCNN2(SpatialEmbeddingsLite):
    '''
    UResNet with coordinate convolution block in final layer for clustering.

    Congifurations:
        - coordConv: Option to concat coordinates to input features at
        final linear layer.
        - embedding_dim: dimension of final embedding space for clustering.
    '''

    MODULES = ['network_base', 'uresnet', 'spice_loss', 'spatial_embeddings']

    def __init__(self, cfg):
        super(ClusterCNN2, self).__init__(cfg)


# class ClusterCNN3(SpatialEmbeddings3):

#     def __init__(self, cfg):
#         super(ClusterCNN3, self).__init__(cfg)


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
