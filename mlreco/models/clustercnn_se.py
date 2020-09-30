import torch
import torch.nn as nn
import numpy as np
import sparseconvnet as scn

from .cluster_cnn.spatial_embeddings import SpatialEmbeddings1, SpatialEmbeddings2
from .cluster_cnn.losses.spatial_embeddings import *
from .cluster_cnn import cluster_model_construct, backbone_construct, clustering_loss_construct

class ClusterCNN(SpatialEmbeddings1):
    '''
    UResNet with coordinate convolution block in final layer for clustering.

    Congifurations:
        - coordConv: Option to concat coordinates to input features at
        final linear layer.
        - embedding_dim: dimension of final embedding space for clustering.
    '''

    MODULES = ['network_base', 'uresnet', 'clustering_loss', 'spatial_embeddings']

    def __init__(self, cfg):
        super(ClusterCNN, self).__init__(cfg)
        #print(self)


class ClusterCNN2(SpatialEmbeddings2):
    '''
    UResNet with coordinate convolution block in final layer for clustering.

    Congifurations:
        - coordConv: Option to concat coordinates to input features at
        final linear layer.
        - embedding_dim: dimension of final embedding space for clustering.
    '''

    MODULES = ['network_base', 'uresnet', 'clustering_loss', 'spatial_embeddings']

    def __init__(self, cfg):
        super(ClusterCNN2, self).__init__(cfg)


# class ClusterCNN3(SpatialEmbeddings3):

#     def __init__(self, cfg):
#         super(ClusterCNN3, self).__init__(cfg)


class ClusteringLoss(nn.Module):
    '''
    Loss function for Proposal-Free Mask Generators.
    '''
    def __init__(self, cfg, name='clustering_loss'):
        super(ClusteringLoss, self).__init__()

        self.loss_config = cfg[name]

        self.loss_func_name = self.loss_config.get('name', 'se_lovasz_inter')
        self.loss_func = clustering_loss_construct(self.loss_func_name)
        self.loss_func = self.loss_func(cfg)
        #print(self.loss_func)

    def forward(self, result, cluster_label):
        segment_label = [cluster_label[0][:, [0, 1, 2, 3, -1]]]
        group_label = [cluster_label[0][:, [0, 1, 2, 3, 5]]]
        return self.loss_func(result, segment_label, group_label)

# class ClusteringLoss1(MaskBCELoss2):

#     def __init__(self, cfg, name='clustering_loss'):
#         super(ClusteringLoss1, self).__init__(cfg)


# class ClusteringLoss2(MaskBCELossBivariate):

#     def __init__(self, cfg, name='clustering_loss'):
#         super(ClusteringLoss2, self).__init__(cfg)


# class ClusteringLoss3(MaskLovaszHingeLoss):

#     def __init__(self, cfg, name='clustering_loss'):
#         super(ClusteringLoss3, self).__init__(cfg)


# class ClusteringLoss4(MaskLovaszInterLoss):

#     def __init__(self, cfg, name='clustering_loss'):
#         super(ClusteringLoss4, self).__init__(cfg)


# class ClusteringLoss6(EllipsoidalKernelLoss):

#     def __init__(self, cfg, name='clustering_loss'):
#         super(ClusteringLoss6, self).__init__(cfg)

# class ClusteringLoss7(MaskFocalLoss):

#     def __init__(self, cfg, name='clustering_loss'):
#         super(ClusteringLoss7, self).__init__(cfg)

# class ClusteringLoss8(MaskWeightedFocalLoss):

#     def __init__(self, cfg, name='clustering_loss'):
#         super(ClusteringLoss8, self).__init__(cfg)
