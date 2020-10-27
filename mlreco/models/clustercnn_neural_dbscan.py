import torch
import torch.nn as nn
import numpy as np
import sparseconvnet as scn

from .cluster_cnn.spatial_embeddings import CoordinateEmbeddings
from .cluster_cnn.losses.radius_nnloss import *


class ClusterCNN(CoordinateEmbeddings):
    '''
    UResNet with coordinate convolution block in final layer for clustering.

    Congifurations:
        - coordConv: Option to concat coordinates to input features at
        final linear layer.
        - embedding_dim: dimension of final embedding space for clustering.
    '''

    MODULES = ['network_base', 'uresnet', 'coordinate_embeddings']

    def __init__(self, cfg):
        super(ClusterCNN, self).__init__(cfg)


class ClusteringLoss(nn.Module):
    '''
    Loss function for Proposal-Free Mask Generators.
    '''
    def __init__(self, cfg, name='spice_loss'):
        super(ClusteringLoss, self).__init__()

        self.loss_func = DensityBasedNNLoss(cfg)
        #print(self.loss_func)

    def forward(self, result, cluster_label):
        segment_label = [cluster_label[0][:, [0, 1, 2, 3, -1]]]
        group_label = [cluster_label[0][:, [0, 1, 2, 3, 5]]]
        return self.loss_func(result, segment_label, group_label)