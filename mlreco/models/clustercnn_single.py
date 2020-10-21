import torch
import torch.nn as nn
import numpy as np
import sparseconvnet as scn

from .cluster_cnn.utils import add_normalized_coordinates
from .cluster_cnn.losses.single_layers import DiscriminativeLoss
from mlreco.models.uresnet import UResNet


class ClusterCNN(UResNet):
    '''
    UResNet with coordinate convolution block in final layer for clustering.

    Congifurations:
        - coordConv: Option to concat coordinates to input features at
        final linear layer.
        - embedding_dim: dimension of final embedding space for clustering.
    '''

    MODULES = ['clustercnn_single', 'spice_loss']

    def __init__(self, cfg, name='clustercnn_single'):
        super(ClusterCNN, self).__init__(cfg, name)
        self._coordConv = self._model_config.get('coordConv', False)
        self._embedding_dim = self._model_config.get('embedding_dim', 8)
        m = self._model_config.get('filters', 16)
        if self._coordConv:
            self.linear = torch.nn.Linear(m + self._dimension, self._embedding_dim)
        else:
            self.linear = torch.nn.Linear(m, self._embedding_dim)


    def forward(self, input):
        """
        point_cloud is a list of length minibatch size (assumes mbs = 1)
        point_cloud[0] has 3 spatial coordinates + 1 batch coordinate + 1 feature
        shape of point_cloud[0] = (N, 4)
        """
        point_cloud, = input
        coords = point_cloud[:, :-2].float()
        features = point_cloud[:, -1][:, None].float()
        fout = self.sparseModel((coords, features))

        if self._coordConv:
            normalized_coords = (coords - self.spatial_size / 2)\
                / float(self.spatial_size / 2)
            fout = torch.cat([normalized_coords, fout], dim=1)
        else:
            fout = self.sparseModel((coords, features))
        embedding = self.linear(fout)

        return {
            'cluster_feature': [embedding]
        }


class ClusteringLoss(DiscriminativeLoss):
    '''
    Vanilla discriminative clustering loss applied to final embedding layer.
    '''
    def __init__(self, cfg, name='spice_loss'):
        super(ClusteringLoss, self).__init__(cfg)
