import torch
import torch.nn as nn
import numpy as np
import sparseconvnet as scn
from collections import defaultdict

from torch_geometric.nn import GINConv, knn_graph

from mlreco.models.cluster_cnn.losses.gs_embeddings import *

from mlreco.models.layers.base import SCNNetworkBase

from pprint import pprint

"""
I've borrowed the implementation details in Pytorch Geometric's exmaple
PointNet++ Implementations with minor adjustments:

https://github.com/rusty1s/pytorch_geometric/blob/master/examples/pointnet2_segmentation.py
https://github.com/rusty1s/pytorch_geometric/blob/master/examples/pointnet2_classification.py

"""

def MLP(channels, batch_norm=True): # Author: Matthias Fey (rusty1s)
    return nn.Sequential(*[
        nn.Sequential(nn.Linear(channels[i - 1], channels[i]), 
            nn.LeakyReLU(0.33), nn.BatchNorm1d(channels[i]))
        for i in range(1, len(channels))
    ])

class GINModule(torch.nn.Module):
    def __init__(self, mlp, k=10, eps=0, train_eps=False, **kwargs):
        super(GINModule, self).__init__()
        self.k = k
        self.mlp = mlp
        self.conv = GINConv(self.mlp, eps=eps, train_eps=train_eps, **kwargs)

    def forward(self, x, pos, batch):
        edge_index = knn_graph(pos, self.k, batch, loop=False)
        # print(edge_index)
        x = self.conv(x, edge_index)
        return x


class PointNet2(nn.Module):

    def __init__(self, cfg, name='pointnet2'):
        super(PointNet2, self).__init__()
        self.dimension = 3
        self.spatial_size = 768
        self.coordConv = True

        self.gin1 = GINModule(MLP([4, 16, 16, 32]))
        self.gin2 = GINModule(MLP([32, 64, 64, 64]))
        self.gin3 = GINModule(MLP([64, 128, 128, 128]))
        # self.mlp2 = MLP([32, 64, 64, 128])
        self.mlp3 = MLP([128, 256, 256, 1024])
        self.final = nn.Linear(1024, 22)

    def forward(self, input):

        point_cloud, = input
        coords = point_cloud[:, 0:self.dimension].float()
        features = point_cloud[:, self.dimension+1:].float()
        batch = point_cloud[:, 3]
        features = features[:, -1].view(-1, 1)

        normalized_coords = (coords[:, :3] - float(self.spatial_size) / 2) \
                    / (float(self.spatial_size) / 2)
        if self.coordConv:
            features = torch.cat([normalized_coords, features], dim=1)

        batch = batch.to(torch.long)

        x = self.gin1(features, coords, batch)
        x = self.gin2(x, coords, batch)
        x = self.gin3(x, coords, batch)
        # x = self.mlp2(x)
        x = self.mlp3(x)
        out = self.final(x)

        return out