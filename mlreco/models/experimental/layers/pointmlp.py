from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MLP, PointNetConv, fps, global_max_pool, knn
from torch_geometric.nn.conv import MessagePassing

from torch_geometric.utils import scatter
from .pointnet import GlobalSAModule


# Adapted from PointMLP Github, modified to match lartpc_mlreco3d:
# https://github.com/ma-xu/pointMLP-pytorch/blob/main/classification_ModelNet40/models/pointmlp.py


class PointMLPConv(MessagePassing):

    def __init__(self, in_features, out_features, k=24, fps_ratio=0.5):
        super(PointMLPConv, self).__init__(aggr='max')

        self.pre_block = PreBlock(in_features, out_features)
        self.pos_block = PosBlock(out_features)

        self.alpha = nn.Parameter(torch.ones(in_features))
        self.beta = nn.Parameter(torch.zeros(in_features))

        self.k = k
        self.ratio = fps_ratio

    def reset_parameters(self):
        
        super().reset_parameters()
        
        self.pre_block.reset_parameters()
        self.pos_block.reset_parameters()
        
        self.alpha.fill(1.0)
        self.beta.zero_()

    def forward(self, x, pos, batch):

        n, d = x.shape

        idx = fps(pos, batch, ratio=self.ratio)

        # row (runs over x[idx]), col (runs over x)
        # For each element in pos[idx] (anchors), find self.k nearest
        # neighbors in pos (all points)
        
        # row, col = knn(pos, pos[idx], self.k, batch, batch[idx])
        
        anchors, neighbors = knn(pos, pos[idx], self.k, batch, batch[idx])

        # msgs from edge_index[0] are sent to edge_index[1]
        edge_index = torch.stack([neighbors, anchors], dim=0)

        # Compute norm (Geometric Affine Module)
        var_dst = scatter((x[neighbors] - x[anchors])**2 / (self.k * n * d), anchors, reduce='sum')
        sigma = torch.sqrt(torch.clamp(var_dst.sum(), min=1e-6))
        norm = self.alpha / (sigma + 1e-5)
        
        x = x * norm + self.beta
        x = self.pre_block(x)
        
        # Max-pooling
        out = self.propagate(edge_index, x=x) # n_anchors

        # Apply Second Residual (PosBlock)
        out = self.pos_block(out[idx])

        return out, pos[idx], batch[idx]


class ConvBNReLU1D(torch.nn.Module):

    def __init__(self, in_channels, out_channels, bias=True):
        super(ConvBNReLU1D, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=bias),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class ConvBNReLURes1D(torch.nn.Module):

    def __init__(self, num_features, bn=True):
        super(ConvBNReLURes1D, self).__init__()

        self.linear_1 = nn.Linear(num_features, num_features, bias=not bn)
        self.linear_2 = nn.Linear(num_features, num_features, bias=not bn)

        self.bn_1 = nn.BatchNorm1d(num_features)
        self.bn_2 = nn.BatchNorm1d(num_features)

    def forward(self, x):

        out = self.linear_1(x)
        out = self.bn_1(out)
        out = F.relu(out)
        out = self.linear_2(out)
        out = self.bn_2(out)
        out = out + x
        return out
    

class PreBlock(torch.nn.Module):

    def __init__(self, in_features, out_features, num_blocks=1):
        super(PreBlock, self).__init__()

        blocks = []
        self.transfer = ConvBNReLU1D(in_features, out_features)
        for _ in range(num_blocks):
            blocks.append(ConvBNReLURes1D(out_features))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):

        x = self.transfer(x)
        x = self.net(x)
        return x
    

class PosBlock(nn.Module):

    def __init__(self, out_features, num_blocks=1):
        super(PosBlock, self).__init__()

        blocks = []
        for _ in range(num_blocks):
            blocks.append(ConvBNReLURes1D(out_features))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.net(x)
        return x
    

class GlobalPooling(torch.nn.Module):

    def __init__(self):
        super(GlobalPooling, self).__init__()

    def forward(self, x, pos, batch):
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch
    

class PointMLPEncoder(torch.nn.Module):

    def __init__(self, cfg, name='pointmlp_encoder'):
        super(PointMLPEncoder, self).__init__()
        self.model_cfg = cfg['pointmlp_encoder']

        self.k           = self.model_cfg.get('num_kneighbors', 24)
        self.mlp_specs   = self.model_cfg.get('mlp_specs', [64, 128, 256, 512])
        self.ratio_specs = self.model_cfg.get('ratio_specs', [0.25, 0.5, 0.5, 0.5])
        self.classifier_specs = self.model_cfg.get('classifier_specs', [512, 256, 128])
        assert len(self.mlp_specs) == len(self.ratio_specs)

        self.init_embed = nn.Linear(1, self.mlp_specs[0])
        convs = []
        for i in range(len(self.mlp_specs)-1):
            convs.append(PointMLPConv(self.mlp_specs[i], 
                                      self.mlp_specs[i+1], 
                                      k=self.k, 
                                      fps_ratio=self.ratio_specs[i]))


        self.net = nn.Sequential(*convs)
        self.latent_size = self.mlp_specs[-1]

        self.global_pooling = GlobalPooling()

        self.classifier = []
        for i in range(len(self.classifier_specs)-1):
            fin, fout = self.classifier_specs[i], self.classifier_specs[i+1]
            m = nn.Sequential(
                nn.Linear(fin, fout),
                nn.BatchNorm1d(fout),
                nn.ReLU(),
                nn.Dropout()
            )
            self.classifier.append(m)
        self.latent_size = self.classifier_specs[-1]
        self.classifier = nn.Sequential(*self.classifier)


    def forward(self, data):
        x, pos, batch = data.x, data.pos, data.batch
        x = self.init_embed(x)
        for i, layer in enumerate(self.net):
            out = layer(x, pos, batch)
            x, pos, batch = out
            # print("{} = ".format(i), x.shape, pos.shape, batch.shape)
        x, pos, batch = self.global_pooling(x, pos, batch)

        out = self.classifier(x)
        return out