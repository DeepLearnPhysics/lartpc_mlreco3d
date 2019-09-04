from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import sparseconvnet as scn
import numpy as np


class AddLabels(torch.nn.Module):
    def __init__(self):
        super(AddLabels, self).__init__()

    def forward(self, attention, label):
        output = scn.SparseConvNetTensor()
        output.metadata = attention.metadata
        output.spatial_size = attention.spatial_size
        output.features = attention.features.new().resize_(1).expand_as(attention.features).fill_(1.0)
        output.features = output.features * attention.features
        positions = attention.get_spatial_locations()
        if torch.cuda.is_available():
            positions = positions.cuda()
        # print(positions.max(), label.max())
        for l in label:
            index = (positions == l).all(dim=1)
            output.features[index] = 1.0
        return output

    def input_spatial_size(self, out_size):
        return out_size


class Multiply(torch.nn.Module):
    def __init__(self):
        super(Multiply, self).__init__()

    def forward(self, x, y):
        output = scn.SparseConvNetTensor()
        output.metadata = x.metadata
        output.spatial_size = x.spatial_size
        output.features = x.features.new().resize_(1).expand_as(x.features).fill_(0.0)
        output.features = x.features * y.features[:, 1][:, None]
        return output

    def input_spatial_size(self, out_size):
        return out_size


class Selection(torch.nn.Module):
    def __init__(self, threshold=0.5):
        super(Selection, self).__init__()
        self.threshold = threshold
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, scores):
        output = scn.SparseConvNetTensor()
        output.metadata = scores.metadata
        output.spatial_size = scores.spatial_size
        output.features = scores.features.new().resize_(1).expand_as(scores.features).fill_(1.0)
        output.features = output.features * (self.softmax(scores.features)[:, 1] > self.threshold).float()[:, None]
        return output

    def input_spatial_size(self, out_size):
        return out_size


class GhostMask(torch.nn.Module):
    def __init__(self):
        super(GhostMask, self).__init__()

    def forward(self, ghost_mask, coords, feature_map, factor=0.0):
        output = scn.SparseConvNetTensor()
        output.metadata = feature_map.metadata
        output.spatial_size = feature_map.spatial_size
        # First downsample the ghost mask
        scale_coords, unique_indices = np.unique(np.concatenate([np.floor(coords[:, :-1]/2**factor), coords[:, -1:]], axis=1), axis=0, return_index=True)
        perm2 = np.lexsort((scale_coords[:, 0], scale_coords[:, 1], scale_coords[:, 2], scale_coords[:, 3]))
        scale_ghost_mask = ghost_mask[unique_indices][perm2]
        # Now order the feature map and multiply with ghost mask
        ppn1_coords = feature_map.get_spatial_locations()
        perm = np.lexsort((ppn1_coords[:, 0], ppn1_coords[:, 1], ppn1_coords[:, 2], ppn1_coords[:, 3]))
        # Reverse permutation
        inv_perm = np.argsort(perm)
        new_ghost_mask = scale_ghost_mask[:, None][inv_perm].float()
        output.features = feature_map.features * new_ghost_mask
        return output, new_ghost_mask
