from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import sparseconvnet as scn


class SelectionFeatures(torch.nn.Module):
    def __init__(self, dimension, spatial_size):
        import sparseconvnet as scn
        super(SelectionFeatures, self).__init__()
        self.input_layer = scn.InputLayer(dimension, spatial_size, mode=3)

    def forward(self, input):
        print('selection features', len(input), input[0], input[1])
        return input


class AddLabels(torch.nn.Module):
    def __init__(self):
        super(AddLabels, self).__init__()

    def forward(self, attention, label):
        output = scn.SparseConvNetTensor()
        output.metadata = attention.metadata
        output.spatial_size = attention.spatial_size
        output.features = attention.features.new().resize_(1).expand_as(attention.features).fill_(1.0)
        output.features = output.features * attention.features
        positions = attention.get_spatial_locations().cuda()
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


class ExtractFeatureMap(torch.nn.Module):
    def __init__(self, i, dimension, spatial_size):
        """
        i such that 2**i * small_spatial_size = big_spatial_size
        spatial size of output
        """
        super(ExtractFeatureMap, self).__init__()
        import sparseconvnet as scn
        self.i = i
        self.input_layer = scn.InputLayer(dimension, spatial_size, mode=3)

    def forward(self, x, y):
        """
        x is feature map with smallest spatial size
        y is output feature map with biggest spatial size
        x.features.shape = (N1, N_features)
        x.get_spatial_locations().size() = (N1, 4) (dim + batch_id)
        coords.size() = (N2, 4) in original image size (bigger spatial size)
        Returns (N2, N_features)
        """
        # FIXME no grad?
        # with torch.no_grad():
        # TODO deal with batch id
        # print('expand', i, x.features.shape, x.spatial_size, x.get_spatial_locations().size())
        feature_map = x.get_spatial_locations().cuda().float()
        coords = y.get_spatial_locations().cuda().float()
        N1 = feature_map.size(0)
        N2 = coords.size(0)

        print('N1 = %d, N2 = %d, 2**i = %d' % (N1, N2, 2**self.i))
        # shape (N1, N2, 4) for next 2 lines
        feature_map_coords = (feature_map * (2**self.i))[:, None, :].expand(-1, N2, -1)
        coords_adapted = coords[None, ...].expand(N1, -1, -1)
        part1 = feature_map_coords <= coords_adapted
        part1 = part1.all(dim=-1)
        part2 = feature_map_coords + 2**self.i > coords_adapted
        part2 = part2.all(dim=-1)
        index = part1 & part2  # shape (N1, N2)
        # Make sure that all pixels from original belong to 1 only in feature
        print((index.long().sum(dim=0)>=1).long().sum())
        print((index.long().sum(dim=0)!=1).long().sum())
        print((index.long().sum(dim=1)!=1).long().sum())
        final_features = torch.index_select(x.features, 0, torch.t(index).argmax(dim=1))
        print(final_features.size())
        final_coords = torch.index_select(feature_map, 0, torch.t(index).argmax(dim=1))
        print(final_coords.size())

        return self.input_layer((final_coords, final_features))
