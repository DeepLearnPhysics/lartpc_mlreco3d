import torch
import torch.nn as nn

# For MinkowskiEngine
import MinkowskiEngine as ME
from MinkowskiNonlinearity import MinkowskiModuleBase

# Custom Normalization Layers

class MinkowskiPixelNorm(nn.Module):
    '''
    Pixel Normalization Layer for Sparse Tensors.
    PixelNorm layers were used in NVIDIA's ProGAN.

    This layer normalizes the feature vector in each
    pixel to unit length, and has no trainable weights.

    References:
        - NVIDIA ProGAN: https://arxiv.org/pdf/1710.10196.pdf
    '''
    def __init__(self,
                 num_features,
                 eps=1e-5,
                 dimension=3):
        super(MinkowskiPixelNorm, self).__init__()
        self.num_features = num_features
        self.dimension = dimension
        self.eps = eps

    def forward(self, input):
        features = input.F
        norm = torch.sum(torch.pow(features, 2), dim=1, keepdim=True)
        out = features / (norm + self.eps).sqrt()
        return ME.SparseTensor(
            out,
            coords_key=input.coords_key,
            coords_manager=input.coords_man)

    def __repr__(self):
        s = '({}, eps={}, dimension={})'.format(
            self.num_features, self.eps, self.dimension)
        return self.__class__.__name__ + s


class MinkowskiAdaIN(nn.Module):
    '''
    Adaptive Instance Normalization Layer
    Original Paper: https://arxiv.org/pdf/1703.06868.pdf

    Many parts of the code is borrowed from pytorch original
    BatchNorm implementation.

    INPUT:
        - input: ME.SparseTensor

    RETURNS:
        - out: ME.SparseTensor
    '''
    def __init__(self, in_channels, dimension=3, eps=1e-5):
        super(MinkowskiAdaIN, self).__init__()
        self.in_channels = in_channels
        self.dimension = dimension
        self.eps = eps
        self._weight = torch.ones(in_channels)
        self._bias = torch.zeros(in_channels)

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, weight):
        '''
        Set weight and bias parameters for AdaIN Layer.
        Note that in AdaptIS, the parameters to the AdaIN layer
        are trainable outputs from the controller network.
        '''
        if weight.shape[0] != self.in_channels:
            raise ValueError('Supplied weight vector feature dimension\
                does not match AdaIN layer definition!')
        self._weight = weight

    @property
    def bias(self):
        return self._bias

    @bias.setter
    def bias(self, bias):
        if bias.shape[0] != self.in_channels:
            raise ValueError('Supplied bias vector feature dimension\
                does not match AdaIN layer definition!')
        self._bias = bias

    def forward(self, x):
        '''
        INPUTS:
            - x (ME.SparseTensor)

        RETURNS:
            - out (ME.SparseTensor)
        '''
        f = x.F
        norm = (f - f.mean(dim=0)) / (f.var(dim=0) + self.eps).sqrt()
        out = self.weight * norm + self.bias
        return ME.SparseTensor(
            out,
            coords_key=input.coords_key,
            coords_manager=input.coords_man)


class MinkowskiGroupNorm(nn.Module):
    '''
    TODO
    '''
    pass
