import torch
import numpy as np

from mlreco.models.layers.base import NetworkBase


class CPN(NetworkBase):
    '''
    Centroid Proposal Network

    This module takes intermediate feature planes from a backbone decoder
    and predicts the likelihood of a point being a cluster centroid in
    embedding space.

    The network is inspired from PPN and the seediness construction of 
    SpatialEmbeddings. 

    Configuration
    -------------
    num_strides : int
        Depth of UResNet, also corresponds to how many times we down/upsample.
    filters : int
        Number of filters in the first convolution of UResNet.
        Will increase linearly with depth.
    num_classes : int
        Should be number of classes (+1 if we include ghost points directly)
    data_dim : int
        Dimension 2 or 3
    use_encoding: bool, optional
        Whether to use feature maps from the encoding or decoding path of UResNet.
    downsample_ghost: bool, optional
        Whether to apply the downsampled ghost mask.
    ppn_num_conv: int, optional
        How many convolutions to apply at each level of PPN.
    ppn1_size: int, optional
        Size in px of the coarsest feature map used by PPN1. Should divide the original spatial size.
    ppn2_size: int, optional
        Size in px of the intermediate feature map used by PPN2. Should divide the original spatial size.
    spatial_size: int, optional
        Size in px of the original image.
    '''
    def __init__(self, cfg, name='cpn'):
        super(CPN, self).__init__(cfg)

        self.model_config = cfg['cpn']
        self.num_strides = self.model_config.get('num_strides', 5)
        self.num_filters = self.model_config.get('filters', 16)
        self.num_classes = self.model_config.get('num_classes', 5)

        self.nPlanes = [i * self.num_strides for i in range(1, self.num_strides+1)]

        self.cpn_conv = scn.Sequential()
