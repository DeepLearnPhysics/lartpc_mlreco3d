import torch
import torch.nn as nn

# For MinkowskiEngine
import MinkowskiEngine as ME
from .factories import *

class MENetworkBase(nn.Module):
    '''
    Abstract Base Class for global network parameters.
    '''
    def __init__(self, cfg, name='network_base'):
        super(MENetworkBase, self).__init__()
        model_cfg = cfg[name]
        # Dimension of dataset
        self.D = model_cfg.get('D', 3)
        # Number of input data features
        self.num_input = model_cfg.get('num_input', 1)
        # Allow biases in convolutions and linear layers
        self.allow_bias = model_cfg.get('allow_bias', True)
        # Spatial size of dataset
        self.spatial_size = model_cfg.get('spatial_size', 512)

        # Define activation function
        self.leakiness = model_cfg.get('leakiness', 0.33)
        self.activation_cfg = model_cfg.get('activation', {})
        self.activation_name = self.activation_cfg.get('name', 'lrelu')
        self.activation_args = self.activation_cfg.get(
            'args', {})
        print('Using activation: {}'.format(self.activation_name))

        # Define normalization function
        print(model_cfg)
        self.norm_cfg = model_cfg.get('norm_layer', {})
        self.norm = self.norm_cfg.get('name', 'batch_norm')
        self.norm_args = self.norm_cfg.get('args', {})
        print('Using normalization layer mode: {}'.format(self.norm))


    def forward(self, input):
        raise NotImplementedError
