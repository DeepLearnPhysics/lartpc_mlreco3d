import numpy as np
import torch
import torch.nn as nn
import time

# MinkowskiEngine Backend
import MinkowskiEngine as ME
import MinkowskiFunctional as MF
from mlreco.models.mink.layers.network_base import MENetworkBase
from mlreco.models.mink.layers.ppn import PPN

from pprint import pprint


class ChainBase(MENetworkBase):
    '''
    Abstract Base Class for chain models. 
    '''
    def __init__(self, cfg, name='chain_base'):
        super(Chain1, self).__init__()

        # CNN Tasks
        self.segnet = None
        self.clusternet = None
        self.ppn = None

        # GNN Tasks
        self.shower_gnn = None
        self.interaction_gnn = None
        self.kinematics_gnn = None
        

    def fit_predict(self, input):
        '''
        Given input data, run full reconstruction using saved weights.

        INPUTS:
            - input: (N x F) Tensor in ME convention (batch index = 0)

        RETURNS:

            NOTE: The following conventions hold:

                1. N: Total Pixel Count
                2. N_f: Total fragment count
                3. N_g: Total particle (group) count

            - res: (dict) result dictionary, with the following 
            (key, val) pairs:

            coords: (N, 4) Tensor with batch index (0) and ghost-removed
                coordinates (1-3)

            semantics: (N, ) Tensor with predicted semantic labels

            fragment_labels: (N, ) Tensor with predicted fragment labels

            points: (N_p, ) Tensor with coordinates of PPN point predictions

            group_labels: (N, ) Tensor with predicted group labels

            group_index: (N_g, ) Tensor with group and batch ids. 

            particles_label: (N_g, ) Tensor with particle type labels

            particles_momentum: (N_g, ) Tensor with particle momentum predictions

            interaction_label: (N_g, ) Tensor with particle interaction labels
        '''

        raise NotImplementedError


    def forward(self, input):

        raise NotImplementedError



class Chain1(ChainBase):
    '''
    ME full chain with no parameter sharing
    '''
    def __init__(self, cfg, name='chain1'):
        super(Chain1, self).__init__(cfg)
        pass


class Chain2(ChainBase):
    '''
    ME full chain with:
        1. Shared encoder + three decoder (segment, embedding, ppn) CNN
        2. Separate node/edge encoders for each gnn task
    '''
    def __init__(self, cfg, name='chain2'):
        super(Chain2, self).__init__(cfg)
        pass


class Chain3(ChainBase):
    '''
    ME full chain with:
        1. Shared encoder + three decoder (segment, embedding, ppn) CNN
        2. Node/Edge encoder for fragments and groups. 
    '''
    def __init__(self, cfg, name='chain3'):
        super(Chain3, self).__init__(cfg)
        pass


class Chain4(ChainBase):
    '''
    ME full chain with:
        1. Shared encoder + three decoder (segment, embedding, ppn) CNN
        2. Node/Edge encoder for fragments and groups. 
    '''
    def __init__(self, cfg, name='chain4'):
        super(Chain4, self).__init__(cfg)
        pass


class Chain5(ChainBase):
    '''
    ME full chain with:
        1. Shared encoder + three decoder (segment, embedding, ppn) CNN
        2. Node/Edge encoder for fragments and groups. 
    '''
    def __init__(self, cfg, name='chain4'):
        super(Chain4, self).__init__(cfg)
        pass