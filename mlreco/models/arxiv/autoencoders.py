import numpy as np
import torch
import torch.nn as nn

import MinkowskiEngine as ME
import MinkowskiFunctional as MF

from mlreco.models.layers.sparse_generator import SparseGenerator
from mlreco.models.layers.cnn_encoder import SparseResidualEncoder2
from collections import defaultdict


class VAEProGAN(nn.Module):

    def __init__(self, cfg, name='progan'):
        super(VAEProGAN, self).__init__()
        self.model_config = cfg[name]

    def forward(self, input):
        pass


class VAEStyleGAN(nn.Module):

    def __init__(self, cfg, name='stylegan'):
        super(VAEStyleGAN, self).__init__()
        self.model_config = cfg[name]

    def forward(self, input):
        pass


class VAEGAN(nn.Module):
    def __init__(self, cfg, name='progan'):
        super(VAEGAN, self).__init__()
        self.model_config = cfg[name]

    def forward(self, input):
        pass