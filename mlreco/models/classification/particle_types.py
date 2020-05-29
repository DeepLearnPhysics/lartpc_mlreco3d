import torch
import torch.nn as nn
import numpy as np
import sparseconvnet as scn
from collections import defaultdict

from mlreco.models.layers.uresnet import UResNet, UResNetEncoder
from mlreco.models.ppn import PPN, PPNLoss
from mlreco.models.layers.stacknet import StackUNet
from .utils import add_normalized_coordinates, distance_matrix


class ParticleTypeEncoder(nn.Module):

    def __init__(self, cfg, name='particle_type_encoder'):
        super(ParticleTypeEncoder, self).__init__()
        self.model_config = cfg[name]

        self.mode = self.model_config.get('mode', 'flatten')

        self.encoder = UResNetEncoder(cfg)