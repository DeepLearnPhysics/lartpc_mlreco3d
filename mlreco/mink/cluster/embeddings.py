import numpy as np
import torch
import torch.nn as nn

import MinkowskiEngine as ME
import MinkowskiFunctional as MF

from collections import defaultdict
from mlreco.mink.layers.factories import activations_dict, activations_construct
from mlreco.mink.layers.network_base import MENetworkBase
from mlreco.mink.layers.blocks import ResNetBlock, ConvolutionBlock


class ClusterResNet(nn.Module):

    pass


class ClusterSENet(nn.Module):

    pass


class ClusterResNeXt(nn.Module):

    pass