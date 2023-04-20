import torch
import torch.nn as nn

import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiOps as me

from mlreco.models.layers.common.uresnet_layers import UResNetDecoder, UResNetEncoder
from mlreco.models.experimental.transformers.positional_encodings import FourierEmbeddings
from mlreco.models.experimental.cluster.pointnet2.pointnet2_utils import furthest_point_sample
from mlreco.models.experimental.transformers.positional_encodings import get_normalized_coordinates
from mlreco.models.experimental.transformers.transformer import TransformerDecoder, GenericMLP
from torch_geometric.nn import MLP


class TransformerSPICE(nn.Module):
    
    def __init__(self, cfg, name='transformer_spice'):
        super(TransformerSPICE, self).__init__()
        
        self.model_config = cfg[name]