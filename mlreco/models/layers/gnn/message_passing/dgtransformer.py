import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepGraphTransformer(nn.Module):

    def __init__(self, cfg):
        super(DeepGraphTransformer, self).__init__()

        