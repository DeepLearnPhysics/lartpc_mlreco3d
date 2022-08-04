import torch
import torch.nn as nn
import numpy as np
import sparseconvnet as scn
from collections import defaultdict

class GatedAveragePool(nn.Module):

    def __init__(self, num_features, eps=1e-6):
        super(GatedAveragePool, self).__init__()
        self.eps = 1e-6

    def forward(self, x, weights):
        w = weights + self.eps
        x = weights * x
        norm = torch.sum(weights)
        x = torch.mean(x, dim=0)
        return x
