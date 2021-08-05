import torch
import torch.nn as nn
import torch.nn.functional as F

# For MinkowskiEngine
import MinkowskiEngine as ME
from MinkowskiNonlinearity import MinkowskiNonlinearityBase

# Custom Nonlinearities
class MinkowskiLeakyReLU(MinkowskiNonlinearityBase):
    MODULE = nn.LeakyReLU

class MinkowskiELU(MinkowskiNonlinearityBase):
    MODULE = nn.ELU
class MinkowskiMish(nn.Module):
    '''
    Mish Nonlinearity: https://arxiv.org/pdf/1908.08681.pdf
    '''
    def __init__(self):
        super(MinkowskiMish, self).__init__()

    def forward(self, input):
        out = F.softplus(input.F)
        out = torch.tanh(out)
        out = out * input.F
        return ME.SparseTensor(
            out,
            coords_key=input.coords_key,
            coords_manager=input.coords_man)

    def __repr__(self):
        return self.__class__.__name__ + '()'
