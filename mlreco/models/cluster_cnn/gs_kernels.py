import torch
import torch.nn as nn

# class EdgeKernel(nn.module):

#     def __init__(self, in1_features, in2_features, out_features=1, **kwargs):
#         super(EdgeKernel, self).__init__()
#         self.in1_features = in1_features
#         self.in2_features = in2_features
#         self.out_features = out_features

#     def forward(self, x):
#         raise NotImplementedError

class BilinearKernel(nn.Module):

    def __init__(self, num_features, bias=False):
        super(BilinearKernel, self).__init__()
        self.m = nn.Bilinear(num_features, num_features, 1, bias=bias)

    def forward(self, x1, x2):
        return self.m(x1, x2)