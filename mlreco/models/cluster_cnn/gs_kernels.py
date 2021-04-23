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



class BilinearNNKernel(nn.Module):

    def __init__(self, num_features, bias=False):
        super(BilinearNNKernel, self).__init__()
        
        self.m = nn.Linear(64, 1, bias=bias)

        self.nn1 = nn.Sequential(
            nn.Linear(num_features, 32, bias=bias),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Linear(32, 32, bias=bias),
            nn.BatchNorm1d(32),
            nn.ELU(),
        )

        self.nn2 = nn.Sequential(
            nn.Linear(num_features, 32, bias=bias),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Linear(32, 32, bias=bias),
            nn.BatchNorm1d(32),
            nn.ELU(),
        )

    def forward(self, x1, x2):

        f1 = self.nn1(x1)
        f2 = self.nn2(x2)

        out = torch.cat([f1, f2], dim=1)
        
        return self.m(out)