# Pool sparse tensor by clusters
from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np


class ClusterPool(nn.Module):
    """
    PyTorch Layer that pools features by cluster assignment
    options:
        pooltype - how to pool
            'max' (default) - take max of features
            'sum' - sum features
            'average' - average features
            'pnorm' - p-norm of features
        p - norm to use for 'pnorm' option (default 2)
    forward method:
        inputs:
            features tensor (e.g. from scn sparse tensor)
            list of clusters
        output:
            pytorch tensor of size # clusters x # features
            
    """
    def __init__(self, pooltype='max', p=2):
        super(ClusterPool, self).__init__()
        self.pooltype = pooltype
        self.p = p
        
    def forward(self, features, cs):
        # TODO - handle batches in SCN tensors
        
        pools = []
        for c in cs:
            # step 1 - find coordinates indices
            inds = c.inds
            # step 2 - use pooling over coords
            # TODO - add softmax function
            if self.pooltype == 'max':
                pools.append(torch.max(features[inds], 0)[0])
            elif self.pooltype == 'sum':
                pools.append(torch.sum(features[inds], 0))
            elif self.pooltype == 'average':
                pools.append(torch.mean(features[inds], 0))
            elif self.pooltype == 'pnorm':
                pools.append(torch.norm(features[inds], p=self.p, dim=0))
            else:
                print("bad pooltype!")
                return None
        # return pools stacked as a torch tensor
        return torch.stack(pools)