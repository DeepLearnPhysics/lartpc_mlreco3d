from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, LeakyReLU, Dropout, BatchNorm1d, Bilinear

# Edge prediction modules

# final prediction layer
class EdgeModel(torch.nn.Module):
    def __init__(self, node_in, edge_in, leak):
        """
        Basic model for making edge predictions
        
        parameters:
            node_in - number of node features coming in
            edge_in - number of edge features coming in
            leak - leakiness of leakyrelus
        """
        super(EdgeModel, self).__init__()

        self.edge_mlp = Seq(Lin(2*node_in + edge_in, 64),
                            LeakyReLU(leak),
                            Lin(64, 32),
                            LeakyReLU(leak),
                            Lin(32, 16),
                            LeakyReLU(leak),
                            Lin(16,8),
                            LeakyReLU(leak),
                            Lin(8,2)
                            )

    def forward(self, src, dest, edge_attr, u, batch):
        return self.edge_mlp(torch.cat([src, dest, edge_attr], dim=1))
 

class BilinEdgeModel(torch.nn.Module):
    def __init__(self, node_in, edge_in, leak):
        """
        Bilinear model for making edge predictions
        
        parameters:
            node_in - number of node features coming in
            edge_in - number of edge features coming in
            leak - leakiness of leakyrelus
        """
        super(BilinEdgeModel, self).__init__()
        
        self.bse = Bilinear(node_in, edge_in, 16, bias=True)
        self.bte = Bilinear(node_in, edge_in, 16, bias=True)
        self.bst = Bilinear(node_in, node_in, edge_in, bias=False)
        self.bee = Bilinear(edge_in, edge_in, 16, bias=True)
        
        self.mlp = Seq(
            Lin(3*16, 64),
            LeakyReLU(leak),
            Lin(64, 64),
            LeakyReLU(leak),
            Lin(64,32),
            LeakyReLU(leak),
            Lin(32,16),
            LeakyReLU(leak),
            Lin(16,2)
        )
        
    def forward(self, source, target, edge_attr, u, batch):
        # two bilinear forms
        x = self.bse(source, edge_attr)
        y = self.bte(target, edge_attr)
        
        # trilinear form
        z = self.bst(source, target)
        z = self.bee(z, edge_attr)
        
        out = torch.cat([x, y, z], dim=1)
        out = self.mlp(out)
        return out
    
