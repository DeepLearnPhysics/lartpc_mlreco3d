from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch_scatter import scatter_mean
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, LeakyReLU, Dropout, BatchNorm1d, Bilinear

# Node prediction modules

# final prediction layer
class NodeModel(torch.nn.Module):
    def __init__(self, node_in, edge_in, leak):
        """
        Basic model for making node predictions
        
        parameters:
            node_in - number of node features coming in
            edge_in - number of edge features coming in
            leak - leakiness of leakyrelus
        """
        super(NodeModel, self).__init__()

        self.node_mlp_1 = Seq(Lin(node_in + edge_in, 64),
                              LeakyReLU(leak),
                              Lin(64, 32),
                              LeakyReLU(leak),
                              Lin(32, 16),
                              LeakyReLU(leak),
                              Lin(16,8),
                              LeakyReLU(leak),
                              Lin(8,2)
                              )

        self.node_mlp_2 = Seq(Lin(node_in + 2, 64),
                              LeakyReLU(leak),
                              Lin(64, 32),
                              LeakyReLU(leak),
                              Lin(32, 16),
                              LeakyReLU(leak),
                              Lin(16,8),
                              LeakyReLU(leak),
                              Lin(8,2)
                              )

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        print(edge_index, edge_index.shape)
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out], dim=1)
        return self.node_mlp_2(out)
 
