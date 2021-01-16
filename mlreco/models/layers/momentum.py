import torch.nn as nn

class MomentumNet(nn.Module):
    '''
    Small MLP for extracting input edge features from two node features.

    USAGE:
        net = EdgeFeatureNet(16, 16)
        node_x = torch.randn(16, 5)
        node_y = torch.randn(16, 5)
        edge_feature_x2y = net(node_x, node_y) # (16, 5)
    '''
    def __init__(self, num_input, num_output=1, num_hidden=128):
        super(MomentumNet, self).__init__()
        self.linear1 = nn.Linear(num_input, num_hidden)
        self.norm1 = nn.BatchNorm1d(num_hidden)
        self.linear2 = nn.Linear(num_hidden, num_hidden)
        self.norm2 = nn.BatchNorm1d(num_hidden)
        self.linear3 = nn.Linear(num_hidden, num_output)

        self.elu = nn.LeakyReLU(negative_slope=0.33)
        self.softplus = nn.Softplus()

    def forward(self, x):
        if x.shape[0] > 1:
            self.norm1(x)
        x = self.linear1(x)
        x = self.elu(x)
        if x.shape[0] > 1:
            x = self.norm2(x)
        x = self.linear2(x)
        x = self.elu(x)
        x = self.linear3(x)
        out = self.softplus(x)
        return out
