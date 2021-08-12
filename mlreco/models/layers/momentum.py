import torch
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
    def __init__(self, num_input, num_output=1, num_hidden=128, evidential=False):
        super(MomentumNet, self).__init__()
        self.linear1 = nn.Linear(num_input, num_hidden)
        self.norm1 = nn.BatchNorm1d(num_input)
        self.linear2 = nn.Linear(num_hidden, num_hidden)
        self.norm2 = nn.BatchNorm1d(num_hidden)
        self.linear3 = nn.Linear(num_hidden, num_output)

        self.elu = nn.LeakyReLU(negative_slope=0.33)
        if evidential:
            self.evidence = nn.Softplus()
        else:
            self.evidence = nn.Identity()

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
        out = self.evidence(x)
        return out


class EvidentialMomentumNet(nn.Module):

    def __init__(self, num_input, num_output=4, num_hidden=128, eps=0.0):
        super(EvidentialMomentumNet, self).__init__()
        self.linear1 = nn.Linear(num_input, num_hidden)
        self.norm1 = nn.BatchNorm1d(num_input)
        self.linear2 = nn.Linear(num_hidden, num_hidden)
        self.norm2 = nn.BatchNorm1d(num_hidden)
        self.linear3 = nn.Linear(num_hidden, num_output)

        self.elu = nn.LeakyReLU(negative_slope=0.33)

        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        self.eps = eps

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
        vab = self.softplus(x[:, :3]) + self.eps
        alpha = torch.clamp(vab[:, 1] + 1.0, min=1.0).view(-1, 1)
        gamma = 2.0 * self.sigmoid(x[:, 3]).view(-1, 1)
        out = torch.cat([gamma, vab[:, 0].view(-1, 1), 
                         alpha, vab[:, 2].view(-1, 1)], dim=1)

        evidence = torch.clamp(out, min=self.eps)
        return evidence