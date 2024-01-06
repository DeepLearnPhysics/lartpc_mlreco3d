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
    def __init__(self, num_input, num_output=1, num_hidden=128, positive_outputs=False):
        super(MomentumNet, self).__init__()
        self.linear1 = nn.Linear(num_input, num_hidden)
        self.norm1 = nn.BatchNorm1d(num_input)
        self.linear2 = nn.Linear(num_hidden, num_hidden)
        self.norm2 = nn.BatchNorm1d(num_hidden)
        self.linear3 = nn.Linear(num_hidden, num_output)

        self.lrelu = nn.LeakyReLU(negative_slope=0.33)
        if positive_outputs:
            self.final = nn.Softplus()
        else:
            self.final = nn.Identity()

    def forward(self, x):
        if x.shape[0] > 1:
            x = self.norm1(x)
        x = self.linear1(x)
        x = self.lrelu(x)
        if x.shape[0] > 1:
            x = self.norm2(x)
        x = self.linear2(x)
        x = self.lrelu(x)
        x = self.linear3(x)
        out = self.final(x)
        return out


class VertexNet(MomentumNet):
    '''
    Small MLP for handling vertex regression and particle primary prediction.
    '''
    def __init__(self, num_input, num_output=1, num_hidden=128, positive_outputs=False, batch_norm=False):
        super(VertexNet, self).__init__(num_input, num_output, num_hidden, positive_outputs)
        self.num_output = num_output
        self.batch_norm = batch_norm

    def forward(self, x):
        if self.batch_norm and x.shape[0] > 1:
            x = self.norm1(x)
        x = self.linear1(x)
        x = self.lrelu(x)
        if self.batch_norm and x.shape[0] > 1:
            x = self.norm2(x)
        x = self.linear2(x)
        x = self.lrelu(x)
        x = self.linear3(x)
        if self.num_output == 5:
            vtx_pred = self.final(x[:, :3])
            out = torch.cat([vtx_pred, x[:, 3:]], dim=1)
            return out
        else:
            return x


class DeepVertexNet(nn.Module):
    '''
    Small MLP for extracting input edge features from two node features.

    USAGE:
        net = EdgeFeatureNet(16, 16)
        node_x = torch.randn(16, 5)
        node_y = torch.randn(16, 5)
        edge_feature_x2y = net(node_x, node_y) # (16, 5)
    '''
    def __init__(self, num_input, num_output=1, num_hidden=512, num_layers=5, positive_outputs=False):
        super(DeepVertexNet, self).__init__()
        self.num_output = num_output
        self.linear = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.num_layers = num_layers

        for i in range(num_layers):
            self.norm.append(nn.BatchNorm1d(num_input))
            self.linear.append(nn.Linear(num_input, num_hidden))
            num_input = num_hidden

        self.final = nn.Linear(num_hidden, num_output)

        self.lrelu = nn.LeakyReLU(negative_slope=0.33)
        if positive_outputs:
            self.final = nn.Softplus()
        else:
            self.final = nn.Identity()

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.norm[i](x)
            x = self.lrelu(x)
            x = self.linear[i](x)
        if self.num_output == 5:
            vtx_pred = self.final(x[:, :3])
            out = torch.cat([vtx_pred, x[:, 3:]], dim=1)
            return out
        else:
            return x


class EvidentialMomentumNet(nn.Module):

    def __init__(self, num_input, num_output=4,
                 num_hidden=128, eps=0.0, logspace=False):
        super(EvidentialMomentumNet, self).__init__()
        self.linear1 = nn.Linear(num_input, num_hidden)
        self.norm1 = nn.BatchNorm1d(num_input)
        self.linear2 = nn.Linear(num_hidden, num_hidden)
        self.norm2 = nn.BatchNorm1d(num_hidden)
        self.linear3 = nn.Linear(num_hidden, num_output)

        self.elu = nn.LeakyReLU(negative_slope=0.33)

        self.softplus = nn.Softplus()
        self.logspace = logspace
        if logspace:
            self.gamma = nn.Identity()
        else:
            self.gamma = nn.Sigmoid()
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
        gamma = 2.0 * self.gamma(x[:, 3]).view(-1, 1)
        out = torch.cat([gamma, vab[:, 0].view(-1, 1),
                         alpha, vab[:, 2].view(-1, 1)], dim=1)
        if not self.logspace:
            evidence = torch.clamp(out, min=self.eps)
        else:
            evidence = out
        return evidence
