import torch
import torch.nn as nn
import torch.nn.functional as F

from .lovasz import StableBCELoss

# Collection of Miscellaneous Loss Functions not yet implemented in Pytorch.

def multivariate_kernel(centroid, log_sigma, Lprime, eps=1e-8):
    def f(x):
        N = x.shape[0]
        L = torch.zeros(3, 3)
        tril_indices = torch.tril_indices(row=3, col=3, offset=-1)
        L[tril_indices[0], tril_indices[1]] = Lprime
        sigma = torch.exp(log_sigma) + eps
        L += torch.diag(sigma)
        cov = torch.matmul(L, L.T)
        dist = torch.matmul((x - centroid), torch.inverse(cov))
        dist = torch.bmm(dist.view(N, 1, -1), (x-centroid).view(N, -1, 1)).squeeze()
        probs = torch.exp(-dist)
        return probs
    return f


def squared_distances(v1, v2):
    v1_2 = v1.unsqueeze(1).expand(v1.size(0), v2.size(0), v1.size(1)).double()
    v2_2 = v2.unsqueeze(0).expand(v1.size(0), v2.size(0), v1.size(1)).double()
    return torch.pow(v2_2 - v1_2, 2).sum(2)


def bhattacharyya_distance_matrix(v1, v2, eps=1e-8):
    x1, s1 = v1[:, :3], v1[:, 3].view(-1)
    x2, s2 = v2[:, :3], v1[:, 3].view(-1)
    g1 = torch.ger(s1**2, 1.0 / (s2**2 + eps))
    g2 = g1.t()
    dist = squared_distances(x1.contiguous(), x2.contiguous())
    denom = 1.0 / (eps + s1.unsqueeze(1)**2 + s2**2)
    out = 0.25 * torch.log(0.25 * (g1 + g2 + 2)) + 0.25 * dist / denom
    return out


def bhattacharyya_coeff_matrix(v1, v2, eps=1e-6):
    x1, s1 = v1[:, :3], v1[:, 3].view(-1)
    x2, s2 = v2[:, :3], v1[:, 3].view(-1)
    g1 = torch.ger(s1**2, 1.0 / (s2**2 + eps))
    g2 = g1.t()
    dist = squared_distances(x1.contiguous(), x2.contiguous())
    denom = 1.0 / (eps + s1.unsqueeze(1)**2 + s2**2)
    out = 0.25 * torch.log(0.25 * (g1 + g2 + 2)) + 0.25 * dist / denom
    out = torch.exp(-out)
    return out


class FocalLoss(nn.Module):
    '''
    Original Paper: https://arxiv.org/abs/1708.02002
    Implementation: https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938
    '''
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.stable_bce = StableBCELoss()

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = self.stable_bce(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class WeightedFocalLoss(FocalLoss):

    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(WeightedFocalLoss, self).__init__(alpha=alpha, gamma=gamma, logits=logits, reduce=reduce)

    def forward(self, inputs, targets):
        with torch.no_grad():
            pos_weight = torch.sum(targets == 0) / (1.0 + torch.sum(targets == 1))
            weight = torch.ones(inputs.shape[0]).cuda()
            weight[targets == 1] = pos_weight
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = self.stable_bce(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        F_loss = torch.mul(F_loss, weight)

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
