import numpy as np
import torch
import torch.nn as nn


def digamma_evd_loss(alpha, y):
    '''
    Bayes risk loss for Dirichlet prior Evidential Learning

    INPUTS:
        - alpha (FloatTensor): N x C concentration parameters, 
        where C is the number of class labels.
        - y (FloatTensor): N x C one-hot encoded class labels

    RETURNS:
        - loss (FloatTensor): N x 1 non-reduced loss for each example. 
    '''
    S = alpha.sum(dim=1, keepdim=True)
    loss = torch.sum((torch.digamma(S) - torch.digamma(alpha)) * y, dim=1)
    return loss


def sumsq_evd_loss(alpha, y):
    '''
    Negative log loss for Dirichlet prior evidential learning.

    INPUTS:
        - alpha (FloatTensor): N x C concentration parameters, 
        where C is the number of class labels.
        - y (FloatTensor): N x C one-hot encoded class labels

    RETURNS:
        - loss (FloatTensor): N x 1 non-reduced loss for each example. 
    '''
    S = alpha.sum(dim=1, keepdim=True)
    prediction_err = (y - alpha / S)**2
    variance = alpha * (S - alpha) / (S * S * (S + 1.0))
    loss = torch.sum(prediction_err + variance, dim=1)
    return loss


def nll_evd_loss(alpha, y):
    '''
    Negative log loss for Dirichlet prior evidential learning.

    INPUTS:
        - alpha (FloatTensor): N x C concentration parameters, 
        where C is the number of class labels.
        - y (FloatTensor): N x C one-hot encoded class labels

    RETURNS:
        - loss (FloatTensor): N x 1 non-reduced loss for each example. 
    '''
    S = alpha.sum(dim=1, keepdim=True)
    loss = torch.sum(y * (torch.log(S) - torch.log(alpha)), dim=1)
    return loss


def evd_kl_divergence(alpha, beta=None):
    '''
    KL Divergence between Dir(p|alpha) and Dir(p|beta), where
    alpha and beta are Dirichlet concentration parameters. 

    INPUTS:
        - alpha (FloatTensor): N x C concentration parameters
        - beta (FloatTensor): N x C concentration parameters. In case of 
        truth labels, this is a one-hot encoded class label tensor. 
        If None, this will compute the KL Divergence between Dir(p|alpha)
        and Dir(p|1), which is the uniform distribution over C classes. 

    RETURNS:
        - loss (FloatTensor): N x 1 non-reduced kl divergence loss. 
    '''
    device = alpha.device
    S_alpha = torch.sum(alpha, dim=1)
    if beta is None:
        beta = torch.ones([1, alpha.shape[1]], device=device)
    S_beta = torch.sum(beta, dim=1)
    loss = torch.lgamma(S_alpha) - torch.lgamma(S_beta)
    loss -= torch.sum(torch.lgamma(alpha), dim=1)
    A = (alpha - beta) * (torch.digamma(alpha) - torch.digamma(S_alpha.view(-1, 1)))
    loss += torch.sum(A, dim=1)
    return loss


def evd_loss_dict():
    loss_fn = {
        'edl_digamma': digamma_evd_loss,
        'edl_sumsq': sumsq_evd_loss,
        'edl_nll': nll_evd_loss
    }
    return loss_fn


def evd_loss_construct(name):
    losses = evd_loss_dict()
    if name not in losses:
        raise Exception("Unknown evd loss algorithm name provided: %s" % name)
    return losses[name]


class EVDLoss(nn.Module):
    '''
    Base class for loss used in the paper:
    Sensoy et. al., Evidential Deep Learning to Quantify 
    Classification Uncertainty
    '''
    def __init__(self, evd_loss_name, reduction='none', T=50000, 
                 one_hot=True, num_classes=5, mode='concentration'):
        super(EVDLoss, self).__init__()
        self.T = T  # Total epoch counts for which to anneal kld component. 
        self.evd_loss = evd_loss_construct(evd_loss_name)
        print("EVD LOSS NAME = ", evd_loss_name)
        self.kld_loss = evd_kl_divergence
        self.reduction = reduction
        self.one_hot = one_hot
        self.num_classes = num_classes
        self.mode = mode

    def forward(self, alpha, labels, t=0):

        device = alpha.device
        if self.one_hot:
            eye = torch.eye(self.num_classes).to(device=device)
            y = eye[labels.long()]
        else:
            y = labels

        if self.mode != 'concentration':
            evidence = alpha
            alpha = evidence + 1.0

        annealing = min(1.0, float(t) / self.T)

        evd_loss = self.evd_loss(alpha, y)
        alpha_tilde = y + (1 - y) * alpha
        kld_loss = self.kld_loss(alpha_tilde)

        loss = evd_loss + annealing * kld_loss
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise Exception("Unknown reduction method %s provided" % self.reduction)


def nll_regression_loss(logits, targets, eps=1e-6):
    '''
    Negative log loss for Dirichlet prior evidential learning.

    INPUTS:
        - alpha (FloatTensor): N x C concentration parameters, 
        where C is the number of class labels.
        - y (FloatTensor): N x 1 regression targets

    RETURNS:
        - loss (FloatTensor): N x 1 non-reduced loss for each example. 
    '''
    logits = logits.view(-1, 4)
    gamma, nu, alpha, beta = torch.split(logits, 4, dim=1)
    omega = 2.0 * beta * (1.0 + nu)
    nll = 0.5 * (np.log(np.pi) - torch.log(nu + 1e-5))  \
        - alpha * torch.log(omega)  \
        + (alpha+0.5) * torch.log(nu * (targets - gamma)**2 + omega)  \
        + torch.lgamma(alpha)  \
        - torch.lgamma(alpha+0.5)
    return torch.clamp(nll, min=0)

def kld_regression_loss(logits, targets, eps=1e-6):
    logits = logits.view(-1, 4)
    gamma, nu, alpha = logits[:, 0], logits[:, 1], logits[:, 2]
    loss = torch.abs(targets - gamma + eps) * (2.0 * nu + alpha)
    return loss

def kld_evd_l2_loss(logits, targets, eps=1e-6):
    logits = logits.view(-1, 4)
    gamma, nu, alpha = logits[:, 0], logits[:, 1], logits[:, 2]
    loss = torch.pow(targets - gamma + eps, 2) * (2.0 * nu + alpha)
    return loss

def kl_nig(logits, targets, eps=0.01):

    logits = logits.view(-1, 4)
    gamma, nu, alpha = logits[:, 0], logits[:, 1], logits[:, 2]

    error = torch.abs(targets - gamma + 1e-6)

    kl = 0.5 * (1.0 + eps + 0.001) / (nu+0.001) \
        - 0.5 - torch.lgamma(alpha / (1.0 + eps)) \
        + (alpha - (1.0 + eps)) * torch.digamma(alpha)

    loss = kl * error
    return loss

class EDLRegressionLoss(nn.Module):

    def __init__(self, reduction='none', w=0.0, kl_mode='evd',
                 one_hot='True', mode='concentration', eps=1e-6, T=50000):
        super(EDLRegressionLoss, self).__init__()
        self.reduction = reduction
        self.one_hot = one_hot
        self.mode = mode
        self.eps = eps
        self.nll_loss = nll_regression_loss
        self.kl_mode = kl_mode
        if self.kl_mode == 'evd':
            self.kld_loss = kld_regression_loss
        elif self.kl_mode == 'kl':
            self.kld_loss = kl_nig
        elif self.kl_mode == 'evd_l2':
            self.kld_loss = kld_evd_l2_loss
        else:
            raise ValueError('Unrecognized KL Divergence Error Loss')
        self.w = w
        self.T = T

    def forward(self, logits, targets, iteration=None):

        if iteration is not None:
            annealing = min(1.0, float(iteration) / self.T)
        else:
            annealing = self.w

        nll_loss = self.nll_loss(logits, targets, eps=self.eps)
        kld_loss = self.kld_loss(logits, targets, eps=self.eps)

        return nll_loss + annealing * kld_loss, nll_loss
