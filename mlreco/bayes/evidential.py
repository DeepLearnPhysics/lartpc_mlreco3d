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
    loss = torch.sum((torch.digamma(S) - torch.digamma(alpha) * y), dim=1)
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


class EVDLoss(nn.Module):
    '''
    Base class for loss used in the paper:
    Sensoy et. al., Evidential Deep Learning to Quantify 
    Classification Uncertainty
    '''
    def __init__(self, reduction='none', T=10):
        super(EVDLoss, self).__init__()
        self.T = T  # Total epoch counts for which to anneal kld component. 
        
    @property
    def prediction_loss(self, alpha, y):
        raise NotImplementedError

    @property
    def kl_divergence_loss(self, alpha, y):
        raise NotImplementedError

    def forward(self, alpha, y, t=0):

        annealing = min(1.0, t / float(self.T))

        loss = self.prediction_loss(alpha, y) + \
               annealing * self.kl_divergence_loss(alpha, y)
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise Exception("Unknown reduction method %s provided" % self.reduction)