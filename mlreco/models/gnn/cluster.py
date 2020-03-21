import torch
import torch.nn as nn
import numpy as np
import sparseconvnet as scn
from collections import defaultdict


# KERNEL FUNCTIONS FOR CLUSTERING

def gauss(centroid, sigma):
    '''
    Constructor for a gaussian kernel functions.

    INPUTS:
        - centroid: (D, ) Tensor for the coordinates of the gaussian centroid.
        - sigma: value for gaussian bandwidth.

    RETURNS:
        - f (function): kernel function defined by centroid and sigma.
    '''
    def f(x):
        dists = torch.sum(torch.pow(x - centroid, 2), dim=1)
        probs = torch.exp(-dists / (2.0 * sigma**2))
        return probs
    return f

def mvgauss(centroid, L, dim=3):
    '''
    Constructor for multivariate gaussian kernels.

    L (torch.Tensor): D x D tensor representing Cholesky decomposition of
    the covariance matrix. The covariance matrix is then calculated as:

    \Sigma = LL^T.
    '''
    def f(x):
        N = x.shape[0]
        cov = torch.zeros(dim, dim)
        tril_indices = torch.tril_indices(row=dim, col=dim, offset=0)
        cov[tril_indices[0], tril_indices[1]] = L
        cov = torch.matmul(cov, cov.T)
        dist = torch.matmul((x - centroid), cov)
        dist = torch.bmm(dist.view(N, 1, -1), (x-centroid).view(N, -1, 1)).squeeze()
        probs = torch.exp(-dist)
        return probs
    return f

def laplace(centroid, sigma):
    def f(x):
        dists = torch.sum(torch.norm(x - centroid), dim=1)
        probs = torch.exp(-dist / sigma)
        return probs
    return f

def student_t(centroid):
    '''
    Pairwise student t distribution as used in TSNE
    '''
    def f(x):
        dists = torch.sum(torch.pow(x - centroid, 2), dim=1)
        probs = 1 / (1 + dists)
        return probs
    return f
