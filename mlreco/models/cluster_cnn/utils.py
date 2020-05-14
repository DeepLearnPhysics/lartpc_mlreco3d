import torch
import torch.nn as nn
import numpy as np
import sparseconvnet as scn


def add_normalized_coordinates(input):
    '''
    Utility Method for attaching normalized coordinates to
    sparse tensor features.

    INPUTS:
        - input (scn.SparseConvNetTensor): sparse tensor to
        attach normalized coordinates with range (-1, 1)

    RETURNS:
        - output (scn.SparseConvNetTensor): sparse tensor with 
        normalized coordinate concatenated to first three dimensions.
    '''
    output = scn.SparseConvNetTensor()
    with torch.no_grad():
        coords = input.get_spatial_locations().float()
        normalized_coords = (coords[:, :3] - input.spatial_size.float() / 2) \
            / (input.spatial_size.float() / 2)
        if torch.cuda.is_available():
            normalized_coords = normalized_coords.cuda()
        output.features = torch.cat([normalized_coords, input.features], dim=1)
    output.metadata = input.metadata
    output.spatial_size = input.spatial_size
    return output


def distance_matrix(points):
    """
    Uses BLAS/LAPACK operations to efficiently compute pairwise distances.
    
    INPUTS:
        - points (N x d Tensor): torch.Tensor with each row 
        corresponding to a point in vector space.

    RETURNS:
        - inner_prod (N x N Tensor): computed pairwise distance
        matrix squared.
    """
    M = points[None,...]
    zeros = torch.zeros(1, 1, 1)
    if torch.cuda.is_available():
        zeros = zeros.cuda()
    inner_prod = torch.baddbmm(zeros, M, M.permute([0, 2, 1]), alpha=-2.0, beta=0.0)
    squared    = torch.sum(torch.mul(M, M), dim=-1, keepdim=True)
    inner_prod += squared
    inner_prod += squared.permute([0, 2, 1])
    return inner_prod.squeeze(0)


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    y_norm = (y**2).sum(1).view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return dist