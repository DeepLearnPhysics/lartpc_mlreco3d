import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sparseconvnet as scn

from mlreco.models.cluster_cnn.losses.lovasz import mean, lovasz_hinge_flat, StableBCELoss, iou_binary
from collections import defaultdict


def bc_distance(gauss1, gauss2, eps=1e-6, debug=False):
    '''
    Computes the Bhattacharya similarity measure for two spherical gaussians.
    '''
    g1 = gauss1.expand_as(gauss2)
    mu1, sigma1 = g1[:, :3], g1[:, 3:]
    mu2, sigma2 = gauss2[:, :3], gauss2[:, 3:]
    variance_term = (sigma1**2 + sigma2**2 + 2 * eps)

    beta = torch.pow(0.5 * ((sigma1 + eps) / (sigma2 + eps) + (sigma2 + eps) / (sigma1 + eps)), -1.5)
    dist = 0.25 * torch.pow(torch.norm(mu1 - mu2, dim=1), 2) / variance_term.squeeze()

    # if debug:
    #     print('variance_term = ', variance_term)
    #     print('beta = ', beta)
    #     print('dist = ', dist)

    return torch.clamp(beta.squeeze() * torch.exp(-dist), min=1e-6, max=1-1e-6)


class GNNGroupingLoss(nn.Module):

    def __init__(self, cfg, name='gnn_grouping_loss'):
        super(GNNGroupingLoss, self).__init__()
        self.loss_config = cfg[name]
        self.kernel = bc_distance
        self.bceloss = StableBCELoss()

    def forward(self, nodes, node_batch_labels, node_group_labels):

        loss, accuracy = [], []
        for bidx in node_batch_labels.unique():
            batch_mask = node_batch_labels == bidx
            groups = node_group_labels[batch_mask]
            nodes_batch = nodes[batch_mask]
            for g in groups.unique():
                group_mask = groups == g
                grouped_nodes = nodes_batch[group_mask]
                others = nodes_batch[~group_mask]
                intra_dist = 0
                # print('------------------------------')
                if grouped_nodes.shape[0] > 1:
                    gauss1 = grouped_nodes.mean(dim=0)
                else:
                    gauss1 = grouped_nodes
                p = bc_distance(gauss1, nodes_batch)
                kernel_loss = self.bceloss(p, group_mask.float())
                loss.append(kernel_loss)
                with torch.no_grad():
                    acc = iou_binary(p > 0.5, group_mask, per_image=False)
                    accuracy.append(float(acc))
        # print(loss)
        loss = sum(loss) / len(loss)
        accuracy = sum(accuracy) / len(accuracy)
        # print(loss, accuracy)

        return {'loss': loss, 'accuracy': accuracy}
