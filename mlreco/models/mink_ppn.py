import numpy as np
import torch
import torch.nn as nn
import time

import MinkowskiEngine as ME
import MinkowskiFunctional as MF

from mlreco.mink.layers.ppnplus import PPNTest
from mlreco.mink.chain.factories import *
from collections import defaultdict


class PPNLonely(nn.Module):

    def __init__(self, cfg, name='full_chain'):
        super(PPNLonely, self).__init__()
        self.net = PPNTest(cfg, name='ppn')


    def forward(self, input):
        device = input[0].device
        out = defaultdict(list)

        for igpu, x in enumerate(input):
            input_data = x[:, :5]
            res = self.net(input_data)
            out['ppn_scores'] += res['ppn_scores']
            out['ppn_logits'] += res['ppn_logits']
            out['points'] += res['points']
            
        return out


class PPNLonelyLoss(torch.nn.modules.loss._Loss):

    def __init__(self, cfg, name='ppn_loss'):
        super(PPNLonelyLoss, self).__init__()
        self.loss_config = cfg[name]
        self.lossfn = torch.nn.functional.binary_cross_entropy_with_logits
        self.resolution = self.loss_config.get('ppn_resolution', 5.0)


    def pairwise_distances(self, x, y, eps=1e-8):
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
        return torch.sqrt(dist + eps)


    def forward(self, outputs, segment_label, particles_label):
        '''
        segmentation[0], label and weight are lists of size #gpus = batch_size.
        segmentation has as many elements as UResNet returns.
        label[0] has shape (N, dim + batch_id + 1)
        where N is #pts across minibatch_size events.
        '''
        # TODO Add weighting
        ppn_scores = outputs['ppn_scores']
        assert len(ppn_scores) == len(segment_label)
        batch_ids = [d[:, 0] for d in segment_label]
        num_batches = len(batch_ids[0].unique())
        highE = [t[t[:, -1].long() != 4] for t in segment_label]
        total_loss = 0
        total_acc = 0
        count = 0
        device = segment_label[0].device

        loss, accuracy = 0, []
        res = {}
        # Semantic Segmentation Loss
        for igpu in range(len(segment_label)):
            particles = particles_label[igpu]
            ppn_logits = outputs['ppn_logits'][igpu]
            points = outputs['points'][igpu]
            loss_gpu, acc_gpu = 0.0, 0.0
            for layer in range(len(ppn_logits)):
                ppn_score_layer = ppn_logits[layer]
                points_layer = points[layer]
                loss_layer = 0.0
                for b in batch_ids[igpu].unique():
                    batch_index = batch_ids[igpu] == b
                    points_label = particles[particles[:, 0] == b][:, 1:4]
                    scores_event = ppn_score_layer[points_layer[:, 0] == b].squeeze()
                    points_event = points_layer[points_layer[:, 0] == b]
                    d = self.pairwise_distances(points_label, points_event[:, 1:4])
                    d_positives = (d < self.resolution * 2**(len(ppn_logits) - layer)).any(dim=0)
                    num_positives = d_positives.sum()
                    num_negatives = d_positives.nelement() - num_positives
                    w = num_positives.float() / (num_positives + num_negatives).float()
                    weight_ppn = torch.zeros(d_positives.shape[0]).to(device)
                    weight_ppn[d_positives] = 1 - w
                    weight_ppn[~d_positives] = w
                    loss_batch = self.lossfn(scores_event, d_positives.float(), weight=weight_ppn, reduction='mean')
                    loss_layer += loss_batch
                    if layer == len(ppn_logits)-1:
                        acc = (d_positives == (scores_event > 0)).sum().float() / float(scores_event.shape[0])
                        total_acc += acc
                loss_layer /= num_batches
                total_loss += loss_layer

        total_acc /= num_batches
        res['loss'] = total_loss
        res['accuracy'] = total_acc


        return res
