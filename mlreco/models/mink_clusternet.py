import numpy as np
import torch
import torch.nn as nn
import time

import MinkowskiEngine as ME
import MinkowskiFunctional as MF

from mlreco.mink.layers.ppnplus import PPNTest
from mlreco.mink.cluster.embeddings import ClusterResNet
from collections import defaultdict
from torch_scatter import scatter_mean

from mlreco.mink.cluster.losses import EmbeddingLoss

from pprint import pprint

class ClusterNetPP(nn.Module):

    def __init__(self, cfg, name='full_chain'):
        super(ClusterNetPP, self).__init__()
        self.net = ClusterResNet(cfg, name='cluster_resnet')


    def forward(self, input):
        device = input[0].device
        out = defaultdict(list)

        for igpu, x in enumerate(input):
            input_data = x[:, :5]
            res = self.net(input_data)
            # out['embedding_dec'] += res['embedding_dec']
            # out['ppn_dec'] += res['ppn_dec']
            out['ppn_scores'] += res['ppn_scores']
            out['ppn_logits'] += res['ppn_logits']
            out['points'] += res['points']
            out['point_predictions'] += res['point_predictions']
            out['embeddings'] += res['embeddings']
            out['margins'] += res['margins']
            
        return out


class ClusterChainLoss(torch.nn.modules.loss._Loss):
    
    def __init__(self, cfg, name='cluster_chain_loss'):
        super(ClusterChainLoss, self).__init__()
        self.ppn_loss = PPNSeedLoss(cfg)
        self.embedding_loss = EmbeddingLoss(cfg)

    def forward(self, outputs, segment_label, particles_label):

        res_ppn = self.ppn_loss(outputs, segment_label, particles_label)
        group_label = [t[:, [0, 1, 2, 3, 5]] for t in segment_label]
        semantic_label = [t[:, [0, 1, 2, 3, -1]] for t in segment_label]
        res_embedding = self.embedding_loss(outputs, semantic_label, group_label)
        res = {}

        res['loss'] = res_ppn['loss'] + res_embedding['loss']
        res['accuracy'] = (res_ppn['accuracy'] + res_embedding['accuracy']) / 2.0
        res['attention_loss'] = res_ppn['attention_loss']
        res['category_loss'] = res_ppn['category_loss']
        res['mask_loss'] = res_embedding['mask_loss']
        res['cluster_accuracy'] = res_embedding['accuracy']
        res['ppn_accuracy'] = res_ppn['accuracy']

        pprint(res)

        return res


class PPNSeedLoss(torch.nn.modules.loss._Loss):

    def __init__(self, cfg, name='ppn_loss'):
        super(PPNSeedLoss, self).__init__()
        self.loss_config = cfg[name]
        self.lossfn = torch.nn.functional.binary_cross_entropy_with_logits
        self.resolution = self.loss_config.get('ppn_resolution', 5.0)
        self.point_lossfn = torch.nn.functional.cross_entropy


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
        attention_loss = 0
        category_loss = 0
        total_acc = 0
        count = 0
        device = segment_label[0].device

        loss, accuracy = 0, []
        res = {}
        # Semantic Segmentation Loss
        for igpu in range(len(segment_label)):
            voxels = segment_label[igpu]
            particles = particles_label[igpu]
            ppn_logits = outputs['ppn_logits'][igpu]
            points = outputs['points'][igpu]
            point_predictions = outputs['point_predictions'][igpu]
            loss_gpu, acc_gpu = 0.0, 0.0
            # PPN Layer Loss Attention
            for layer in range(len(ppn_logits)):
                ppn_score_layer = ppn_logits[layer]
                points_layer = points[layer]
                loss_layer = 0.0
                for b in batch_ids[igpu].unique():
                    batch_index = batch_ids[igpu] == b
                    voxel_batch = voxels[batch_index]
                    index = voxel_batch[voxel_batch[:, 5] != -1][:, 5].to(dtype=torch.int64)
                    bincount = torch.bincount(index)
                    zero_bins = bincount > 0
                    bincount[bincount == 0] = 1.0
                    numerator = torch.zeros(bincount.shape[0], \
                        voxel_batch[voxel_batch[:, 5] != -1].shape[1]).to(device)
                    numerator = numerator.index_add(0, index, voxel_batch[voxel_batch[:, 5] != -1])
                    centroids = numerator / bincount.view(-1, 1)
                    centroids = centroids[:, 1:4][zero_bins]
                    points_label = particles[particles[:, 0] == b]
                    combined_label = torch.cat([centroids, points_label[:, 1:4]], dim=0)
                    scores_event = ppn_score_layer[points_layer[:, 0] == b].squeeze()
                    points_event = points_layer[points_layer[:, 0] == b]
                    d = self.pairwise_distances(combined_label, points_event[:, 1:4])
                    d_positives = (d < self.resolution * 2**(len(ppn_logits) - layer-1)).any(dim=0)
                    num_positives = d_positives.sum()
                    num_negatives = d_positives.nelement() - num_positives
                    w = num_positives.float() / (num_positives + num_negatives).float()
                    weight_ppn = torch.zeros(d_positives.shape[0]).to(device)
                    weight_ppn[d_positives] = 1 - w
                    weight_ppn[~d_positives] = w
                    loss_batch = self.lossfn(scores_event, d_positives.float(), weight=weight_ppn, reduction='mean')
                    loss_layer += loss_batch
                    # For last layer, compute labels for ppn point category prediction (PPN Labels vs Cluster Centroids)
                    if layer == len(ppn_logits)-1:
                        d_centroids = d[0:len(centroids), :]
                        d_cent_pos = (d_centroids < self.resolution * 2**(len(ppn_logits) - layer-1)).any(dim=0).int()
                        d_ppn = d[len(centroids):, :]
                        d_ppn_pos = (d_ppn < self.resolution * 2**(len(ppn_logits) - layer-1)).any(dim=0).int()
                        null = (d < self.resolution * 2**(len(ppn_logits) - layer-1)).any(dim=0)
                        null = (~null).int()
                        onehot = torch.cat([null.view(-1, 1), d_cent_pos.view(-1, 1), d_ppn_pos.view(-1, 1)], dim=1)
                        point_prediction_truth = torch.argmax(onehot, dim=1)
                        point_prediction_logits = point_predictions[batch_index]
                        # Calculate Weights
                        num_null = float((point_prediction_truth == 0).sum())
                        num_cent = float((point_prediction_truth == 1).sum())
                        num_ppn = float((point_prediction_truth == 2).sum())
                        denom = 1.0 / (num_null+1) + 1.0 / (num_cent+1) + 1.0 / (num_ppn+1) 
                        w = torch.zeros(3).to(device)
                        w[0] = (1.0 / (num_null+1)) / denom
                        w[1] = (1.0 / (num_cent+1)) / denom
                        w[2] = (1.0 / (num_ppn+1)) / denom
                        point_category_loss = self.point_lossfn(point_prediction_logits, point_prediction_truth, weight=w)
                        category_loss += point_category_loss
                        total_acc += float(torch.sum(torch.argmax(point_prediction_logits, dim=1) == point_prediction_truth)) \
                            / float(point_prediction_truth.shape[0])
                loss_layer /= num_batches
                attention_loss += loss_layer

        attention_loss /= num_batches
        category_loss /= num_batches
        total_acc /= num_batches
        res['loss'] = attention_loss + category_loss
        res['attention_loss'] = float(attention_loss)
        res['category_loss'] = float(category_loss)
        res['accuracy'] = total_acc


        return res