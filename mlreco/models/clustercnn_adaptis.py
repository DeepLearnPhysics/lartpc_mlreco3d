import torch
import torch.nn as nn
import numpy as np
import sparseconvnet as scn
from collections import defaultdict

from .cluster_cnn.adaptis import *
from .cluster_cnn.losses.sequential_mask import SequentialMaskLoss


class ClusterCNN(AdaptIS):

    def __init__(self, cfg, name='adaptis'):
        super(ClusterCNN, self).__init__(cfg, name=name)


class ClusteringLoss(nn.Module):

    def __init__(self, cfg, name='clustering_loss'):
        super(ClusteringLoss, self).__init__()
        self.loss_config = cfg[name]
        self.mask_loss = SequentialMaskLoss(cfg, name=name)
        self.seg_loss = torch.nn.CrossEntropyLoss(reduction='mean')
        self.ppn_loss = torch.nn.BCEWithLogitsLoss()
        self.segmentation_weight = self.loss_config.get('segmentation_weight', 1.0)
        self.clustering_weight = self.loss_config.get('clustering_weight', 1.0)


    def forward(self, out, semantic_labels, group_labels, particle_labels):

        res = {}
        num_gpu = len(semantic_labels)
        loss_seg, loss_ppn = 0.0, 0.0
        loss_dict, acc_dict = defaultdict(list), defaultdict(list)

        for igpu in range(num_gpu):
            segmentation = out['segmentation'][igpu]
            attention = out['ppn'][igpu]
            cluster_masks = out['instance_scores'][igpu]
            particles = particle_labels[igpu]
            clabels = group_labels[igpu]
            slabels = semantic_labels[igpu].float()
            loss_dict['seg_loss'].append(
                self.seg_loss(segmentation, slabels[:, -1].long()))
            # loss_dict['seg_acc'].append(
            #     torch.argmax(segmentation, dim=1))
            res_mask = self.mask_loss(cluster_masks, clabels)
            for key, val in res_mask.items():
                loss_dict[key].append(val)

        for key, val in loss_dict.items():
            res[key] = sum(val) / len(val)
        res['loss'] = self.segmentation_weight * res['seg_loss'] \
                    + self.clustering_weight * res['mask_loss']

        print(res)

        return res
