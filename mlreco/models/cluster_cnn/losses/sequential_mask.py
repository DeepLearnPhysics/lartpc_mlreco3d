import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import sparseconvnet as scn

from .lovasz import mean, lovasz_hinge_flat, StableBCELoss
from collections import defaultdict, namedtuple


def compute_iou(pred, labels, smoothing=1e-6):
    with torch.no_grad():
        intersection = (pred & labels).float().sum()
        union = (pred | labels).float().sum()
        iou = (intersection + smoothing) / (union + smoothing)
        return iou


# Sequential Mask Loss
class SequentialMaskLoss(nn.Module):
    '''
    Loss for sequential mask generating architectures (SMGA)
    SMGAs generate foreground/background scores for each instance
    mask sequentially.
    '''
    def __init__(self, cfg, name='clustering_loss'):
        super(SequentialMaskLoss, self).__init__()
        self.loss_config = cfg[name]
        self.loss_scheme = self.loss_config.get('loss_scheme', 'BCE')

        if self.loss_scheme == 'BCE':
            self.loss_fn = F.binary_cross_entropy_with_logits
        elif self.loss_scheme == 'lovasz_hinge':
            self.loss_fn = lovasz_hinge_flat
        elif self.loss_scheme == 'focal':
            raise NotImplementedError
        else:
            raise NotImplementedError

    def forward(self, cluster_masks, labels):
        '''
        INPUTS:
            - logits: list of namedtuples
            - labels: ground truth cluster labels.

        RETURNS:
            - res (dict): loss and accuracy results for backprop/logging.
        '''
        res = {}
        loss_dict = defaultdict(list)
        loss, acc = 0.0, 0.0
        batch_idx = labels[:, 3].unique()
        for bidx in batch_idx:
            clabels_event = labels[labels[:, 3] == bidx]
            num_clusters = len(clabels_event[:, -1].unique())
            logits_event = cluster_masks[int(bidx)]
            loss_event, acc_event = 0.0, 0.0
            for logits in logits_event:
                assert int(bidx) == logits.batch_id
                cluster_id = logits.group_id
                class_id = logits.class_id
                scores = logits.scores
                gt = clabels_event[:, -1] == cluster_id
                n1 = sum(gt == 1)
                weight = float(gt.shape[0]) / n1
                if torch.cuda.is_available():
                    weight = weight.cuda()
                l = self.loss_fn(scores, gt.float())
                acc = compute_iou(scores > 0, gt)
                #acc = 0.0
                loss_event += l
                acc_event += acc
                loss_dict['mask_loss_{}'.format(class_id)].append(float(l))
                loss_dict['acc_{}'.format(class_id)].append(float(acc))
            loss_event /= num_clusters
            acc_event /= num_clusters
            loss += loss_event
            acc += acc_event
        loss /= len(batch_idx)
        acc /= len(batch_idx)
        for key, val in loss_dict.items():
            res[key] = sum(val) / len(val)
        res['mask_loss'] = loss
        res['accuracy'] = acc
        return res
