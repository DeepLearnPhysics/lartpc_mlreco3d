import numpy as np
import torch
import torch.nn as nn


def segmentation_loss_dict():

    pass


class SegmentationLoss(nn.modules.loss._Loss):
    '''
    Compute semantic segmentation loss for one event.
    '''
    def __init__(self, cfg, name='segmentation_loss'):
        super(SegmentationLoss, self).__init__()

        self.loss_config = cfg[name]
        self.loss_type = self.loss_config.get('loss_type', 'cross_entropy')

        self.loss_fn = segmentation_loss_construct(self.loss_config)

    def forward(self, logits, labels, **kwargs):
        loss = self.loss_fn(segmentation, labels, **kwargs)
        pred = torch.argmax(logits, dim=1)
        acc = (pred == labels).sum().item() / float(labels.nelement())
        return loss, acc
