import numpy as np
import torch
import torch.nn as nn
import time

import MinkowskiEngine as ME
import MinkowskiFunctional as MF

from mlreco.mink.layers.ppnplus import PPN, PPNLonelyLoss
from mlreco.models.mink_uresnet import SegmentationLoss
from collections import defaultdict
from mlreco.models.mink_uresnet import UResNet_Chain

from pprint import pprint
class UResNetPPN(nn.Module):

    MODULES = ['mink_uresnet', 'mink_uresnet_ppn_chain', 'mink_ppn']

    def __init__(self, cfg, name='mink_uresnet_ppn_chain'):
        super(UResNetPPN, self).__init__()
        self.model_config = cfg[name]
        self.backbone = UResNet_Chain(self.model_config)
        self.ppn = PPN(self.model_config)
        self.num_classes = self.model_config.get('num_classes', 5)
        self.num_filters = self.model_config.get('num_filters', 16)
        self.segmentation = ME.MinkowskiLinear(
            self.num_filters, self.num_classes)

    def forward(self, input):
        device = input[0].device

        out = defaultdict(list)
        input_tensors = [input[0]]

        for igpu, x in enumerate(input_tensors):
            input_data = x[:, :5]
            res = self.backbone([input_data])
            res_ppn = self.ppn(res['finalTensor'], res['encoderTensors'])
            # if self.training:
            #     res_ppn = self.ppn(res['finalTensor'], res['encoderTensors'], particles_label)
            # else:
            #     res_ppn = self.ppn(res['finalTensor'], res['encoderTensors'])
            segmentation = self.segmentation(res['decoderTensors'][-1])
            out['segmentation'].append(segmentation.F)
            out.update(res_ppn)
            
        return out


class UResNetPPNLoss(nn.Module):

    def __init__(self, cfg, name='mink_uresnet_ppn_chain'):
        super(UResNetPPNLoss, self).__init__()
        self.model_config = cfg[name]
        self.ppn_loss = PPNLonelyLoss(self.model_config)
        self.segmentation_loss = SegmentationLoss(self.model_config)

    def forward(self, outputs, segment_label, particles_label, weight=None):

        res_segmentation = self.segmentation_loss(
            outputs, segment_label, weight=weight)

        res_ppn = self.ppn_loss(
            outputs, segment_label, particles_label)

        res = {
            'loss': res_segmentation['loss'] + res_ppn['loss'],
            'accuracy': (res_segmentation['accuracy'] + res_ppn['accuracy']) / 2.0,
            'reg_loss': res_ppn['reg_loss'],
            'type_loss': res_ppn['type_loss']
        }
        return res