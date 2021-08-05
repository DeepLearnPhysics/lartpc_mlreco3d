import numpy as np
import torch
import torch.nn as nn
import time

import MinkowskiEngine as ME
import MinkowskiFunctional as MF

from mlreco.mink.layers.ppnplus import PPN, PPNLonelyLoss
from mlreco.models.uresnet_lonely import SegmentationLoss
from collections import defaultdict
from mlreco.models.mink_uresnet import UResNet_Chain

class UResNetPPN(nn.Module):

    MODULES = ['mink_uresnet', 'mink_uresnet_ppn_chain', 'mink_ppn']

    def __init__(self, cfg):
        super(UResNetPPN, self).__init__()
        self.model_config = cfg
        self.ghost = cfg['uresnet_lonely'].get('ghost', False)
        assert self.ghost == cfg['ppn'].get('ghost', False)
        self.backbone = UResNet_Chain(cfg)
        self.ppn = PPN(cfg)
        self.num_classes = self.backbone.num_classes
        self.num_filters = self.backbone.F
        self.segmentation = ME.MinkowskiLinear(
            self.num_filters, self.num_classes)

    def forward(self, input):

        labels = None

        if len(input) == 1:
            # PPN without true ghost mask propagation
            input_tensors = [input[0]]
        elif len(input) == 2:
            # PPN with true ghost mask propagation
            input_tensors = [input[0]]
            labels = input[1]

        out = defaultdict(list)

        for igpu, x in enumerate(input_tensors):
            # input_data = x[:, :5]
            res = self.backbone([x])
            out.update({'ghost': res['ghost']})
            if self.ghost:
                if self.ppn.use_true_ghost_mask:
                    res_ppn = self.ppn(res['finalTensor'][igpu], 
                                    res['decoderTensors'][igpu],
                                    ghost=res['ghost_sptensor'][igpu],
                                    ghost_labels=labels)
                else:
                    res_ppn = self.ppn(res['finalTensor'][igpu], 
                                    res['decoderTensors'][igpu],
                                    ghost=res['ghost_sptensor'][igpu])

            else:
                res_ppn = self.ppn(res['finalTensor'][igpu], 
                                   res['decoderTensors'][igpu])
            # if self.training:
            #     res_ppn = self.ppn(res['finalTensor'], res['encoderTensors'], particles_label)
            # else:
            #     res_ppn = self.ppn(res['finalTensor'], res['encoderTensors'])
            segmentation = self.segmentation(res['decoderTensors'][igpu][-1])
            out['segmentation'].append(segmentation.F)
            out.update(res_ppn)
            
        return out


class UResNetPPNLoss(nn.Module):

    def __init__(self, cfg):
        super(UResNetPPNLoss, self).__init__()
        self.ppn_loss = PPNLonelyLoss(cfg)
        self.segmentation_loss = SegmentationLoss(cfg)

    def forward(self, outputs, segment_label, particles_label, weights=None):

        res_segmentation = self.segmentation_loss(
            outputs, segment_label, weights=weights)

        res_ppn = self.ppn_loss(
            outputs, segment_label, particles_label)

        res = {
            'loss': res_segmentation['loss'] + res_ppn['ppn_loss'],
            'accuracy': (res_segmentation['accuracy'] + res_ppn['ppn_acc']) / 2.0,
            'reg_loss': res_ppn['reg_loss'],
            'type_loss': res_ppn['type_loss']
        }
        return res