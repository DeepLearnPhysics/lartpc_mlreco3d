import numpy as np
import torch
import torch.nn as nn
import time

import MinkowskiEngine as ME
import MinkowskiFunctional as MF

from mlreco.models.layers.common.ppnplus import PPN, PPNLonelyLoss
from mlreco.models.uresnet import SegmentationLoss
from collections import defaultdict
from mlreco.models.uresnet import UResNet_Chain

class UResNetPPN(nn.Module):
    """
    A model made of UResNet backbone and PPN layers. Typical configuration:

    .. code-block:: yaml

        model:
          name: uresnet_ppn_chain
          modules:
            uresnet_lonely:
              # Your uresnet config here
            ppn:
              # Your ppn config here

    Configuration
    -------------
    data_dim: int, default 3
    num_input: int, default 1
    allow_bias: bool, default False
    spatial_size: int, default 512
    leakiness: float, default 0.33
    activation: dict
        For activation function, defaults to `{'name': 'lrelu', 'args': {}}`
    norm_layer: dict
        For normalization function, defaults to `{'name': 'batch_norm', 'args': {}}`

    depth: int, default 5
        Depth of UResNet, also corresponds to how many times we down/upsample.
    filters: int, default 16
        Number of filters in the first convolution of UResNet.
        Will increase linearly with depth.
    reps: int, default 2
        Convolution block repetition factor
    input_kernel: int, default 3
        Receptive field size for very first convolution after input layer.

    num_classes: int, default 5
    score_threshold: float, default 0.5
    classify_endpoints: bool, default False
        Enable classification of points into start vs end points.
    ppn_resolution: float, default 1.0
    ghost: bool, default False
    downsample_ghost: bool, default True
    use_true_ghost_mask: bool, default False
    mask_loss_name: str, default 'BCE'
        Can be 'BCE' or 'LogDice'
    particles_label_seg_col: int, default -2
        Which column corresponds to particles' semantic label
    track_label: int, default 1

    See Also
    --------
    mlreco.models.uresnet.UResNet_Chain, mlreco.models.layers.common.ppnplus.PPN
    """
    MODULES = ['mink_uresnet', 'mink_uresnet_ppn_chain', 'mink_ppn']

    RETURNS = dict(UResNet_Chain.RETURNS, **PPN.RETURNS)

    def __init__(self, cfg):
        super(UResNetPPN, self).__init__()
        self.model_config = cfg
        self.ghost = cfg.get('uresnet_lonely', {}).get('ghost', False)
        assert self.ghost == cfg.get('ppn', {}).get('ghost', False)
        self.backbone = UResNet_Chain(cfg)
        self.ppn = PPN(cfg)

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
            res = self.backbone([x])
            out.update({'ghost': res['ghost'],
                        'segmentation': res['segmentation']})
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
            out.update(res_ppn)

        return out


class UResNetPPNLoss(nn.Module):
    """
    See Also
    --------
    mlreco.models.uresnet.SegmentationLoss, mlreco.models.layers.common.ppnplus.PPNLonelyLoss
    """

    RETURNS = {
        'loss': ['scalar'],
        'accuracy': ['scalar']
    }

    def __init__(self, cfg):
        super(UResNetPPNLoss, self).__init__()
        self.ppn_loss = PPNLonelyLoss(cfg)
        self.segmentation_loss = SegmentationLoss(cfg)

        self.RETURNS.update({'segmentation_'+k:v for k, v in self.segmentation_loss.RETURNS.items()})
        self.RETURNS.update({'ppn_'+k:v for k, v in self.ppn_loss.RETURNS.items()})

    def forward(self, outputs, segment_label, particles_label, weights=None):

        res_segmentation = self.segmentation_loss(
            outputs, segment_label, weights=weights)

        res_ppn = self.ppn_loss(
            outputs, segment_label, particles_label)

        res = {
            'loss': res_segmentation['loss'] + res_ppn['loss'],
            'accuracy': (res_segmentation['accuracy'] + res_ppn['accuracy'])/2
        }

        res.update({'segmentation_'+k:v for k, v in res_segmentation.items()})
        res.update({'ppn_'+k:v for k, v in res_ppn.items()})

        #for key, val in res.items():
        #    if 'ppn' in key:
        #        print('{}: {}'.format(key, val))

        return res
