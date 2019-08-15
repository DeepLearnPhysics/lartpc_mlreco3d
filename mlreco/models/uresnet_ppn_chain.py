from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
from mlreco.models.uresnet_lonely import UResNet, SegmentationLoss
from mlreco.models.ppn import PPN, PPNLoss


class Chain(torch.nn.Module):
    """
    Run UResNet and use its encoding/decoding feature maps for PPN layers
    """
    INPUT_SCHEMA = [
        ["parse_sparse3d_scn", (float,)],
        ["parse_particle_points", (int,)]
    ]
    MODULES = ['ppn', 'uresnet_lonely']

    def __init__(self, model_config):
        super(Chain, self).__init__()
        self.ppn = PPN(model_config)
        self.uresnet_lonely = UResNet(model_config)

    def forward(self, input):
        point_cloud, label = input
        x = self.uresnet_lonely((point_cloud,))
        y = self.ppn((label, x[0][0], x[1][0], x[2][0]))
        return [x[0]] + y


class ChainLoss(torch.nn.modules.loss._Loss):
    """
    Loss for UResNet + PPN chain
    """
    INPUT_SCHEMA = [
        ["parse_sparse3d_scn", (int,)],
        ["parse_particle_points", (int,)]
    ]

    def __init__(self, cfg):
        super(ChainLoss, self).__init__()
        self.uresnet_loss = SegmentationLoss(cfg)
        self.ppn_loss = PPNLoss(cfg)

    def forward(self, segmentation, label, particles):
        uresnet_res = self.uresnet_loss([segmentation[0]], label)
        ppn_res = self.ppn_loss(segmentation[1:], label, particles)
        res = { **ppn_res, **uresnet_res }
        res['uresnet_acc'] = uresnet_res['accuracy']
        res['uresnet_loss'] = uresnet_res['loss']
        # Don't forget to sum all losses
        res['loss'] = ppn_res['loss_ppn1'].float() + ppn_res['loss_ppn2'].float() + \
                        ppn_res['loss_class'].float() + ppn_res['loss_distance'].float() \
                        + uresnet_res['loss'].float()
        return res
