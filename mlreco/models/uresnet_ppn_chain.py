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
        ["parse_sparse3d_scn", (float,), (3, 1)],
        ["parse_particle_points", (int,), (3, 1)]
    ]
    MODULES = ['ppn', 'uresnet_lonely']

    def __init__(self, model_config):
        super(Chain, self).__init__()
        self.ppn = PPN(model_config)
        self.uresnet_lonely = UResNet(model_config)
        self._freeze_uresnet = model_config['modules']['uresnet_lonely'].get('freeze', False)

        if self._freeze_uresnet:
            for param in self.uresnet_lonely.parameters():
                param.requires_grad = False

    def forward(self, input):
        """
        Assumes single GPU/CPU.
        No multi-GPU! (We select index 0 of input['ppn1_feature_enc'])
        """
        point_cloud, label = input
        # if self._freeze_uresnet:
        #     with torch.no_grad():
        #         x = self.uresnet_lonely((point_cloud,))
        # else:
        #     x = self.uresnet_lonely((point_cloud,))
        x = self.uresnet_lonely((point_cloud,))
        x['label'] = label
        y = {}
        y.update(x)
        y['ppn_feature_enc'] = y['ppn_feature_enc'][0]
        y['ppn_feature_dec'] = y['ppn_feature_dec'][0]
        if 'ghost' in y:
            y['ghost'] = y['ghost'][0]
        z = self.ppn(y)
        x.update(z)
        return x


class ChainLoss(torch.nn.modules.loss._Loss):
    """
    Loss for UResNet + PPN chain
    """
    INPUT_SCHEMA = [
        ["parse_sparse3d_scn", (int,), (3, 1)],
        ["parse_particle_points", (int,), (3, 1)]
    ]

    def __init__(self, cfg):
        super(ChainLoss, self).__init__()
        self.uresnet_loss = SegmentationLoss(cfg)
        self.ppn_loss = PPNLoss(cfg)

    def forward(self, result, label, particles):
        uresnet_res = self.uresnet_loss(result, label)
        ppn_res = self.ppn_loss(result, label, particles)
        uresnet_res.update(ppn_res)
        # Don't forget to sum all losses
        uresnet_res['loss'] = ppn_res['ppn_loss'].float() + uresnet_res['loss'].float()
        return uresnet_res
