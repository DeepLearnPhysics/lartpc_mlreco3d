import numpy as np
import torch
import torch.nn as nn

import MinkowskiEngine as ME

from mlreco.mink.layers.uresnet import UResNet, ACASUNet, ASPPUNet
from collections import defaultdict
from mlreco.mink.layers.factories import activations_construct

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class UResNet_Chain(nn.Module):


    INPUT_SCHEMA = [
        ["parse_sparse3d_scn", (float,), (3, 1)]
    ]

    MODULES = ['uresnet_lonely']

    def __init__(self, cfg, name='uresnet_lonely'):
        super(UResNet_Chain, self).__init__()
        self.model_config = cfg[name]
        mode = self.model_config.get('aspp_mode', None)
        self.D = self.model_config.get('data_dim', 3)
        self.F = self.model_config.get('num_filters', 16)
        self.num_classes = self.model_config.get('num_classes', 5)\

        # Parameters for Deghosting
        self.ghost = self.model_config.get('ghost', False)
        self.ghost_label = self.model_config.get('ghost_label', -1)


        if mode == 'acas':
            self.net = ACASUNet(cfg, name=name)
        elif mode == 'aspp':
            self.net = ASPPUNet(cfg, name=name)
        else:
            self.net = UResNet(cfg, name=name)

        self.output = [
            ME.MinkowskiBatchNorm(self.F,
                eps=self.net.norm_args.get('eps', 0.00001),
                momentum=self.net.norm_args.get('momentum', 0.1)),
            activations_construct('lrelu', negative_slope=0.33),
            ME.MinkowskiLinear(self.F, self.num_classes)]
        self.output = nn.Sequential(*self.output)

        if self.ghost:
            self.linear_ghost = ME.MinkowskiLinear(self.F, 2)

        print('Total Number of Trainable Parameters (mink_uresnet)= {}'.format(
                    sum(p.numel() for p in self.parameters() if p.requires_grad)))
        #print(self)
    def forward(self, input):
        out = defaultdict(list)
        for igpu, x in enumerate(input):
            res = self.net(x)
            feats = res['decoderTensors'][-1]
            seg = self.output(feats)
            out['segmentation'].append(seg.F)
            out['finalTensor'].append(res['finalTensor'])
            out['encoderTensors'].append(res['encoderTensors'])
            out['decoderTensors'].append(res['decoderTensors'])
            if self.ghost:
                ghost = self.linear_ghost(feats)
                out['ghost'].append(ghost.F)
                out['ghost_sptensor'].append(ghost)
        return out


class SegmentationLoss(nn.Module):

    def __init__(self, cfg, name='segmentation_loss'):
        super(SegmentationLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

    def forward(self, outputs, label, weight=None):
        '''
        segmentation[0], label and weight are lists of size #gpus = batch_size.
        segmentation has as many elements as UResNet returns.
        label[0] has shape (N, dim + batch_id + 1)
        where N is #pts across minibatch_size events.
        '''
        # TODO Add weighting
        segmentation = outputs['segmentation']

        assert len(segmentation) == len(label)
        # if weight is not None:
        #     assert len(data) == len(weight)
        batch_ids = [d[:, 0] for d in label]
        total_loss = 0
        total_acc = 0
        count = 0
        # Loop over GPUS
        for i in range(len(segmentation)):
            for b in batch_ids[i].unique():
                batch_index = batch_ids[i] == b
                event_segmentation = segmentation[i][batch_index]
                event_label = label[i][:, -1][batch_index]
                event_label = torch.squeeze(event_label, dim=-1).long()
                loss_seg = self.cross_entropy(event_segmentation, event_label)
                if weight is not None:
                    event_weight = weight[i][batch_index]
                    event_weight = torch.squeeze(event_weight, dim=-1).float()
                    total_loss += torch.mean(loss_seg * event_weight)
                else:
                    total_loss += torch.mean(loss_seg)
                # Accuracy
                predicted_labels = torch.argmax(event_segmentation, dim=-1)
                acc = (predicted_labels == event_label).sum().item() / float(predicted_labels.nelement())
                total_acc += acc
                count += 1

        return {
            'accuracy': total_acc/count,
            'loss': total_loss/count
        }
