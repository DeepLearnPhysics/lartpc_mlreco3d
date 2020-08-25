import numpy as np
import torch
import torch.nn as nn
import time

import MinkowskiEngine as ME
import MinkowskiFunctional as MF

from mlreco.mink.chain.full_chain import FullChainCNN1
from mlreco.mink.layers.ppn import PPNLoss
from mlreco.mink.chain.factories import *
from collections import defaultdict


class FullChain(nn.Module):

    def __init__(self, cfg, name='full_chain'):
        super(FullChain, self).__init__()
        self.model_cfg = cfg[name]
        # print('-------------Full Chain Model Config----------------')
        # pprint(self.model_cfg)
        self.net = chain_construct(self.model_cfg['name'], self.model_cfg)

        self.seg_F = self.model_cfg.get('seg_features', 16)
        self.num_classes = self.model_cfg.get('num_classes', 5)


    def forward(self, input):
        device = input[0].device
        out = defaultdict(list)

        for igpu, x in enumerate(input):
            input_data = x[:, :5]

            # CNN Phase
            res = self.net(input_data)
            for key, val in res.items():
                out[key].append(val[igpu])
            # print([k for k in ppn_output.keys()])
            # print([k for k in res.keys()])
            segmentation = res['segmentation'][igpu]
            embeddings = res['embeddings'][igpu]
            margins = res['margins'][igpu]

            out['segmentation']

            
        return out


class ChainLoss(nn.Module):

    def __init__(self, cfg, name='segmentation_loss'):
        super(ChainLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        self.loss_config = cfg['full_chain']
        self.ppn_type_weight = self.loss_config['ppn'].get('ppn_type_weight', 1.0)
        self.ppn_loss_weight = self.loss_config['ppn'].get('ppn_loss_weight', 1.0)
        # PPN Loss
        self.ppn_loss = PPNLoss(self.loss_config)


    def forward(self, outputs, segment_label, particles_label, graph, weight=None):
        '''
        segmentation[0], label and weight are lists of size #gpus = batch_size.
        segmentation has as many elements as UResNet returns.
        label[0] has shape (N, dim + batch_id + 1)
        where N is #pts across minibatch_size events.
        '''
        # TODO Add weighting
        segmentation = outputs['segmentation']
        assert len(segmentation) == len(segment_label)
        batch_ids = [d[:, 0] for d in segment_label]
        highE = [t[t[:, -1].long() != 4] for t in segment_label]
        total_loss = 0
        total_acc = 0
        count = 0

        loss, accuracy = 0, []
        res = {}
        # Semantic Segmentation Loss
        for i in range(len(segmentation)):
            for b in batch_ids[i].unique():
                batch_index = batch_ids[i] == b
                event_segmentation = segmentation[i][batch_index]
                event_label = segment_label[i][:, -1][batch_index]
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
                acc = (predicted_labels == event_label).sum().item() / \
                    float(predicted_labels.nelement())
                total_acc += acc
                count += 1

        loss_seg = total_loss / count
        acc_seg = total_acc / count
        res['loss_seg'] = float(loss_seg)
        res['acc_seg'] = float(acc_seg)
        loss += loss_seg
        accuracy.append(acc_seg)

        # PPN Loss
        # ppn_results = self.ppn_loss(outputs, segment_label, particles_label)
        # loss += ppn_results['ppn_loss'] * self.ppn_loss_weight
        # loss += ppn_results['loss_type'] * self.ppn_type_weight
        # accuracy.append(float(ppn_results['ppn_acc']))
        # accuracy.append(float(ppn_results['acc_ppn_type']))
        # res['ppn_loss'] = float(ppn_results['ppn_loss'] * self.ppn_loss_weight)
        # res['ppn_type_loss'] = float(ppn_results['loss_type'] * self.ppn_type_weight)
        # res['ppn_acc'] = ppn_results['ppn_acc']
        # res['ppn_type_acc'] = ppn_results['acc_ppn_type']

        return res
