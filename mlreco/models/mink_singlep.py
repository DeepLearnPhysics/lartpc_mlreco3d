import numpy as np
import torch
import torch.nn as nn

import MinkowskiEngine as ME
import MinkowskiFunctional as MF

from mlreco.mink.layers.cnn_encoder import SparseResidualEncoder
from collections import defaultdict
from mlreco.mink.layers.factories import activations_construct


class ParticleImageClassifier(nn.Module):

    MODULES = ['particle_image_classifier', 'network_base', 'res_encoder', 'uresnet_encoder']

    def __init__(self, cfg, name='particle_image_classifier'):
        super(ParticleImageClassifier, self).__init__()
        self.encoder = SparseResidualEncoder(cfg)
        self.num_classes = cfg[name].get('num_classes', 5)
        self.final_layer = nn.Linear(self.encoder.latent_size, self.num_classes)

        print('Total Number of Trainable Parameters = {}'.format(
                    sum(p.numel() for p in self.parameters() if p.requires_grad)))

    def forward(self, input):
        point_cloud, = input
        out = self.encoder(point_cloud)
        out = self.final_layer(out)
        res = {
            'logits': [out]
        }
        return res


class ParticleTypeLoss(nn.Module):

    def __init__(self, cfg, name='particle_type_loss'):
        super(ParticleTypeLoss, self).__init__()
        self.xentropy = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, out, type_labels):
        # print(type_labels)
        logits = out['logits'][0]
        labels = type_labels[0][:, 0].to(dtype=torch.long)
        # print(labels)
        loss = self.xentropy(logits, labels)
        pred = torch.argmax(logits, dim=1)

        accuracy = float(torch.sum(pred == labels)) / float(labels.shape[0])

        res = {
            'loss': loss,
            'accuracy': accuracy
        }
        acc_types = {}
        for c in labels.unique():
            mask = labels == c
            acc_types['accuracy_{}'.format(int(c))] = \
                float(torch.sum(pred[mask] == labels[mask])) / float(torch.sum(mask))
        return res
