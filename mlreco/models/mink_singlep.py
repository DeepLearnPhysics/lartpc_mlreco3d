import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import MinkowskiEngine as ME
import MinkowskiFunctional as MF

from mlreco.mink.layers.cnn_encoder import SparseResidualEncoder
from collections import defaultdict
from mlreco.mink.layers.factories import activations_construct
from mlreco.mink.layers.network_base import MENetworkBase
from mlreco.bayes.encoder import BayesianEncoder


class ParticleImageClassifier(nn.Module):

    MODULES = ['particle_image_classifier', 'network_base', 'mink_encoder']

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


class BayesianParticleClassifier(MENetworkBase):

    MODULES = ['network_base', 'bayesian_encoder']

    def __init__(self, cfg, name='bayesian_particle_classifier'):
        super(BayesianParticleClassifier, self).__init__(cfg)
        self.model_config = cfg[name]
        self.num_classes = self.model_config.get('num_classes', 5)
        self.encoder_type = self.model_config.get('encoder_type', 'full_dropout')
        self.encoder = BayesianEncoder(cfg)
        self.logit_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.encoder.latent_size, self.num_classes))

        self.num_samples = self.model_config.get('num_samples', None)
        self.mode = self.model_config.get('mode', 'mc_dropout')

        print()

    def mc_forward(self, input):
        assert self.num_samples is not None

        print("Number of Samples = {}".format(self.num_samples))

        point_cloud, = input

        for m in self.modules():
            if m.__class__.__name__ == 'Dropout':
                m.train()

        pvec = []

        logits = []

        num_batch = torch.unique(point_cloud[:, 0].int()).shape[0]

        discrete = torch.zeros((self.num_samples, num_batch, self.num_classes)).int()
        
        eye = torch.eye(self.num_classes).int()

        for i in range(self.num_samples):
            x = self.encoder(point_cloud)
            out = self.logit_layer(x)
            logits.append(out)
            pred = torch.argmax(out, dim=1)
            pvec.append(F.softmax(out, dim=1))
            discrete[i, :, :] = eye[pred]

        mc_dist = discrete.sum(axis=0).float() / float(self.num_samples)
        softmax_probs = torch.stack(pvec).mean(dim=0)
        logits = torch.stack(logits).mean(dim=0)
        # logits = torch.logit(softmax_probs)
        res = {
            'softmax': [softmax_probs],
            'logits': [logits],
            'mc_dist': [mc_dist]
        }
        return res

    def standard_forward(self, input):
        print("Forwarding using weight averaging (standard dropout) ...")
        point_cloud, = input
        out = self.encoder(point_cloud)
        out = self.logit_layer(out)
        res = {
            'logits': [out]
        }
        return res

        
    def forward(self, input):
        if (not self.training) and (self.mode == 'mc_dropout'):
            return self.mc_forward(input)
        else:
            return self.standard_forward(input)


class ParticleTypeLoss(nn.Module):

    def __init__(self, cfg, name='particle_type_loss'):
        super(ParticleTypeLoss, self).__init__()
        self.xentropy = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, out, type_labels):
        # print(type_labels)
        logits = out['logits'][0]
        labels = type_labels[0][:, 0].to(dtype=torch.long)
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
