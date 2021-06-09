import sys

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
from mlreco.bayes.evidential import EVDLoss

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


class EvidentialParticleClassifier(ParticleImageClassifier):

    MODULES = ['network_base', 'particle_image_classifier', 'mink_encoder']
    def __init__(self, cfg, name='evidential_image_classifier'):
        super(EvidentialParticleClassifier, self).__init__(cfg, name=name)
        self.final_layer_name = cfg[name].get('final_layer_name', 'relu')
        if self.final_layer_name == 'relu':
            self.final_layer = nn.Sequential(
                nn.Linear(self.encoder.latent_size, self.num_classes),
                nn.ReLU())
        elif self.final_layer_name == 'softplus':
            self.final_layer = nn.Sequential(
                nn.Linear(self.encoder.latent_size, self.num_classes),
                nn.Softplus())
        else:
            raise Exception("Unknown output activation name %s provided" % self.final_layer_name)
        self.eps = cfg[name].get('eps', 0.0)
        
    def forward(self, input):
        point_cloud, = input
        out = self.encoder(point_cloud)
        evidence = self.final_layer(out)
        # print("Evidence = ", evidence)
        concentration = evidence + 1.0
        S = torch.sum(concentration, dim=1, keepdim=True)
        uncertainty = self.num_classes / (S + self.eps)
        res = {
            'evidence': [evidence],
            'uncertainty': [uncertainty],
            'concentration': [concentration],
            'expected_probability': [concentration / S]
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

        self.num_samples = self.model_config.get('num_samples', 20)
        self.mode = self.model_config.get('mode', 'mc_dropout')

    def mc_forward(self, input, num_samples=None):

        with torch.no_grad():
            if num_samples is None:
                num_samples = self.num_samples

            print("Number of Samples = {}".format(num_samples))

            point_cloud, = input

            device = point_cloud.device

            for m in self.modules():
                if m.__class__.__name__ == 'Dropout':
                    m.train()

            num_batch = torch.unique(point_cloud[:, 0].int()).shape[0]

            pvec = torch.zeros((num_batch, self.num_classes)).to(device)
            logits = torch.zeros((num_batch, self.num_classes)).to(device)
            discrete = torch.zeros((num_batch, self.num_classes)).to(device)
            
            eye = torch.eye(self.num_classes).int().to(device)

            for i in range(num_samples):
                x = self.encoder(point_cloud)
                out = self.logit_layer(x)
                logits += out
                pred = torch.argmax(out, dim=1)
                pvec += F.softmax(out, dim=1)
                discrete += eye[pred]

            mc_dist = discrete / float(num_samples)
            softmax_probs = pvec / float(num_samples)
            logits = logits / float(num_samples)
            # logits = torch.logit(softmax_probs)
            res = {
                'softmax': [softmax_probs],
                'logits': [logits],
                'mc_dist': [mc_dist]
            }
            return res

    def standard_forward(self, input, verbose=False):
        if verbose:
            sys.stdout.write("Forwarding using weight averaging (standard dropout) ...")
            sys.stdout.flush()
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


class EvidentialLearningLoss(nn.Module):

    def __init__(self, cfg, name='evidential_learning_loss'):
        super(EvidentialLearningLoss, self).__init__()
        self.loss_config = cfg[name]
        self.evd_loss_name = self.loss_config.get('evd_loss_name', 'sumsq')
        self.num_classes = self.loss_config.get('num_classes', 5)
        self.num_total_iter = self.loss_config.get('num_total_iter', 50000)
        self.loss_fn = EVDLoss(self.evd_loss_name, 'mean', T=self.num_total_iter)

    def forward(self, out, type_labels, iteration=0):

        alpha = out['concentration'][0]
        probs = out['expected_probability'][0]
        device = alpha.device

        labels = type_labels[0][:, 0].to(dtype=torch.long)

        labels_onehot = torch.eye(self.num_classes, device=device)[labels]

        loss_batch = self.loss_fn(alpha, labels_onehot, t=iteration)
        loss = loss_batch.mean()
        pred = torch.argmax(probs, dim=1)

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