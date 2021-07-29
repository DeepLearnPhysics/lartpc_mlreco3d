import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import MinkowskiEngine as ME
import MinkowskiFunctional as MF

from mlreco.mink.layers.cnn_encoder import SparseResidualEncoder
from collections import defaultdict, Counter, OrderedDict
from mlreco.mink.layers.factories import activations_construct
from mlreco.mink.layers.network_base import MENetworkBase
from mlreco.bayes.encoder import BayesianEncoder
from mlreco.bayes.evidential import EVDLoss
from mlreco.xai.simple_cnn import VGG16
from mlreco.models.cluster_cnn.losses.lovasz import StableBCELoss

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


class SingleParticleVGG(nn.Module):

    MODULES = ['vgg', 'network_base']

    def __init__(self, cfg, name='vgg_classifier'):
        super(SingleParticleVGG, self).__init__()
        self.net = VGG16(cfg)

    def forward(self, input):
        point_cloud, = input
        out = self.net(point_cloud)
        # out = self.final_layer(out)
        res = {
            'logits': [out]
        }
        return res

class DUQParticleClassifier(ParticleImageClassifier):
    """
    Uncertainty Estimation Using a Single Deep Deterministic Neural Network
    https://arxiv.org/pdf/2003.02037.pdf
    Joost van Amersfoort, Lewis Smith, Yee Whye Teh, Yarin Gal.

    Pytorch Implementation for SparseConvNets with MinkowskiEngine backend.
    """

    MODULES = ['network_base', 'particle_image_classifier', 'mink_encoder']

    def __init__(self, cfg, name='duq_particle_classifier'):
        super(DUQParticleClassifier, self).__init__(cfg, name=name)

        self.batch_size = cfg[name].get('batch_size', 512)

        if cfg[name].get('grad_w', 0.0) > 0:
            self.grad_penalty = True
        else:
            self.grad_penalty = False

        self.final_layer = None
        self.embedding_dim = cfg[name].get('embedding_dim', 256)
        self.embedding_layer = nn.Sequential(
            nn.BatchNorm1d(self.encoder.latent_size),
            nn.ReLU(),
            nn.Linear(self.encoder.latent_size, self.embedding_dim)
        )
        self.centroid_size = cfg[name].get('centroid_size', 256)

        self.num_classes = cfg[name].get('num_classes', 5)
        self.weight_matrices = nn.ModuleList()

        for i in range(self.num_classes):
            self.weight_matrices.append(nn.Linear(self.embedding_dim, self.centroid_size))

        self.sigma_layer = nn.Sequential(
            nn.Linear(self.centroid_size, 1),
            nn.Softplus())
        
        self.momentum = cfg[name].get('momentum', 0.99)
        self.n_classes = None
        self.m = None

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.centroids_init()


    def centroids_init(self):

        self.centroids = torch.normal(
            mean=torch.zeros((self.centroid_size, 5)), 
            std=torch.ones((self.centroid_size, 5))).to(self.device) * 0.01
        
        self.centroids.requires_grad = True


    def kernel(self, em, sigma):

        numer = (1.0 / float(self.centroid_size)) * \
                 torch.sum(torch.pow(em - self.centroids.view(1, -1, self.num_classes), 2), dim=1)
        denom = 2.0 * torch.pow(sigma, 2)
        return torch.exp(-numer / (denom + 1e-6))


    def _forward(self, input):
        point_cloud, = input
        out = self.encoder(point_cloud)
        features = self.embedding_layer(out)
        embeddings = []
        for layer in self.weight_matrices:
            rbf_feature = layer(features)
            embeddings.append(
                rbf_feature.view(rbf_feature.shape[0], rbf_feature.shape[1], 1))
        embeddings = torch.cat(embeddings, dim=2)
        length_scale = self.sigma_layer(features)
        res = {
            'embeddings': [embeddings],
            'length_scale': [length_scale]
        }
        return res

    def update_state(self, embeddings, scores):
        with torch.no_grad():
            pred = torch.argmax(scores, dim=1).int()
            counts = torch.bincount(pred, minlength=self.num_classes)
            # Update class counts
            if self.n_classes is None:
                self.n_classes = counts
            else:
                self.n_classes = self.momentum * self.n_classes + (1 - self.momentum) * counts
            # Update m_ct
            if self.m is None:
                self.m = torch.sum(embeddings, dim=0)
            else:
                self.m = self.m * self.momentum + (1 - self.momentum) * torch.sum(embeddings, dim=0)
            
            self.centroids = self.m.view(-1, self.num_classes) / self.n_classes.view(1, -1)

    def train_forward(self, input, label):
        if self.grad_penalty:
            input[0].requires_grad_(True)
        res = self._forward(input)
        res['inputs'] = [input]
        embeddings = res['embeddings'][0]
        length_scale = res['length_scale'][0]
        scores = self.kernel(embeddings, length_scale)
        self.update_state(embeddings, scores)
        res['probas'] = [scores]
        return res

    def inference_forward(self, input):
        assert self.centroids is not None

    def forward(self, input, label=None):
        res = self.train_forward(input, label)
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


class MultiLabelCrossEntropy(nn.Module):

    def __init__(self, cfg, name='duq_particle_classifier'):
        super(MultiLabelCrossEntropy, self).__init__()
        self.xentropy = nn.BCELoss(reduction='none')
        self.num_classes = 5
        self.grad_w = cfg[name].get('grad_w', 0.0)

    def calculate_io_gradient(self, x, scores):
        gradients = torch.autograd.grad(
            outputs=scores,
            inputs=x,
            grad_outputs=torch.ones_like(scores),
            create_graph=True)[0]
        gradients = gradients.flatten(start_dim=1)
        return gradients


    def gradient_penalty(self, x, scores):

        gradients = self.calculate_io_gradient(x, scores)
        grad_norm = gradients.norm(2, dim=1)
        grad_loss = ((grad_norm - 1)**2).mean()
        return grad_loss

    def forward(self, out, type_labels):
        # print(type_labels)
        probas = out['probas'][0]
        device = probas.device
        labels_one_hot = torch.eye(self.num_classes)[type_labels[0][:, 0].long()].to(device=device)
        loss = self.xentropy(probas, labels_one_hot)
        K_c_sum = loss.sum(dim=1)
        loss = K_c_sum.mean()
        pred = torch.argmax(probas, dim=1)
        labels = type_labels[0][:, 0].long()

        # Comptue gradient penalty
        grad_penalty = self.gradient_penalty(out['inputs'][0], K_c_sum)

        loss += grad_penalty * self.grad_w

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