import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from mlreco.models.layers.common.cnn_encoder import SparseResidualEncoder
from collections import defaultdict, Counter, OrderedDict
from mlreco.models.layers.common.activation_normalization_factories import activations_construct
from mlreco.models.layers.common.configuration import setup_cnn_configuration
from mlreco.models.experimental.bayes.encoder import MCDropoutEncoder
from mlreco.models.experimental.bayes.evidential import EVDLoss
from mlreco.models.experimental.xai.simple_cnn import VGG16
from mlreco.models.layers.cluster_cnn.losses.lovasz import StableBCELoss

class ParticleImageClassifier(nn.Module):

    MODULES = ['particle_image_classifier', 'network_base', 'mink_encoder']

    def __init__(self, cfg, name='particle_image_classifier'):
        super(ParticleImageClassifier, self).__init__()
        self.encoder = SparseResidualEncoder(cfg)
        self.num_classes = cfg.get(name, {}).get('num_classes', 5)
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
        self.model_config = cfg.get(name, {})
        self.final_layer = None
        self.gamma = self.model_config.get('gamma', 0.99)
        self.sigma = self.model_config.get('sigma', 0.3)

        self.embedding_dim = self.model_config.get('embedding_dim', 10)
        self.latent_size = self.model_config.get('latent_size', 256)

        self.w = nn.Parameter(
            torch.normal(torch.zeros(self.embedding_dim, self.num_classes, self.latent_size), 1))

        self.register_buffer('N', torch.ones(self.num_classes) * 20)
        self.register_buffer('m', torch.normal(torch.zeros(self.embedding_dim, self.num_classes), 1))

        self.m = self.m * self.N.unsqueeze(0)

    def embed(self, x):

        feats = self.encoder(x)
        out = torch.einsum('ij,mnj->imn', feats, self.w)
        return out

    def bilinear(self, z):
        embeddings = self.m / self.N.unsqueeze(0)

        diff = z - embeddings.unsqueeze(0)
        y_pred = (- diff**2).mean(1).div(2 * self.sigma**2).exp()

        return y_pred

    def forward(self, input):

        point_cloud, = input
        if self.training:
            point_cloud.requires_grad_(True)

        z = self.embed(point_cloud)
        y_pred = self.bilinear(z)

        res = {
            'score': [y_pred],
            'embedding': [z],
            'input': [point_cloud],
            'centroids' : [self.m.detach().cpu().numpy() / self.N.detach().cpu().numpy()]
        }

        self.z = z
        self.y_pred = y_pred

        return res

    def update_buffers(self):
        with torch.no_grad():
            # normalizing value per class, assumes y is one_hot encoded
            self.N = torch.max(self.gamma * self.N + (1 - self.gamma) * self.y_pred.sum(0), torch.ones_like(self.N))
            # compute sum of embeddings on class by class basis
            features_sum = torch.einsum('ijk,ik->jk', self.z, self.y_pred)
            self.m = self.gamma * self.m + (1 - self.gamma) * features_sum



class EvidentialParticleClassifier(ParticleImageClassifier):

    MODULES = ['network_base', 'particle_image_classifier', 'mink_encoder']
    def __init__(self, cfg, name='evidential_image_classifier'):
        super(EvidentialParticleClassifier, self).__init__(cfg, name=name)
        self.final_layer_name = cfg.get(name, {}).get('final_layer_name', 'relu')
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
        self.eps = cfg.get(name, {}).get('eps', 0.0)

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


class BayesianParticleClassifier(torch.nn.Module):

    MODULES = ['network_base', 'bayesian_encoder']

    def __init__(self, cfg, name='bayesian_particle_classifier'):
        super(BayesianParticleClassifier, self).__init__()
        setup_cnn_configuration(self, cfg, 'network_base')

        self.model_config = cfg.get(name, {})
        self.num_classes = self.model_config.get('num_classes', 5)
        self.encoder = MCDropoutEncoder(cfg)

        self.mode = self.model_config.get('mode', 'mc_dropout')

        if self.mode == 'evidential':
            self.logit_layer = nn.Sequential(
                nn.Linear(self.encoder.latent_size, self.num_classes),
                nn.Softplus())
        else:
            self.logit_layer = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.encoder.latent_size, self.num_classes))

        self.num_samples = self.model_config.get('num_samples', 20)
        self.eps = self.model_config.get('eps', 0.0)
        print('Dropout network will run inference on {} mode'.format(self.mode))


    def evidential_forward(self, input):
        point_cloud, = input
        out = self.encoder(point_cloud)
        out = self.logit_layer(out) + self.eps

        concentration = out + 1.0
        S = torch.sum(concentration, dim=1, keepdim=True)
        uncertainty = self.num_classes / (S + 0.000001)

        res = {}

        res['evidence'] = [out]
        res['uncertainty'] = [uncertainty]
        res['concentration'] = [concentration]
        res['expected_probability'] = [concentration / S]

        return res


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
        elif self.mode == 'evidential':
            return self.evidential_forward(input)
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
        model_cfg = cfg.get(name, {})
        self.grad_w = model_cfg.get('grad_w', 0.0)
        self.grad_penalty = model_cfg.get('grad_penalty', True)


    @staticmethod
    def calc_gradient_penalty(x, y_pred):
        '''
        Code From the DUQ main Github Repository:
        https://github.com/y0ast/deterministic-uncertainty-quantification

        Author: Joost van Amersfoort
        '''
        gradients = torch.autograd.grad(
                outputs=y_pred,
                inputs=x,
                grad_outputs=torch.ones_like(y_pred),
                create_graph=True,
            )[0]

        gradients = gradients.flatten(start_dim=1)

        # L2 norm
        grad_norm = gradients.norm(2, dim=1)

        # Two sided penalty
        gradient_penalty = ((grad_norm - 1) ** 2).mean()

        # One sided penalty - down
    #     gradient_penalty = F.relu(grad_norm - 1).mean()

        return gradient_penalty

    def forward(self, out, type_labels):
        # print(type_labels)
        probas = out['score'][0]
        device = probas.device
        labels_one_hot = torch.eye(self.num_classes)[type_labels[0][:, 0].long()].to(device=device)
        loss1 = self.xentropy(probas, labels_one_hot)
        pred = torch.argmax(probas, dim=1)
        labels = type_labels[0][:, 0].long()

        # Comptue gradient penalty
        loss2 = 0
        if self.grad_penalty:
            loss2 = self.calc_gradient_penalty(out['input'][0], probas)

        loss = loss1.sum(dim=1).mean() + self.grad_w * loss2

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
        self.loss_config = cfg.get(name, {})
        self.evd_loss_name = self.loss_config.get('evd_loss_name', 'edl_sumsq')
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
