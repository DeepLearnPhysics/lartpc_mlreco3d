import torch
import torch.nn as nn
import MinkowskiEngine as ME
import torch.nn.functional as F

from collections import defaultdict
from mlreco.models.layers.common.activation_normalization_factories import (activations_dict,
                                          activations_construct,
                                          normalizations_construct)
from mlreco.models.layers.common.configuration import setup_cnn_configuration
from mlreco.models.experimental.bayes.encoder import MCDropoutEncoder
from mlreco.models.experimental.bayes.decoder import MCDropoutDecoder
from mlreco.models.experimental.bayes.factories import uq_classification_loss_construct

class BayesianUResNet(torch.nn.Module):

    def __init__(self, cfg, name='bayesian_uresnet'):
        super(BayesianUResNet, self).__init__()
        setup_cnn_configuration(self, cfg, name)

        self.model_config = cfg[name]
        self.num_classes = self.model_config.get('num_classes', 5)
        self.num_samples = self.model_config.get('num_samples', 20)

        self.encoder = BayesianEncoder(cfg)
        self.decoder = MCDropoutDecoder(cfg)

        self.mode = self.model_config.get('mode', 'standard')

        if 'evd' in self.model_config.get('loss_fn', 'cross_entropy'):
            self.classifier = nn.Sequential(
                ME.MinkowskiLinear(self.encoder.num_filters, self.num_classes),
                ME.MinkowskiSoftplus()
            )
        else:
            self.classifier = ME.MinkowskiLinear(self.encoder.num_filters,
                                                self.num_classes)

    def mc_forward(self, input, num_samples=None):

        res = defaultdict(list)

        if num_samples is None:
            num_samples = self.num_samples

        for m in self.modules():
            if m.__class__.__name__ == 'Dropout':
                m.train()

        for igpu, x in enumerate(input):

            num_voxels = x.shape[0]

            device = x.device

            x_sparse = ME.SparseTensor(coordinates=x[:, :4],
                                       features=x[:, -1].view(-1, 1))

            pvec = torch.zeros((num_voxels, self.num_classes)).to(device)
            logits = torch.zeros((num_voxels, self.num_classes)).to(device)

            for i in range(num_samples):
                res_encoder = self.encoder.encoder(x_sparse)
                decoderTensors = self.decoder(
                    res_encoder['finalTensor'], res_encoder['encoderTensors'])
                feats = decoderTensors[-1]
                out = self.classifier(feats)
                logits += out.F
                pvec += F.softmax(out.F, dim=1)

            logits /= num_samples
            softmax_probs = pvec / num_samples

            res['softmax'].append(softmax_probs)
            res['segmentation'].append(logits)

        return res

    def evidential_forward(self, input):

        out = defaultdict(list)
        for igpu, x in enumerate(input):
            x = ME.SparseTensor(coordinates=x[:, :4],
                                features=x[:, -1].view(-1, 1))
            res_encoder = self.encoder.encoder(x)
            print([t.F.shape for t in res_encoder['encoderTensors']])
            decoderTensors = self.decoder(res_encoder['finalTensor'],
                                          res_encoder['encoderTensors'])
            feats = decoderTensors[-1]
            # For evidential models, logits correspond to collected evidence.
            logits = self.classifier(feats)
            ev = logits.F
            concentration = ev + 1.0
            S = torch.sum(concentration, dim=1, keepdim=True)
            uncertainty = self.num_classes / (S + 0.000001)
            out['segmentation'].append(ev)
            out['evidence'].append(ev)
            out['uncertainty'].append(uncertainty)
            out['concentration'].append(concentration)
            out['expected_probability'].append(concentration / S)
        return out


    def standard_forward(self, input):

        out = defaultdict(list)
        for igpu, x in enumerate(input):
            x = ME.SparseTensor(coordinates=x[:, :4],
                                features=x[:, -1].view(-1, 1))
            res_encoder = self.encoder.encoder(x)
            print([t.F.shape for t in res_encoder['encoderTensors']])
            decoderTensors = self.decoder(res_encoder['finalTensor'],
                                          res_encoder['encoderTensors'])
            feats = decoderTensors[-1]
            # For evidential models, logits correspond to collected evidence.
            logits = self.classifier(feats)
            out['segmentation'].append(logits.F)
        return out

    def forward(self, input):

        if self.mode == 'mc_dropout':
            return self.mc_forward(input)
        elif self.mode == 'evidential':
            return self.evidential_forward(input)
        else:
            return self.standard_forward(input)


class DUQUResNet(MENetworkBase):

    def __init__(self, cfg, name='duq_uresnet'):
        super(DUQUResNet, self).__init__(cfg)
        self.model_config = cfg[name]
        self.num_classes = self.model_config.get('num_classes', 5)
        self.num_samples = self.model_config.get('num_samples', 20)

        self.net = UResNet(cfg)

        self.gamma = self.model_config.get('gamma', 0.999)
        self.sigma = self.model_config.get('sigma', 0.3)

        self.embedding_dim = self.model_config.get('embedding_dim', 10)
        self.latent_size = self.model_config.get('latent_size', 32)

        self.w = nn.Parameter(torch.zeros(self.embedding_dim, self.num_classes, self.latent_size))

        nn.init.kaiming_normal_(self.w, nonlinearity='relu')

        self.register_buffer('N', torch.ones(self.num_classes) * 20)
        self.register_buffer('m', torch.normal(torch.zeros(self.embedding_dim, self.num_classes), 0.05))

        self.m = self.m * self.N.unsqueeze(0)

    def embed(self, x):

        res = self.net(x)
        feats = res['decoderTensors'][-1]
        print(feats.F)
        out = torch.einsum('ij,mnj->imn', feats.F, self.w)
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

        print(res['score'][0].shape)
        print(res['embedding'][0].shape)

        
        return res

    def update_buffers(self):
        with torch.no_grad():
            # normalizing value per class, assumes y is one_hot encoded
            self.N = torch.max(self.gamma * self.N + (1 - self.gamma) * self.y_pred.sum(0), torch.ones_like(self.N))
            # compute sum of embeddings on class by class basis
            features_sum = torch.einsum('ijk,ik->jk', self.z, self.y_pred)
            self.m = self.gamma * self.m + (1 - self.gamma) * features_sum


class SegmentationLoss(nn.Module):

    def __init__(self, cfg, name='bayesian_uresnet'):
        super(SegmentationLoss, self).__init__()
        self.loss_config = cfg[name]
        self.loss_fn_name = self.loss_config['loss_fn']
        self.loss_fn = uq_classification_loss_construct(self.loss_fn_name)
        self.one_hot = self.loss_config.get('one_hot', False)
        self.num_classes = self.loss_config.get('num_classes', 5)


    def forward(self, outputs, label, iteration=0, weight=None):
        '''
        segmentation[0], label and weight are lists of size #gpus = batch_size.
        segmentation has as many elements as UResNet returns.
        label[0] has shape (N, dim + batch_id + 1)
        where N is #pts across minibatch_size events.
        '''
        # TODO Add weighting
        logits = outputs['segmentation']
        if 'evd' in self.loss_fn_name:
            segmentation = [logits[0] + 1.0] # convert evidence to alpha concentration params.
        else:
            segmentation = logits
        device = segmentation[0].device
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
                loss_label = event_label
                if self.one_hot:
                    loss_label = torch.eye(self.num_classes, device=device)[event_label]
                    loss_seg = self.loss_fn(event_segmentation, loss_label, t=iteration)
                else:
                    loss_seg = self.loss_fn(event_segmentation, loss_label)
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


class DUQSegmentationLoss(nn.Module):

    def __init__(self, cfg, name='duq_uresnet'):
        super(DUQSegmentationLoss, self).__init__()
        self.xentropy = nn.BCELoss(reduction='none')
        self.num_classes = 5
        self.grad_w = cfg[name].get('grad_w', 0.0)
        self.grad_penalty = cfg[name].get('grad_penalty', True)


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
        labels = type_labels[0][:, -1].long()
        labels_one_hot = torch.eye(self.num_classes)[labels].to(device=device)
        loss1 = self.xentropy(probas, labels_one_hot)
        pred = torch.argmax(probas, dim=1)

        # Comptue gradient penalty
        loss2 = 0
        if self.grad_penalty:
            loss2 = self.calc_gradient_penalty(out['input'][0], probas)

        loss1 = loss1.sum(dim=1).mean()
        loss = loss1 + self.grad_w * loss2

        accuracy = float(torch.sum(pred == labels)) / float(labels.shape[0])

        res = {
            'loss': loss,
            'loss_embedding': float(loss1),
            'loss_grad_penalty': float(loss2),
            'accuracy': accuracy
        }

        print(res)

        acc_types = {}
        for c in labels.unique():
            mask = labels == c
            acc_types['accuracy_{}'.format(int(c))] = \
                float(torch.sum(pred[mask] == labels[mask])) / float(torch.sum(mask))
        return res