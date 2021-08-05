import torch
import torch.nn as nn
import MinkowskiEngine as ME
import torch.nn.functional as F
import MinkowskiFunctional as MF

from collections import defaultdict
from mlreco.mink.layers.factories import (activations_dict,
                                          activations_construct,
                                          normalizations_construct)
from mlreco.mink.layers.network_base import MENetworkBase
from mlreco.bayes.encoder import BayesianEncoder
from mlreco.bayes.decoder import MCDropoutDecoder
from mlreco.bayes.factories import uq_classification_loss_construct

class BayesianUResNet(MENetworkBase):

    def __init__(self, cfg, name='bayesian_uresnet'):
        super(BayesianUResNet, self).__init__(cfg)
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