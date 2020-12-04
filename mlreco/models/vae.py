import numpy as np
import torch
import torch.nn as nn

import MinkowskiEngine as ME
import MinkowskiFunctional as MF

from mlreco.mink.layers.sparse_generator import SparseGenerator
from mlreco.mink.layers.cnn_encoder import SparseResidualEncoder2
from collections import defaultdict


class VAE(nn.Module):

    def __init__(self, cfg, name='vae'):
        super(VAE, self).__init__()
        self.model_config = cfg[name]
        self.coordConv = self.model_config.get('coordConv', False)
        self.encoder = SparseResidualEncoder2(cfg)
        latent_size = self.encoder.latent_size
        self.mean = ME.MinkowskiLinear(latent_size, latent_size)
        self.log_var = ME.MinkowskiLinear(latent_size, latent_size)
        self.decoder = SparseGenerator(cfg)


    def forward(self, input):
        out = defaultdict(list)
        for igpu, x in enumerate(input):
            coords, features = x[:, :4], x[:, 4:]
            print(coords)
            if self.coordConv:
                normalized_coords = (coords[:, 1:4] - float(self.encoder.spatial_size) / 2) \
                        / (float(self.encoder.spatial_size) / 2)
                features = torch.cat([normalized_coords, features], dim=1)
            x = ME.SparseTensor(coords=coords, feats=features, allow_duplicate_coords=True)
            target_key = x.coords_key
            latent = self.encoder(x)
            mean = self.mean(latent)
            log_var = self.log_var(latent)
            z = mean + torch.exp(0.5 * log_var.F) * torch.randn_like(log_var.F)
            out_train = self.decoder(z, target_key)
            out['out_cls'].append(out_train['out_cls'])
            out['targets'].append(out_train['targets'])
            out['mean'].append(mean)
            out['log_var'].append(log_var)

        return out



class ReconstructionLoss(nn.Module):

    def __init__(self, cfg, name='vae_loss'):
        super(ReconstructionLoss, self).__init__()
        self.cross_entropy = nn.functional.binary_cross_entropy_with_logits
        self.loss_config = cfg[name]
        self.bce_weight = self.loss_config.get('bce_weight', 1.0)
        self.kld_weight = self.loss_config.get('kld_weight', 0.0)
        self.layer = self.loss_config.get('layer', -1)

    def forward(self, out, label, weight=None):
        '''
        segmentation[0], label and weight are lists of size #gpus = batch_size.
        segmentation has as many elements as UResNet returns.
        label[0] has shape (N, dim + batch_id + 1)
        where N is #pts across minibatch_size events.
        '''
        # TODO Add weighting
        device = label[0].device
        count = 0
        loss = 0
        accuracy = 0
        num_gpus = len(out['out_cls'])
        assert num_gpus == len(label)
        # Loop over GPUS
        for i in range(num_gpus):
            bce, kld = 0.0, 0.0
            out_cls = out['out_cls'][i]
            targets = out['targets'][i]
            mean = out['mean'][i]
            log_var = out['log_var'][i]
            num_layers = len(out_cls)
            if self.layer > 0:
                skip = self.layer
            else:
                skip = num_layers + 1
            acc_i = 0
            count = 0
            for out_cl, target in zip(out_cls, targets):
                if count >= skip:
                    continue
                with torch.no_grad():
                    w = 0.5
                    # w = float(torch.sum(target) + 1.0) / float(target.shape[0] + 1.0)
                    pos_weight = torch.Tensor([1.0 / w]).to(device)
                    acc = float(((out_cl.F > 0).squeeze().detach().cpu() == target).sum()) / float(target.shape[0])
                    print(acc, target.shape[0])
                    acc_i += acc
                curr_loss = self.cross_entropy(out_cl.F.squeeze(),
                                target.type(out_cl.F.dtype).to(device), pos_weight=pos_weight)
                bce += curr_loss / num_layers
                count += 1
            kld = -0.5 * torch.mean(
                torch.mean(1 + log_var.F - mean.F.pow(2) - log_var.F.exp(), 1))
            acc_i = acc_i / num_layers
            print(bce, kld)
            loss += self.bce_weight * bce + self.kld_weight * kld
            accuracy += acc_i / num_gpus

        return {
            'accuracy': accuracy,
            'loss': loss
        }
