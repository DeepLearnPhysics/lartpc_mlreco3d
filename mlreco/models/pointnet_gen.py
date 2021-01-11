import numpy as np
import torch
import torch.nn as nn

import MinkowskiEngine as ME
import MinkowskiFunctional as MF

from mlreco.mink.layers.sparse_generator import PointNetGenerator
from mlreco.mink.layers.cnn_encoder import SparseResidualEncoder2
from collections import defaultdict

from torch_geometric.data import Data, Batch
from chamferdist import ChamferDistance


class VAE(nn.Module):

    def __init__(self, cfg, name='vae'):
        super(VAE, self).__init__()
        self.model_config = cfg[name]
        self.coordConv = self.model_config.get('coordConv', False)
        self.encoder = SparseResidualEncoder2(cfg)
        latent_size = self.encoder.latent_size
        self.latent_size = latent_size
        self.mean = ME.MinkowskiLinear(latent_size, latent_size)
        self.log_var = ME.MinkowskiLinear(latent_size, latent_size)
        self.decoder = PointNetGenerator(cfg)

        self.occ_net = nn.Sequential(
            nn.Softplus(),
            nn.Linear(latent_size, 64),
            nn.Softplus(),
            nn.Linear(64, 64),
            nn.Softplus(),
            nn.Linear(64, 64),
            nn.Softplus(),
            nn.Linear(64, 1)
        )
        
        self.noise_dim = self.model_config.get('noise_dim', 10)


    def forward(self, input):
        out = defaultdict(list)
        device = input[0].device
        for igpu, x in enumerate(input):
            coords, features = x[:, :4], x[:, 4:]
            if self.coordConv:
                normalized_coords = (coords[:, 1:4] - float(self.encoder.spatial_size) / 2) \
                        / (float(self.encoder.spatial_size) / 2)
                features = torch.cat([normalized_coords, features], dim=1)
            x = ME.SparseTensor(coords=coords, feats=features, allow_duplicate_coords=True)
            _, counts = torch.unique(x.C[:, 0], return_counts=True)
            target_key = x.coords_key
            latent = self.encoder(x)
            mean = self.mean(latent)
            log_var = self.log_var(latent)
            z = mean + torch.exp(0.5 * log_var.F) * torch.randn_like(log_var.F)
            occupancy = self.occ_net(z.F)
            if self.training:
                true_occupancy = torch.log(counts.float())
                points = [Data(x=torch.randn(c, self.noise_dim)).to(device) for c in counts.int()]
                points = Batch().from_data_list(points)
                out['occupancy'].append(occupancy)
                out['points'].append(points.x)
                out['batch'].append(points.batch)
            else:
                out['occupancy'].append(torch.log(counts.float()))
                # counts = torch.exp(occupancy).int()
                points = [Data(x=torch.randn(c, self.noise_dim)).to(device) for c in counts.int()]
                points = Batch().from_data_list(points)
                print(points)
                out['occupancy'].append(occupancy)
                out['points'].append(points.x)
                out['batch'].append(points.batch)
            out['latent'].append(mean)
            points_gen = self.decoder(points.x, mean.F, points.batch)
            out['points_gen'].append(points_gen)

        return out


class ReconstructionLoss(nn.Module):

    def __init__(self, cfg, name='vae_loss'):
        super(ReconstructionLoss, self).__init__()
        self.cross_entropy = nn.functional.binary_cross_entropy_with_logits
        self.loss_config = cfg[name]
        self.bce_weight = self.loss_config.get('bce_weight', 1.0)
        self.kld_weight = self.loss_config.get('kld_weight', 0.0)
        self.layer = self.loss_config.get('layer', -1)
        self.loss_fn = ChamferDistance()
        self.spatial_size = self.loss_config.get('spatial_size', 256)

    def forward(self, out, label, weight=None):
        '''
        segmentation[0], label and weight are lists of size #gpus = batch_size.
        segmentation has as many elements as UResNet returns.
        label[0] has shape (N, dim + batch_id + 1)
        where N is #pts across minibatch_size events.
        '''
        # TODO Add weighting
        device = label[0].device
        loss = 0
        accuracy = 0
        count = 0
        num_gpus = len(out['latent'])
        # print(num_gpus)
        assert num_gpus == len(label)
        # Loop over GPUS
        for i in range(num_gpus):
            points_gen = out['points_gen'][i]
            batch_gen = out['batch'][i]
            occupancy = out['occupancy'][i]
            batch_gt = label[i][:, 0].int()
            print(batch_gen)
            for bidx in batch_gen.unique():
                batch_index = batch_gen == bidx
                points = points_gen[batch_index]
                points_gt = label[i][:, 1:4][batch_gt == bidx]
                points_gt = (points_gt - float(self.spatial_size) / 2) \
                        / (float(self.spatial_size) / 2)
                loss += self.loss_fn(points.view(1, -1, 3), points_gt.view(1, -1, 3))
                count += 1

        return {
            'accuracy': accuracy,
            'loss': loss / count
        }
