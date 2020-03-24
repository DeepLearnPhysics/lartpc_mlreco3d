# Autoencoder trainer
# Use layer/full_encoder.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import numpy as np
from .layers.full_encoder import EncoderLayer
from mlreco.utils.gnn.cluster import form_clusters, get_cluster_batch
from mlreco.utils.gnn.network import complete_graph
from mlreco.utils.groups import merge_batch
import random


class AutoEncoder(torch.nn.Module):
    '''
    Driver class for autoencoder training

    For use in config:
    model:
      name: cluster_gnn
      modules:
        chain:
          mode: 'node' or 'edge'
        autoencoder:
          <dictionary of arguments to pass to the encoder>
          model_path      : <path to the encoder weights>
    '''
    def __init__(self, cfg):
        super(AutoEncoder, self).__init__()

        # Get the chain input parameters
        chain_config = cfg['chain']

        # What type of training
        # 'node' mode give images with distinct ids
        # 'edge' mode give images with two distinct ids
        self._mode = chain_config.get('mode', 'node')

        # Pick up some to perform trainning
        self.num_to_pick = chain_config.get('num_to_pick', -1) # <0 is using all

        # construct autoencoder
        autoencoder_config = cfg['autoencoder']
        self.autoencoder = EncoderLayer(autoencoder_config)

    def forward(self, data):
        """
        Args:
            data ([torch.tensor]): (N, 6) [x, y, z, batchid, (value,) id]
        Returns:
            dict:
                'result': scalar difference between image and autoencoded images
        """
        data = data[0]
        device = data.device

        # prepare for edge index
        # by default using complete graphes
        clusts = form_clusters(data, -1)
        clusts = [c.cpu().numpy() for c in clusts]
        # Get the batch id for each cluster
        batch_ids = get_cluster_batch(data, clusts)

        images = torch.empty((0,5), dtype=torch.float, device=device)
        if self._mode=='node':
            if self.num_to_pick>0:
                random.shuffle(clusts)
            for i, c in enumerate(clusts):
                if self.num_to_pick>0 and i>=self.num_to_pick:
                    break
                images = torch.cat((images, data[c,:5].float()))
                images[-len(c):,3] = i*torch.ones(len(c)).to(device)
        elif self._mode=='edge':
            # for getting edge index
            edge_index = complete_graph(batch_ids, None, -1).T
            #
            if self.num_to_pick>0:
                edge_index = random.shuffle(edge_index)
            # Go through each edge
            for i, (ind1, ind2) in enumerate(edge_index):
                if self.num_to_pick>0 and i>=self.num_to_pick:
                    break
                c1 = clusts[ind1]
                c2 = clusts[ind2]
                images = torch.cat((images, data[c1,:5].float(), data[c2,:5].float()))
                images[-len(c1)-len(c2):,3] = i*torch.ones(len(c1)+len(c2)).to(device)
        else:
            raise ValueError('Auto-encoder mode not supported!')

        res, _ = self.autoencoder(images)
        return {'result': [res]}


class AutoEncoderLoss(torch.nn.Module):
    '''
    Auto-encoder loss
    using MSE loss
    Mostly borrowed from:
    https://github.com/Picodes/lartpc_mlreco3d/commit/1a940c1ad315d9ac27f123b78f6f71146429ae71#diff-c791653078abc5216cee4258c2a5b96f
    '''
    def __init__(self, cfg):
        super(AutoEncoderLoss, self).__init__()
        self.loss = torch.nn.MSELoss()

    def forward(self, out, clusters):

        loss = self.loss(out['result'][0], torch.tensor([0]).float().to(out['result'][0].device))

        return {
            'accuracy': loss.item(),
            'loss': loss
        }