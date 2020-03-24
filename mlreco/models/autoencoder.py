# Autoencoder trainer
# Use layer/full_encoder.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import numpy as np
from layers.full_encoder import EncoderLayer
from mlreco.utils.gnn.cluster import form_clusters, get_cluster_batch
from mlreco.utils.gnn.network import complete_graph
from mlreco.utils.groups import merge_batch


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

        if self._mode=='node':
            # Reuse the batch merging
            num_batches = data[:3].unique().view(-1).size()[0]
            images, _ = merge_batch(data, num_batches, whether_fluctuate=False)
            # change image ids
            images[:,3] = images[:,5]
            res = self.autoencoder(images)
        elif self._mode=='edge':
            res = 0
            # prepare for edge index
            # by default using complete graphes
            clusts = form_clusters(data, self.node_min_size)
            clusts = [c.cpu().numpy() for c in clusts]
            # Get the batch id for each cluster
            batch_ids = get_cluster_batch(data, clusts)
            # for getting edge index
            edge_index = complete_graph(batch_ids, None, -1)
            # Go through each edge
            for ind1, ind2 in edge_index:
                images = torch.cat([data[clusts[ind1],:5], data[clusts[ind2], :5]])
                r, _ = self.autoencoder(images)
                res += r
        else:
            raise ValueError('Auto-encoder mode not supported!')

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
        loss = self.loss(out['result'][0], torch.tensor([0]).float())

        return {
            'accuracy': loss.item(),
            'loss': loss
        }