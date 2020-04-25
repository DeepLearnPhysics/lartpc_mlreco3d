import torch
import torch.nn as nn
import numpy as np
import sparseconvnet as scn

from .cluster_cnn.losses.multi_layers import DensityDistanceEstimationLoss, DensityLoss
from .cluster_cnn.clusternet import ClusterUNet

class ClusterCNN(ClusterUNet):
    '''
    UResNet with multi-scale convolution blocks for clustering at
    each spatial resolution. In the last clustering feature layer,
    we add a distance estimation branch.
    '''
    def __init__(self, cfg, name='clusterunet_density'):
        super(ClusterCNN, self).__init__(cfg, name='clusterunet')
        self.model_config = cfg[name]
        self.N_dist = self.model_config.get('N_dist', 3)
        self.distance_blocks = self.model_config.get('block', 'conv')
        feature_size = self.model_config.get('feature_size', 16)

        if self.distance_blocks == 'resnet':
            distanceBlock = self._resnet_block
        elif self.distance_blocks == 'conv':
            distanceBlock = self._block
        else:
            raise ValueError('Invalid convolution block mode.')

        self.distance_conv = scn.Sequential()
        distanceBlock(self.distance_conv, 2 * feature_size, feature_size)
        self.distance_branch = scn.Sequential()
        for i in range(self.N_dist):
            m = scn.Sequential()
            distanceBlock(m, feature_size, feature_size)
            self.distance_branch.add(m)

        # Final 1x1 Convolution to Distance Estimation Map
        self._nin_block(self.distance_branch, feature_size, 2)

    def forward(self, input):
        '''
        point_cloud is a list of length minibatch size (assumes mbs = 1)
        point_cloud[0] has 3 spatial coordinates + 1 batch coordinate + 1 feature
        label has shape (point_cloud.shape[0] + 5*num_labels, 1)
        label contains segmentation labels for each point + coords of gt points

        RETURNS:
            - feature_dec: decoder features at each spatial resolution.
            - cluster_feature: clustering features at each spatial resolution.
        '''
        point_cloud, = input
        coords = point_cloud[:, 0:self.dimension+1].float()
        features = point_cloud[:, self.dimension+1:].float()
        res = {}

        x = self.input((coords, features))
        encoder_output = self.encoder(x)
        decoder_output = self.decoder(encoder_output['features_enc'],
                                encoder_output['deepest_layer'])

        res['features_dec'] = [decoder_output['features_dec']]
        # Reverse cluster feature tensor list to agree with label ordering.
        res['cluster_feature'] = [decoder_output['cluster_feature'][::-1]]
        res['final_embedding'] = [decoder_output['final_embedding']]

        distance_input = self.concat([decoder_output['features_dec'][-1], decoder_output['cluster_feature'][-1]])
        distance_input = self.distance_conv(distance_input)

        distance_estimation = self.distance_branch(distance_input)
        res['distance_estimation'] = [distance_estimation]

        return res


class ClusteringLoss(nn.Module):
    '''
    Loss for attention-weighted and multi-scale clustering loss.
    '''
    def __init__(self, cfg, name='clustering_loss'):
        super(ClusteringLoss, self).__init__()
        self.model_config = cfg['modules'][name]

        # TODO: Define single model with configurable enhancements.
        self.loss_func = DensityDistanceEstimationLoss(cfg)

    def forward(self, out, segment_label, cluster_label):
        result = self.loss_func(out, segment_label, cluster_label)
        return result
