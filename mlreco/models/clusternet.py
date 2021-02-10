import torch
import torch.nn as nn
import numpy as np
import sparseconvnet as scn

from .cluster_cnn import cluster_model_construct, backbone_construct, spice_loss_construct
from mlreco.models.layers.base import SCNNetworkBase

###########################################################
#
# Define one multilayer model to incorporate all options.
#
# Embedding Transforming Convolutions are added on top of
# backbone decoder features.
#
# Distance Estimation Map is added on top of final layer of
# backbone decoder concatenated with final layer of clustering.
#
###########################################################

class ClusterCNN(SCNNetworkBase):
    '''
    CNN Based Multiscale Clustering Module.

    Configurations
    ----------------------------------------------------------
    backbone: the type of backbone architecture to be used in the
    feature extraction stages. Currently the following options are
    available:
        - UResNet: Vanilla UResNet
        - FPN: Feature Pyramid Networks

    clustering: configurations for clustering transformation convolutions
    at each decoding path.

    proximity: configurations for final distance estimation map.
    ----------------------------------------------------------
    '''

    MODULES = ['network_base', 'clusternet', 'uresnet', 'spice_loss']

    def __init__(self, cfg, name='clusternet'):
        super(ClusterCNN, self).__init__(cfg)

        self.model_config = cfg[name]

        self.backbone_config = self.model_config.get('backbone', {})
        self.clustering_config = self.model_config.get('clustering', {})
        self.compute_distance_estimate = self.clustering_config.get('compute_distance_estimate', False)

        # Construct Backbone
        backbone_name = self.backbone_config.get('name', 'uresnet')
        net = backbone_construct(backbone_name)
        self.net = net(cfg, name=backbone_name)
        self.num_filters = self.net.num_filters

        # Add N-Convolutions for Clustering
        if self.clustering_config is not None:
            self.clustering_name = self.clustering_config.get('name', 'multi')
            clusternet = cluster_model_construct(self.clustering_name)
            self.net = clusternet(cfg, self.net, name='embeddings')

        # For final embeddings:
        self.final_embeddings = scn.Sequential()
        self.embedding_dim = self.clustering_config.get('embedding_dim', 8)
        self._resnet_block(self.final_embeddings, 2 * self.num_filters, self.embedding_dim)

        self.freeze_embeddings = self.clustering_config.get('freeze_embeddings', False)
        if self.freeze_embeddings:
            print("Embedding Generator Network Freezed.")
            for param in self.parameters():
                param.requires_grad = False

        # Add Distance Estimation Layer
        if self.compute_distance_estimate:
            self.dist_N = self.clustering_config.get('dist_N', 3)
            self.dist_simple_conv = self.clustering_config.get('dist_simple_conv', False)
            self.distance_estimate = scn.Sequential()
            if self.dist_simple_conv:
                distanceBlock = self._block
            else:
                distanceBlock = self._resnet_block
            for i in range(self.dist_N):
                num_input = self.num_filters
                num_output = self.num_filters
                if i == 0:
                    num_input = 2 * self.num_filters
                if i == self.dist_N-1:
                    num_output = 2
                distanceBlock(self.distance_estimate, num_input, num_output)

        self.concat = scn.JoinTable()


    def forward(self, input):
        '''
        Forward function for whole ClusterNet Chain.
        '''
        result = self.net(input)
        # for key, val in result.items():
        #     if isinstance(val[0], list):
        #         l = [str(t.features.shape) for t in val[0]]
        #         s = ' '.join(l)
        #         print('{} = {}'.format(key, s))
        #     else:
        #         print("{} = {}".format(key, val[0].features.shape))
        final_cluster_features = result['cluster_feature'][0][0]
        final_decoder_features = result['features_dec'][0][-1]

        x = self.concat([final_decoder_features, final_cluster_features])
        x_emb = self.final_embeddings(x)
        result['cluster_feature'][0][0] = x_emb
        if self.compute_distance_estimate:
            result['distance_estimation'] = [self.distance_estimate(x)]
        return result


class ClusteringLoss(nn.Module):
    '''
    Loss Function for CNN Based Cascading Clustering.

    Configurations
    ----------------------------------------------------------
    loss_segmentation:
        - configurations for semantic segmentation loss.

    loss_clustering:
        - configurations for clustering loss.

    loss_distance:
        - configurations for distance estimation loss.
    ----------------------------------------------------------
    '''
    def __init__(self, cfg, name='spice_loss'):
        super(ClusteringLoss, self).__init__()

        self.loss_config = cfg[name]

        self.loss_func_name = self.loss_config.get('name', 'multi')
        self.loss_func = spice_loss_construct(self.loss_func_name)
        self.loss_func = self.loss_func(cfg)
        print(self.loss_func)

    def forward(self, result, segment_label, cluster_label):
        '''
        Forward Function for clustering loss , with distance estimation.
        '''
        # for key, val in result.items():
        #     if isinstance(val[0], list):
        #         l = [str(t.features.shape) for t in val[0]]
        #         s = ' '.join(l)
        #         print('{} = {}'.format(key, s))
        #     else:
        #         print("{} = {}".format(key, val[0].features.shape))
        res = self.loss_func(result, segment_label, cluster_label)
        return res
