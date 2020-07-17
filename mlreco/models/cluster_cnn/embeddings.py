import torch
import torch.nn as nn
import numpy as np
import sparseconvnet as scn
from collections import defaultdict

from mlreco.models.layers.base import SCNNetworkBase
from .utils import add_normalized_coordinates, distance_matrix


class ClusterEmbeddings(SCNNetworkBase):

    def __init__(self, cfg, backbone, name='embeddings'):
        '''
        Multi-Scale Clustering Network for Particle Instance Segmentation.

        ClusterUNet adds N convolution layers to each of the decoding path
        feature tensor of UResNet and outputs an embedding for each spatial
        resolution. 
        
        This module serves as a base architecture for clustering on 
        multiple spatial resolutions.

        INPUTS:
            - backbone (torch.nn.Module): backbone network. ClusterEmbeddings
            module is not standalone and it simply adds N-convolution layers
            and upsampling layers built on top of the feature planes returned
            by the backbone net.

        Configuration
        -------------
        N: int, optional
            Performs N convolution operation at each layer 
            to transform from segmentation features to clustering features
        coordConv: bool, optional
            If True, network concatenates normalized coordinates (between -1 and 1) to 
            the feature tensor at every spatial resolution in the decoding path.
        simpleN: bool, optional
            Uses ResNet blocks by default. If True, N-convolution blocks 
            are replaced with simple BN + SubMfdConv layers.
        embedding_dim: int, optional
            Dimension of clustering hyperspace.
        '''
        super(ClusterEmbeddings, self).__init__(cfg, name='network_base')

        if 'modules' in cfg:
            self.model_config = cfg['modules'][name]
        else:
            self.model_config = cfg

        self.num_classes = self.model_config.get('num_classes', 5)
        self.N = self.model_config.get('N', 1)
        self.simpleN = self.model_config.get('simple_conv', False)
        self.embedding_dim = self.model_config.get('embedding_dim', 16)
        self.coordConv = self.model_config.get('coordConv', False)
        self.kernel_size = self.model_config.get('kernel_size', 2)
        self.downsample = [self.kernel_size, 2]
        self.num_filters = self.model_config.get('num_filters', 16)
        self.num_strides = self.model_config.get('num_strides', 5)
        self.nPlanes = [self.num_filters * (i+1) for i in range(self.num_strides)]

        if self.simpleN:
            clusterBlock = self._block
        else:
            clusterBlock = self._resnet_block

        # Backbone Network Producing Feature Planes at each resolution.
        self.net = backbone

        # Intermediate transformations and clustering upsamplings
        self.cluster_conv = scn.Sequential()
        self.cluster_transform = scn.Sequential()
        clusterPlanes = self.nPlanes[::-1]
        for i, fDim in enumerate(clusterPlanes):
            if i < len(clusterPlanes)-1:
                m = scn.Sequential().add(
                    scn.BatchNormLeakyReLU(fDim, leakiness=self.leakiness)).add(
                    scn.Deconvolution(self.dimension, fDim, clusterPlanes[i+1],
                        self.downsample[0], self.downsample[1], self.allow_bias))
                self.cluster_conv.add(m)
            m = scn.Sequential()
            for j in range(self.N):
                num_input = fDim
                if i > 0 and j == 0:
                    num_input *= 2
                if self.coordConv and j == 0:
                    num_input += self.dimension
                clusterBlock(m, num_input, fDim)
            self.cluster_transform.add(m)


    def encoder(self, x):
        return self.net.encoder(x)


    def decoder(self, features_enc, deepest_layer):
        '''
        ClusterUNet Decoder

        INPUTS:
            - features_enc (list of scn.SparseConvNetTensor): output of encoder.

        RETURNS:
            - features_dec (list of scn.SparseConvNetTensor): 
            list of feature tensors in decoding path at each spatial resolution.
            - features_cluster (list of scn.SparseConvNetTensor): 
            list of transformed features on which we apply clustering loss 
            at every spatial resolution.
        '''
        features_dec = []
        cluster_feature = []
        x_seg = deepest_layer
        x_emb = deepest_layer
        for i, layer in enumerate(self.net.decoding_conv):
            encoder_feature = features_enc[-i-2]
            if self.coordConv:
                x_emb = add_normalized_coordinates(x_emb)
            x_emb = self.cluster_transform[i](x_emb)
            cluster_feature.append(x_emb)
            x_emb = self.cluster_conv[i](x_emb)
            x_seg = layer(x_seg)
            x_seg = self.net.concat([encoder_feature, x_seg])
            x_seg = self.net.decoding_block[i](x_seg)
            features_dec.append(x_seg)
            x_emb = self.net.concat([x_emb, x_seg])
        # Compensate for last clustering convolution
        if self.coordConv:
            x_emb = add_normalized_coordinates(x_emb)
        x_emb = self.cluster_transform[-1](x_emb)
        cluster_feature.append(x_emb)

        result = {
            "features_dec": features_dec,
            "cluster_feature": cluster_feature
        }

        return result


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

        x = self.net.input((coords, features))
        encoder_output = self.encoder(x)
        decoder_output = self.decoder(encoder_output['features_enc'],
                                encoder_output['deepest_layer'])
        
        res['features_dec'] = [decoder_output['features_dec']]
        # Reverse cluster feature tensor list to agree with label ordering.
        res['cluster_feature'] = [decoder_output['cluster_feature'][::-1]]

        return res


class ClusterEmbeddingsFPN(ClusterEmbeddings):

    def __init__(self, cfg, backbone, name='embeddings'):
        super(ClusterEmbeddingsFPN, self).__init__(cfg, backbone, name=name)

    def decoder(self, features_enc, deepest_layer):
        '''
        ClusterUNet Decoder for FPN Type Backbones

        INPUTS:
            - features_enc (list of scn.SparseConvNetTensor): output of encoder.

        RETURNS:
            - features_dec (list of scn.SparseConvNetTensor): 
            list of feature tensors in decoding path at each spatial resolution.
            - features_cluster (list of scn.SparseConvNetTensor): 
            list of transformed features on which we apply clustering loss 
            at every spatial resolution.
        '''
        features_dec = []
        cluster_feature = []
        x_seg = deepest_layer
        x_emb = deepest_layer
        for i, layer in enumerate(self.net.decoding_conv):
            encoder_feature = features_enc[-i-2]
            if self.coordConv:
                x_emb = add_normalized_coordinates(x_emb)
            x_emb = self.cluster_transform[i](x_emb)
            cluster_feature.append(x_emb)
            x_emb = self.cluster_conv[i](x_emb)
            x_seg = layer(x_seg)
            x_seg = self.net.add([encoder_feature, x_seg])
            x_seg = self.net.decoding_block[i](x_seg)
            features_dec.append(x_seg)
            x_emb = self.net.concat([x_emb, x_seg])
        # Compensate for last clustering convolution
        if self.coordConv:
            x_emb = add_normalized_coordinates(x_emb)
        x_emb = self.cluster_transform[-1](x_emb)
        cluster_feature.append(x_emb)

        result = {
            "features_dec": features_dec,
            "cluster_feature": cluster_feature
        }

        return result



class StackedEmbeddings(ClusterEmbeddings):


    def __init__(self, cfg, backbone, name='embeddings'):
        super(StackedEmbeddings, self).__init__(cfg, backbone, name=name)
        if 'modules' in cfg:
            self.model_config = cfg['modules'][name]
        else:
            self.model_config = cfg
        
        # Define Backbone Network 
        self.net = backbone
        if self.simpleN:
            clusterBlock = self._block
        else:
            clusterBlock = self._resnet_block

        # StackNet Model Parameters
        self.reduce_feature = self.model_config.get('reduce_feature', 'nin')
        self.sum_features = sum(self.nPlanes)
        clusterPlanes = self.nPlanes[::-1]

        # N Convolution (Intermediate Transforms)
        self.cluster_transform = scn.Sequential()
        for i, fDim in enumerate(clusterPlanes):
            m = scn.Sequential()
            for j in range(self.N):
                num_input = fDim
                if self.coordConv and j == 0:
                    num_input += self.dimension
                clusterBlock(m, num_input, fDim)
            self.cluster_transform.add(m)

        # UnPooling Layers
        self.learnable_upsampling = self.model_config.get('learnable_upsampling', False)
        self.unpooling = scn.Sequential()

        if self.learnable_upsampling:
            # Using transpose convolution to upsample feature maps.
            for i in range(self.num_strides-1):
                m = scn.Sequential()
                for j in range(self.num_strides-2-i, -1, -1):
                    m.add(
                        scn.BatchNormLeakyReLU(self.nPlanes[j+1], leakiness=self.leakiness)).add(
                        scn.Deconvolution(self.dimension, self.nPlanes[j+1], self.nPlanes[j],
                            self.downsample[0], self.downsample[1], self.allow_bias))
                    self._resnet_block(m, self.nPlanes[j], self.nPlanes[j])
                self.unpooling.add(m)
            self.stackPlanes = self.nPlanes[::-1]
        else:
            # Using normal unpooling layers to upsample feature maps.
            for i in range(self.num_strides-1):
                m = scn.Sequential()
                for _ in range(self.num_strides-1-i):
                    m.add(
                        scn.UnPooling(self.dimension, self.downsample[0], self.downsample[1]))
                self.unpooling.add(m)
            self.stackPlanes = [self.sum_features, 100, 50, self.num_filters]

        self.reduction_layers = scn.Sequential()
        if self.reduce_feature == 'resnet':
            reduceBlock = self._resnet_block
        elif self.reduce_feature == 'conv':
            reduceBlock = self._block
        elif self.reduce_feature == 'nin':
            reduceBlock = self._nin_block
        else:
            raise ValueError('Invalid option for StackNet feature reducing layers.')

        # Feature Reducing Layers
        self.cluster_decoder = scn.Sequential()
        for i in range(len(self.stackPlanes)-1):
            m = scn.Sequential()
            reduceBlock(m, self.stackPlanes[i], self.stackPlanes[i+1])
            self.cluster_decoder.add(m)

        self.concat = scn.JoinTable()
        self.last_embedding = scn.Sequential()
        clusterBlock(self.last_embedding, self.num_filters * 2, self.embedding_dim)
    
    def encoder(self, x):
        return self.net.encoder(x)

    def decoder(self, features_enc, deepest_layer):
        '''
        ClusterUNet Decoder

        INPUTS:
            - features_enc (list of scn.SparseConvNetTensor): output of encoder.

        RETURNS:
            - features_dec (list of scn.SparseConvNetTensor): 
            list of feature tensors in decoding path at each spatial resolution.
            - features_cluster (list of scn.SparseConvNetTensor): 
            list of transformed features on which we apply clustering loss 
            at every spatial resolution.
        '''
        features_dec = [deepest_layer]
        cluster_feature = []
        x_seg = deepest_layer
        x_emb = deepest_layer
        for i, layer in enumerate(self.net.decoding_conv):
            encoder_feature = features_enc[-i-2]
            if self.coordConv:
                x_emb = add_normalized_coordinates(x_emb)
            x_emb = self.cluster_transform[i](x_emb)
            cluster_feature.append(x_emb)
            x_seg = layer(x_seg)
            x_seg = self.net.concat([encoder_feature, x_seg])
            x_seg = self.net.decoding_block[i](x_seg)
            features_dec.append(x_seg)
            x_emb = self.cluster_conv[i](x_emb)
        # Compensate for last clustering convolution
        if self.coordConv:
            x_emb = add_normalized_coordinates(x_emb)
        x_emb = self.cluster_transform[-1](x_emb)
        cluster_feature.append(x_emb)

        result = {
            "features_dec": features_dec,
            "cluster_feature": cluster_feature
        }
        return result
    
    def forward(self, input):
        point_cloud, = input
        coords = point_cloud[:, 0:self.dimension+1].float()
        features = point_cloud[:, self.dimension+1:].float()

        x = self.net.input((coords, features))
        encoder_output = self.encoder(x)
        features_enc = encoder_output['features_enc']
        deepest_layer = encoder_output['deepest_layer']
        decoder_output = self.decoder(features_enc, deepest_layer)

        cluster_feature = decoder_output['cluster_feature']
        features_dec = decoder_output['features_dec']
        stack_feature = []

        for i, layer in enumerate(cluster_feature):
            if i < self.num_strides-1:
                f = self.unpooling[i](layer)
                stack_feature.append(f)
            else:
                stack_feature.append(layer)

        stack_feature = self.concat(stack_feature)
        out = self.cluster_decoder(stack_feature)
        out = self.concat([out, cluster_feature[-1]])
        out = self.last_embedding(out)
        cluster_feature[-1] = out

        res = {
            'features_dec': [features_dec],
            'cluster_feature': [cluster_feature[::-1]]
        }

        return res


class GCNNEmbeddings(ClusterEmbeddings):

    def __init__(self, cfg, name='gcnn_embeddings'):
        super(GCNNEmbeddings, self).__init__(self, cfg, name=name)
        self.k = self.model_config.get('num_neighbors', 10)
        self.gcnn_layers = self.model_config.get('gcnn_layers', 2)
        
    def get_nn_map(self, embedding_class, cluster_class):
        """
        Computes voxel team loss.

        INPUTS:
            (torch.Tensor)
            - embedding_class: class-masked hyperspace embedding
            - cluster_class: class-masked cluster labels

        RETURNS:
            - loss (torch.Tensor): scalar tensor representing aggregated loss.
            - dlossF (dict of floats): dictionary of ally loss.
            - dlossE (dict of floats): dictionary of enemy loss.
            - dloss_map (torch.Tensor): computed ally/enemy affinity for each voxel. 
        """
        with torch.no_grad():
            knnFeatures = torch.zeros(embedding_class.shape[0], 
                                      self.k, embedding_class.shape[1])
            allyMap = torch.zeros(embedding_class.shape[0])
            enemyMap = torch.zeros(embedding_class.shape[0])
            if torch.cuda.is_available():
                allyMap = allyMap.cuda()
                enemyMap = enemyMap.cuda() 
            dist = distance_matrix(embedding_class)
            cluster_ids = cluster_class.unique().int()
            num_clusters = float(cluster_ids.shape[0])
            for c in cluster_ids:
                index = cluster_class.int() == c
                allies = dist[index, :][:, index]
                num_allies = allies.shape[0]
                if num_allies <= 1:
                    # Skip if only one point
                    continue
                ind = np.diag_indices(num_allies)
                allies[ind[0], ind[1]] = float('inf')
                allies, _ = torch.min(allies, dim=1)
                allyMap[index] = allies
                if index.all(): 
                    # Skip if there are no enemies
                    continue
                enemies, _ = torch.min(dist[index, :][:, ~index], dim=1)
                enemyMap[index] = enemies

            nnMap = torch.cat([allyMap.view(-1, 1), enemyMap.view(-1, 1)], dim=1)
            topk = min(self.k+1, embedding_class.shape[0])
            _, index = dist.topk(topk, largest=False)
            nnFeatures = embedding_class[index][:, 1:, :]
            knnFeatures[:, :topk] = nnFeatures

            return nnMap, knnFeatures

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

        x = self.net.input((coords, features))
        encoder_output = self.encoder(x)
        decoder_output = self.decoder(encoder_output['features_enc'],
                                encoder_output['deepest_layer'])
        
        res['features_dec'] = [decoder_output['features_dec']]
        # Reverse cluster feature tensor list to agree with label ordering.
        res['cluster_feature'] = [decoder_output['cluster_feature'][::-1]]

        return res