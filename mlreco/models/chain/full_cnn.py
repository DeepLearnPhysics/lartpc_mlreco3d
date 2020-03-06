import torch
import torch.nn as nn
import numpy as np
import sparseconvnet as scn
from collections import defaultdict

from mlreco.models.layers.uresnet import UResNetEncoder, UResNetDecoder
from mlreco.models.layers.base import NetworkBase

from mlreco.models.ppn import PPN

# from mlreco.models.cluster_cnn.utils import *
# from mlreco.models.cluster_cnn.losses.spatial_embeddings import *
# from mlreco.models.cluster_cnn import cluster_model_construct, backbone_construct, clustering_loss_construct

# from mlreco.models.gnn.edge_nnconv import *
# from mlreco.models.gnn.factories import *

from torch_geometric.data import Batch as GraphData

# ----------------------HELPER FUNCTIONS------------------------


def gaussian_kernel(centroid, sigma):
    def f(x):
        dists = np.sum(np.power(x - centroid, 2), axis=1, keepdims=False)
        probs = np.exp(-dists / (2.0 * sigma**2))
        return probs
    return f


def multivariate_kernel(centroid, L):
    def f(x):
        N = x.shape[0]
        cov = torch.zeros(3, 3)
        tril_indices = torch.tril_indices(row=3, col=3, offset=0)
        cov[tril_indices[0], tril_indices[1]] = L
        cov = torch.matmul(cov, cov.T)
        dist = torch.matmul((x - centroid), cov)
        dist = torch.bmm(dist.view(N, 1, -1), (x-centroid).view(N, -1, 1)).squeeze()
        probs = torch.exp(-dist)
        return probs
    return f


def get_edge_features(nodes, batch_idx, edge_net):
    '''
    Compile Fully Connected Edge Features from nodes and batch indices.

    INPUTS:
        - nodes (N x d Tensor): list of node features
        - batch_idx (N x 1 Tensor): list of batch indices for nodes
        - bilinear_net: nn.Module that taks two vectors and returns edge feature vector. 

    RETURNS:
        - edge_features: list of edges features
        - edge_indices: list of edge indices (i->j)
        - edge_batch_indices: list of batch indices (0 to B)
    '''
    unique_batch = batch_idx.unique()
    edge_indices = []
    edge_features = []
    edge_batch_indices = []
    for bidx in unique_batch:
        mask = bidx == batch_idx
        nodes_batch = nodes[mask]
        subindex = torch.arange(nodes_batch.shape[0])
        N = nodes_batch.shape[0]
        for i, row in enumerate(nodes_batch):
            submask = subindex != i
            edge_idx = [torch.Tensor([i, j]).cuda() for j in subindex[submask]]
            edge_indices.extend(edge_idx)
            others = nodes_batch[submask]
            ei2j = edge_net(row.expand_as(others), others)
            edge_features.extend(ei2j)
            edge_batch_indices.extend([bidx for _ in subindex[submask]])
    
    edge_indices = torch.stack(edge_indices, dim=0)
    edge_features = torch.stack(edge_features, dim=0)
    edge_batch_indices = torch.stack(edge_batch_indices, dim=0)

    return edge_indices, edge_features, edge_batch_indices


class EdgeFeatureNet(nn.Module):
    '''
    Small MLP for extracting input edge features from two node features.

    USAGE:
        net = EdgeFeatureNet(16, 16)
        node_x = torch.randn(16, 5)
        node_y = torch.randn(16, 5)
        edge_feature_x2y = net(node_x, node_y) # (16, 5)
    '''
    def __init__(self, num_input, num_output):
        super(EdgeFeatureNet, self).__init__()
        self.linear1 = nn.Linear(num_input * 2, 16)
        # self.norm1 = nn.BatchNorm1d(16)
        self.linear2 = nn.Linear(16, 16)
        # self.norm2 = nn.BatchNorm1d(16)
        self.linear3 = nn.Linear(16, num_output)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.linear1(x)
        # x = self.norm1(x)
        x = self.linear2(x)
        # x = self.norm2(x)
        x = self.linear3(x)
        return x


def get_gnn_input(coords, features_gnn, edge_net, fit_predict, 
                    train=True, **kwargs):
    '''
    Get input features to GNN from CNN clustering output

    INPUTS (TRAINING):

        - featuresAgg (N x F): combined feature tensor per voxel, created
        using semantic/seediness/clustering feature maps. 

        - fragment_labels (N, ): ground truth fragment labels.


    INPUTS (INFERENCE):

        - featuresAgg: same as before
        - embeddings: embedding coordinates from clustering network
        - margins: margin values from CNN clustering
        - seediness: seediness values from CNN clustering
        - kernel_func: choice of probability kernel function to be used
        in CNN clustering inference. 


    RETURNS:

        gnn_result: A dictionary containing input node & edge feature to GNN.
        The items are self-explanatory from their variable names. 
    '''
    nodes_full = []
    nodes_batch_id = []
    batch_index = kwargs['batch_index']

    if train:
        fragment_labels = kwargs['fragment_labels']
        for bidx in batch_index:
            batch_mask = coords[:, 3] == bidx
            featuresAgg_batch = features_gnn[batch_mask]
            fragment_batch = fragment_labels[batch_mask]
            fragment_ids = fragment_batch.unique()
            for fid in fragment_ids:
                mean_features = featuresAgg_batch[fragment_batch == fid].mean(dim=0)
                # print(mean_features)
                nodes_full.append(mean_features)
                nodes_batch_id.append(int(bidx))
    else:
        with torch.no_grad():
            embeddings = kwargs['embeddings']
            margins = kwargs['margins']
            semantic_labels = kwargs['semantic_labels']
            seediness = kwargs['seediness']
            kernel_func = kwargs.get('kernel_func', multivariate_kernel)
            for bidx in batch_index:
                batch_mask = coords[:, 3] == bidx
                slabel = semantic_labels[batch_mask]
                embedding_batch = embeddings[batch_mask]
                featuresAgg_batch = features_gnn[batch_mask]
                margins_batch = margins[batch_mask]
                seediness_batch = seediness[batch_mask]
                for s in slabel.unique():
                    segment_mask = slabel == s
                    featuresAgg_class = featuresAgg_batch[segment_mask]
                    embedding_class = embedding_batch[segment_mask]
                    margins_class = margins_batch[segment_mask]
                    seediness_class = seediness_batch[segment_mask]
                    pred_labels = fit_predict(
                        embedding_class, seediness_class, margins_class, kernel_func)
                    # Adding Node Features
                    for c in pred_labels.unique():
                        mask = pred_labels == c
                        mean_features = featuresAgg_class[mask].mean()
                        nodes_full.append(mean_features)
                        nodes_batch_id.append(bidx)

    nodes_full = torch.stack(nodes_full, dim=0)
    node_batch_id = torch.Tensor(nodes_batch_id).cuda()
    
    # Compile Pairwise Edge Features
    edge_indices, edge_features, edge_batch_indices = get_edge_features(
        nodes_full, node_batch_id, edge_net)

    gnn_input = GraphData(x=nodes_full, 
                          edge_index=edge_indices.view(2, -1).long(), 
                          edge_attr=edge_features,
                          batch=node_batch_id.long())
    gnn_input.edge_batch = edge_batch_indices

    # gnn_input = {
    #     'edge_id': edge_indices,
    #     'edge_features': edge_features,
    #     'edge_batch_id': edge_batch_indices,
    #     'node_features': nodes_full,
    #     'node_batch_id': node_batch_id
    # }
    return gnn_input


def fit_predict(embeddings, seediness, margins, fitfunc, st=0.0, pt=0.5):
    '''
    Inference subroutine for test time behavior of network.
    '''
    pred_labels = torch.zeros(embeddings.shape[0])
    probs = []
    seediness_copy = np.copy(seediness.detach().cpu().numpy())
    count = 0
    while count < seediness.shape[0]:
        i = np.argsort(seediness_copy)[::-1][0]
        seedScore = seediness[i]
        centroid = embeddings[i]
        sigma = margins[i]
        if seedScore < st:
            break
        f = fitfunc(centroid, sigma)
        pValues = f(embeddings)
        probs.append(pValues.reshape(-1, 1))
        cluster_index = pValues > pt
        seediness_copy[cluster_index] = 0
        count += sum(cluster_index)
    if len(probs) == 0:
        return pred_labels
    probs = np.hstack(probs)
    pred_labels = np.argmax(probs, axis=1)
    return pred_labels


# --------------------------CHAINS------------------------------


class FullCNN(NetworkBase):
    '''
    CNN Part of Full Reconstruction Chain for LArTPC Event Reconstruction

    CONFIGURATIONS:
    '''
    def __init__(self, cfg, name='full_cnn'):
        super(FullCNN, self).__init__(cfg, name='network_base')

        self.model_config = cfg[name]
        self.num_filters = self.model_config.get('filters', 16)
        self.ghost = self.model_config.get('ghost', False)
        self.seed_dim = self.model_config.get('seed_dim', 1)
        self.sigma_dim = self.model_config.get('sigma_dim', 6)
        self.embedding_dim = self.model_config.get('embedding_dim', 3)
        self.num_classes = self.model_config.get('num_classes', 5)
        self.num_gnn_features = self.model_config.get('num_gnn_features', 16)
        self.inputKernel = self.model_config.get('input_kernel_size', 3)

        # Network Freezing Options
        self.encoder_freeze = self.model_config.get('encoder_freeze', False)
        self.ppn_freeze = self.model_config.get('ppn_freeze', False)
        self.segmentation_freeze = self.model_config.get('segmentation_freeze', False)
        self.embedding_freeze = self.model_config.get('embedding_freeze', False)
        self.seediness_freeze = self.model_config.get('seediness_freeze', False)

        # Input Layer Configurations and commonly used scn operations. 
        self.input = scn.Sequential().add(
            scn.InputLayer(self.dimension, self.spatial_size, mode=3)).add(
            scn.SubmanifoldConvolution(self.dimension, self.nInputFeatures, \
            self.num_filters, self.inputKernel, self.allow_bias)) # Kernel size 3, no bias
        self.concat = scn.JoinTable()
        self.add = scn.AddTable()

        # Backbone UResNet
        self.encoder = UResNetEncoder(cfg, name='uresnet_encoder')
        self.seg_net = UResNetDecoder(cfg, name='uresnet_decoder')
        self.seed_net = UResNetDecoder(cfg, name='uresnet_decoder')
        self.cluster_net = UResNetDecoder(cfg, name='uresnet_decoder')

        # Output Layers
        self.output_cluster = scn.Sequential()
        self.output_cluster.add(scn.NetworkInNetwork(
            self.num_filters, self.dimension + self.sigma_dim, self.allow_bias))
        self.output_cluster.add(scn.OutputLayer(self.dimension))

        self.output_seediness = scn.Sequential()
        self.output_seediness.add(scn.NetworkInNetwork(
            self.num_filters, self.seed_dim, self.allow_bias))
        self.output_seediness.add(scn.OutputLayer(self.dimension))

        self.output_segmentation = scn.Sequential()
        self.output_segmentation.add(scn.NetworkInNetwork(
            self.num_filters, self.num_classes, self.allow_bias))
        self.output_segmentation.add(scn.OutputLayer(self.dimension))

        self.output_gnn_features = scn.Sequential()
        self.output_gnn_features.add(scn.NetworkInNetwork(
            self.num_filters * 3, self.num_gnn_features, self.allow_bias))
        self.output_gnn_features.add(scn.OutputLayer(self.dimension))

        if self.ghost:
            self.linear_ghost = scn.Sequential()
            self._nin_block(self.linear_ghost, self.num_filters, 2)
            # self.linear_ghost.add(scn.OutputLayer(self.dimension))

        # PPN
        self.ppn  = PPN(cfg)

        # Freeze Layers
        if self.encoder_freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
            print('Encoder Freezed')

        if self.ppn_freeze:
            for p in self.ppn.parameters():
                p.requires_grad = False
            print('PPN Freezed')

        if self.segmentation_freeze:
            for p in self.seg_net.parameters():
                p.requires_grad = False
            for p in self.output_segmentation.parameters():
                p.requires_grad = False
            print('Segmentation Branch Freezed')

        if self.embedding_freeze:
            for p in self.cluster_net.parameters():
                p.requires_grad = False
            for p in self.output_cluster.parameters():
                p.requires_grad = False
            print('Clustering Branch Freezed')

        if self.seediness_freeze:
            for p in self.seed_net.parameters():
                p.requires_grad = False
            for p in self.output_seediness.parameters():
                p.requires_grad = False
            print('Seediness Branch Freezed')

        # Pytorch Activations
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()


    def forward(self, input):
        '''
        point_cloud is a list of length minibatch size (assumes mbs = 1)
        point_cloud[0] has 3 spatial coordinates + 1 batch coordinate + 1 feature
        label has shape (point_cloud.shape[0] + 5*num_labels, 1)
        label contains segmentation labels for each point + coords of gt points

        RETURNS:
            - feature_enc: encoder features at each spatial resolution.
            - feature_dec: decoder features at each spatial resolution.
        '''
        point_cloud, = input
        coords = point_cloud[:, 0:self.dimension+1].float()
        normalized_coords = (coords[:, :self.embedding_dim] - float(self.spatial_size) / 2) \
                    / (float(self.spatial_size) / 2)
        energy = point_cloud[:, self.dimension+1].float().view(-1, 1)
        # if self.coordConv:
        #     features = torch.cat([normalized_coords, features], dim=1)

        x = self.input((coords, energy))
        encoder_res = self.encoder(x)
        features_enc = encoder_res['features_enc']
        deepest_layer = encoder_res['deepest_layer']
        features_cluster = self.cluster_net(features_enc, deepest_layer)
        features_seediness = self.seed_net(features_enc, deepest_layer)
        features_seg = self.seg_net(features_enc, deepest_layer)

        segmentation = features_seg[-1]
        embeddings = features_cluster[-1]
        seediness = features_seediness[-1]

        features_gnn = self.concat([segmentation, seediness, embeddings])
        features_gnn = self.output_gnn_features(features_gnn)

        embeddings = self.output_cluster(embeddings)
        embeddings[:, :self.embedding_dim] = self.tanh(embeddings[:, :self.embedding_dim])
        embeddings[:, :self.embedding_dim] += normalized_coords
        embeddings[:, self.embedding_dim:] = self.tanh(embeddings[:, self.embedding_dim:])

        res = {}

        ppn_inputs = {
            'ppn_feature_enc': encoder_res["features_enc"],
            'ppn_feature_dec': [deepest_layer] + features_seg
        }

        if self.ghost:
            ghost_mask = self.linear_ghost(segmentation)
            res['ghost'] = [ghost_mask.features]
            ppn_inputs['ghost'] = res['ghost'][0]

        # print(ppn_inputs['ppn_feature_dec'][-1].features.shape)

        seediness = self.output_seediness(seediness)
        segmentation = self.output_segmentation(segmentation)

        res.update({
            'embeddings': [embeddings[:, :self.embedding_dim]],
            'margins': [embeddings[:, self.embedding_dim:]],
            'seediness': [seediness],
            'features_gnn': [features_gnn],
            'segmentation': [segmentation],
            'coords': [coords]
        })

        ppn_res = self.ppn(ppn_inputs)
        res.update(ppn_res)
        # print('PPN RES')
        # for key, val in ppn_res.items():
        #     print(key, val[0].shape)

        return res