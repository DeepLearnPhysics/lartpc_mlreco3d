import torch
import torch.nn as nn
import numpy as np
import sparseconvnet as scn
from collections import defaultdict

from mlreco.models.layers.uresnet import UResNetEncoder, UResNetDecoder
from mlreco.models.layers.base import SCNNetworkBase

from mlreco.models.ppn import PPN

from torch_geometric.data import Batch as GraphData

# ----------------------HELPER FUNCTIONS------------------------


def gaussian_kernel(centroid, sigma, eps=1e-8):
    def f(x):
        dists = torch.sum(torch.pow(x - centroid, 2), dim=1)
        probs = torch.exp(-dists / (2.0 * (sigma)**2 + eps))
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
    edge_index = []
    edge_features = []
    for bidx in unique_batch:
        mask = bidx == batch_idx
        clust_ids = torch.nonzero(mask).flatten()
        nodes_batch = nodes[mask]
        subindex = torch.arange(nodes_batch.shape[0])
        N = nodes_batch.shape[0]
        for i, row in enumerate(nodes_batch):
            submask = subindex != i
            edge_idx = [[clust_ids[i].item(), clust_ids[j].item()] for j in subindex[submask]]
            edge_index.extend(edge_idx)
            others = nodes_batch[submask]
            ei2j = edge_net(row.expand_as(others), others)
            edge_features.extend(ei2j)

    edge_index = np.vstack(edge_index)
    edge_features = torch.stack(edge_features, dim=0)

    return edge_index, edge_features


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
        self.linear1 = nn.Linear(num_input, num_output)
        self.norm1 = nn.BatchNorm1d(num_output)
        self.linear2 = nn.Linear(num_output, num_output)
        self.norm2 = nn.BatchNorm1d(num_output)
        self.linear3 = nn.Linear(num_output, num_output)

    def forward(self, x):
        x = self.linear1(x)
        x = self.norm1(x)
        x = self.linear2(x)
        x = self.norm2(x)
        x = self.linear3(x)
        return x


def get_gnn_input(coords, features_gnn, fit_predict,
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
    batch_index = coords[:,-1].unique()
    #node_group_labels = []
    #centroids = []
    fragments = []

    if train:
        fragment_labels = kwargs['fragment_labels']
        fragments = []
        #group_labels = kwargs['group_labels']
        embeddings = kwargs['embeddings']
        # print(embeddings.shape)
        for bidx in batch_index:
            batch_mask = coords[:, 3] == bidx
            batch_map = torch.nonzero(batch_mask).flatten()
            featuresAgg_batch = features_gnn[batch_mask]
            fragment_batch = fragment_labels[batch_mask]
            #group_batch = group_labels[batch_mask]
            embeddings_batch = embeddings[batch_mask]
            fragment_batch_ids = fragment_batch.unique()
            for fid in fragment_batch_ids:
                fragments.append(batch_map[fragment_batch == fid])
                mean_features = featuresAgg_batch[fragment_batch == fid].mean(dim=0)
                #group_id = group_batch[fragment_batch == fid].unique()
                #centroid = torch.mean(embeddings_batch[fragment_batch == fid], dim=0)
                # print(centroid)
                #centroids.append(centroid)
                #node_group_labels.append(int(group_id))
                nodes_full.append(mean_features)
                nodes_batch_id.append(int(bidx))

        fragments = np.array([f.detach().cpu().numpy() for f in fragments])
    else:
        with torch.no_grad():
            fragments = []
            semantic_labels = kwargs['semantic_labels']
            #fragment_labels = torch.ones(len(semantic_labels), dtype=torch.int32)*-1
            high_energy_mask = torch.nonzero(semantic_labels < 4).flatten()
            semantic_labels_highE = semantic_labels[high_energy_mask]
            embeddings = kwargs['embeddings'][high_energy_mask]
            margins = kwargs['margins'][high_energy_mask]
            seediness = kwargs['seediness'][high_energy_mask]
            features_gnn_highE = features_gnn[high_energy_mask]
            kernel_func = kwargs.get('kernel_func', multivariate_kernel)
            coords_highE = coords[high_energy_mask]
            for bidx in batch_index:
                batch_mask = torch.nonzero(coords_highE[:, 3] == bidx).flatten()
                slabel = semantic_labels_highE[batch_mask]
                embedding_batch = embeddings[batch_mask]
                featuresAgg_batch = features_gnn_highE[batch_mask]
                margins_batch = margins[batch_mask]
                seediness_batch = seediness[batch_mask]
                for s in slabel.unique():
                    segment_mask = torch.nonzero(slabel == s).flatten()
                    featuresAgg_class = featuresAgg_batch[segment_mask]
                    embedding_class = embedding_batch[segment_mask]
                    margins_class = margins_batch[segment_mask]
                    seediness_class = seediness_batch[segment_mask]
                    pred_labels = fit_predict(
                        embedding_class, seediness_class, margins_class, kernel_func)
                    # Adding Node Features
                    for c in pred_labels.unique():
                        mask = pred_labels == c
                        fragments.append(np.array(high_energy_mask[batch_mask[segment_mask[mask]]]))
                        mean_features = featuresAgg_class[mask].mean(dim=0)
                        nodes_full.append(mean_features)
                        nodes_batch_id.append(bidx)
                        #centroid = torch.mean(embedding_class[mask], dim=0)
                        #centroids.append(centroid)
            fragments = np.array(fragments)

    device = features_gnn.device
    nodes_full = torch.stack(nodes_full, dim=0)
    node_batch_id = torch.Tensor(nodes_batch_id).to(device)
    #centroids = torch.stack(centroids, dim=0)

    # Compile Pairwise Edge Features
    # edge_indices, edge_features, edge_batch_indices = get_edge_features(
    #     nodes_full, node_batch_id, edge_net)

    # gnn_input = GraphData(x=nodes_full,
    #                       edge_index=edge_indices.view(2, -1).long(),
    #                       edge_attr=edge_features,
    #                       batch=node_batch_id.long())
    # gnn_input.edge_batch = edge_batch_indices
    # gnn_input.centroids = centroids
    # if train:
    #     node_group_labels = torch.Tensor(node_group_labels).to(device)
    #     gnn_input.node_group_labels = node_group_labels

    return nodes_full, node_batch_id.long(), fragments


def fit_predict(embeddings, seediness, margins, fitfunc,
                 s_threshold=0.0, p_threshold=0.5, cluster_all=False):
    pred_labels = -np.ones(embeddings.shape[0])
    probs = []
    spheres = []
    seediness_copy = seediness.clone()
    count = 0
    if seediness_copy.shape[0] == 1:
        return torch.argmax(seediness_copy)
    while count < int(seediness.shape[0]):
        i = torch.argsort(seediness_copy.squeeze(), descending=True)[0]
        seedScore = seediness[i]
        if seedScore < s_threshold:
            break
        centroid = embeddings[i]
        sigma = margins[i]
        spheres.append((centroid, sigma))
        f = fitfunc(centroid, sigma)
        pValues = f(embeddings)
        probs.append(pValues.view(-1, 1))
        cluster_index = (pValues > p_threshold).view(-1) & (seediness_copy > 0).view(-1)
        seediness_copy[cluster_index] = -1
        count += torch.sum(cluster_index).item()
        if torch.sum(cluster_index).item() == 0:
            break
    if len(probs) == 0:
        return torch.tensor(pred_labels)
    probs = torch.cat(probs, dim=1)
    pred_labels = torch.argmax(probs, dim=1)
    if not cluster_all:
        mask = torch.max(probs, dim=1).values < p_threshold
        pred_labels[mask] = -1
    return pred_labels


# --------------------------CHAINS------------------------------


class FullCNN(SCNNetworkBase):
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
        self.sigma_dim = self.model_config.get('sigma_dim', 1)
        self.embedding_dim = self.model_config.get('embedding_dim', 3)
        self.num_classes = self.model_config.get('num_classes', 5)
        self.num_gnn_features = self.model_config.get('num_gnn_features', 16)
        self.inputKernel = self.model_config.get('input_kernel_size', 3)
        self.coordConv = self.model_config.get('coordConv', False)

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

        # Backbone UResNet. Do NOT change namings!
        self.encoder = UResNetEncoder(cfg, name='uresnet_encoder')
        print('Segmentation')
        self.seg_net = UResNetDecoder(cfg, name='segmentation_decoder')
        print('Seediness')
        self.seed_net = UResNetDecoder(cfg, name='seediness_decoder')
        print('Clustering')
        self.cluster_net = UResNetDecoder(cfg, name='embedding_decoder')

        # Encoder-Decoder 1x1 Connections
        encoder_planes = [i for i in self.encoder.nPlanes]
        seg_planes = [i for i in self.seg_net.nPlanes]
        cluster_planes = [i for i in self.cluster_net.nPlanes]
        seed_planes = [i for i in self.seed_net.nPlanes]

        # print("Encoder Planes: ", encoder_planes)
        # print("Seg Planes: ", seg_planes)
        # print("Cluster Planes: ", cluster_planes)
        # print("Seediness Planes: ", seed_planes)

        self.skip_mode = self.model_config.get('skip_mode', 'default')

        self.seg_skip = scn.Sequential()
        self.cluster_skip = scn.Sequential()
        self.seed_skip = scn.Sequential()

        # print(self.seg_skip)
        # print(self.cluster_skip)
        # print(self.seed_skip)

        # Output Layers
        self.output_cluster = scn.Sequential()
        self._nin_block(self.output_cluster, self.cluster_net.num_filters, 4)
        self.output_cluster.add(scn.OutputLayer(self.dimension))

        self.output_seediness = scn.Sequential()
        self._nin_block(self.output_seediness, self.seed_net.num_filters, 1)
        self.output_seediness.add(scn.OutputLayer(self.dimension))

        self.output_segmentation = scn.Sequential()
        self._nin_block(self.output_segmentation, self.seg_net.num_filters, self.num_classes)
        self.output_segmentation.add(scn.OutputLayer(self.dimension))

        self.output_gnn_features = scn.Sequential()
        sum_filters = self.seg_net.num_filters + self.seed_net.num_filters + self.cluster_net.num_filters
        self._resnet_block(self.output_gnn_features, sum_filters, self.num_gnn_features)
        self._nin_block(self.output_gnn_features, self.num_gnn_features, self.num_gnn_features)
        self.output_gnn_features.add(scn.OutputLayer(self.dimension))

        if self.ghost:
            self.linear_ghost = scn.Sequential()
            self._nin_block(self.linear_ghost, self.num_filters, 2)
            # self.linear_ghost.add(scn.OutputLayer(self.dimension))

        # PPN
        self.ppn  = PPN(cfg)

        if self.skip_mode == 'default':

            for p1, p2 in zip(encoder_planes, seg_planes):
                self.seg_skip.add(scn.Identity())
            for p1, p2 in zip(encoder_planes, cluster_planes):
                self.cluster_skip.add(scn.Identity())
            for p1, p2 in zip(encoder_planes, seed_planes):
                self.seed_skip.add(scn.Identity())
            self.ppn_transform = scn.Sequential()
            ppn1_num_filters = seg_planes[self.ppn.ppn1_stride-self.ppn._num_strides]
            self._nin_block(self.ppn_transform, encoder_planes[-1], ppn1_num_filters)


        elif self.skip_mode == '1x1':

            for p1, p2 in zip(encoder_planes, seg_planes):
                self._nin_block(self.seg_skip, p1, p2)

            for p1, p2 in zip(encoder_planes, cluster_planes):
                self._nin_block(self.cluster_skip, p1, p2)

            for p1, p2 in zip(encoder_planes, seed_planes):
                self._nin_block(self.seed_skip, p1, p2)

            self.ppn_transform = scn.Identity()

        else:
            raise ValueError('Invalid skip connection mode!')

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

        #print(self)


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
        features = point_cloud[:, self.dimension+1].float().view(-1, 1)
        if self.coordConv:
            features = torch.cat([normalized_coords, features], dim=1)

        x = self.input((coords, features))
        encoder_res = self.encoder(x)
        features_enc = encoder_res['features_enc']
        deepest_layer = encoder_res['deepest_layer']

        # print([t.features.shape for t in features_enc])

        seg_decoder_input = [None]
        for i, layer in enumerate(features_enc[1:]):
            seg_decoder_input.append(self.seg_skip[i](layer))
        deep_seg = self.seg_skip[-1](deepest_layer)

        seed_decoder_input = [None]
        for i, layer in enumerate(features_enc[1:]):
            seed_decoder_input.append(self.seed_skip[i](layer))
        deep_seed = self.seed_skip[-1](deepest_layer)
        #
        cluster_decoder_input = [None]
        for i, layer in enumerate(features_enc[1:]):
            cluster_decoder_input.append(self.cluster_skip[i](layer))
        deep_cluster = self.cluster_skip[-1](deepest_layer)

        # print([t.features.shape for t in seg_decoder_input[1:]])
        # print([t.features.shape for t in seed_decoder_input[1:]])
        # print([t.features.shape for t in cluster_decoder_input[1:]])

        features_cluster = self.cluster_net(features_enc, deepest_layer)
        features_seediness = self.seed_net(seed_decoder_input, deep_seed)
        features_seg = self.seg_net(seg_decoder_input, deep_seg)

        segmentation = features_seg[-1]
        embeddings = features_cluster[-1]
        seediness = features_seediness[-1]

        features_gnn = self.concat([segmentation, seediness, embeddings])
        features_gnn = self.output_gnn_features(features_gnn)

        embeddings = self.output_cluster(embeddings)
        embeddings[:, :self.embedding_dim] = self.tanh(embeddings[:, :self.embedding_dim])
        embeddings[:, :self.embedding_dim] += normalized_coords

        res = {}

        ppn_inputs = {
            'ppn_feature_enc': seg_decoder_input,
            'ppn_feature_dec': [self.ppn_transform(deep_seg)] + features_seg
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
            'margins': [2 * self.sigmoid(embeddings[:, self.embedding_dim:])],
            'seediness': [self.sigmoid(seediness)],
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


class FullCNNSELU(SCNNetworkBase):

    def __init__(self, cfg, name='full_cnn'):
        super(FullCNNSELU, self).__init__(cfg, name='network_base')

        self.model_config = cfg[name]
        self.num_filters = self.model_config.get('filters', 16)
        self.ghost = self.model_config.get('ghost', False)
        self.seed_dim = self.model_config.get('seed_dim', 1)
        self.sigma_dim = self.model_config.get('sigma_dim', 1)
        self.embedding_dim = self.model_config.get('embedding_dim', 3)
        self.num_classes = self.model_config.get('num_classes', 5)
        self.num_gnn_features = self.model_config.get('num_gnn_features', 16)
        self.inputKernel = self.model_config.get('input_kernel_size', 3)

        # Network Freezing Options
        self.encoder_freeze = self.model_config.get('encoder_freeze', False)
        self.ppn_freeze = self.model_config.get('ppn_freeze', True)
        self.segmentation_freeze = self.model_config.get('segmentation_freeze', False)
        self.embedding_freeze = self.model_config.get('embedding_freeze', False)
        self.seediness_freeze = self.model_config.get('seediness_freeze', True)

        # Input Layer Configurations and commonly used scn operations.
        self.input = scn.Sequential().add(
            scn.InputLayer(self.dimension, self.spatial_size, mode=3)).add(
            scn.SubmanifoldConvolution(self.dimension, self.nInputFeatures, \
            self.num_filters, self.inputKernel, self.allow_bias)) # Kernel size 3, no bias
        self.concat = scn.JoinTable()
        self.add = scn.AddTable()

        self.encoder = UResNetEncoder(cfg, name='uresnet_encoder')
