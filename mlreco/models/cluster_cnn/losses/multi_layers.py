import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import sparseconvnet as scn

from collections import defaultdict
from mlreco.models.cluster_cnn.utils import distance_matrix, pairwise_distances
from .single_layers import DiscriminativeLoss


class MultiScaleLoss(DiscriminativeLoss):

    def __init__(self, cfg, name='spice_loss'):
        super(MultiScaleLoss, self).__init__(cfg)
        self.loss_config = cfg['spice_loss']
        self.num_strides = self.loss_config.get('num_strides', 5)

        self.intra_margins = self.loss_config.get('intra_margins',
            [self.loss_hyperparams['intra_margin'] for i in range(self.num_strides)])
        self.inter_margins = self.loss_config.get('inter_margins',
            [self.loss_hyperparams['inter_margin'] for i in range(self.num_strides)])


    def compute_loss_layer(self, embedding_scn, slabels, clabels, batch_idx, **kwargs):
        '''
        Compute the multi-class loss for a feature map on a given layer.
        We group the loss computation to a function in order to compute the
        clustering loss over the decoding feature maps.

        INPUTS:
            - embedding (torch.Tensor): (N, d) Tensor with embedding space
                coordinates.
            - slabels (torch.Tensor): (N, 5) Tensor with segmentation labels
            - clabels (torch.Tensor): (N, 5) Tensor with cluster labels
            - batch_idx (list): list of batch indices, ex. [0, 1, ..., 4]

        OUTPUT:
            - loss (torch.Tensor): scalar number (1x1 Tensor) corresponding
                to calculated loss over a given layer.
        '''
        loss = defaultdict(list)
        accuracy = defaultdict(list)

        coords = embedding_scn.get_spatial_locations().numpy()
        perm = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0], coords[:, 3]))
        embedding = embedding_scn.features[perm]
        coords = coords[perm]

        for bidx in batch_idx:
            index = slabels[:, 3].int() == bidx
            embedding_batch = embedding[index]
            slabels_batch = slabels[index][:, -1]
            clabels_batch = clabels[index][:, -2]
            # Compute discriminative loss for current event in batch
            if self.use_segmentation:
                loss_dict, acc_segs = self.combine_multiclass(
                    embedding_batch, slabels_batch, clabels_batch, **kwargs)
                for key, val in loss_dict.items():
                    loss[key].append( sum(val) / float(len(val)) )
                for s, acc in acc_segs.items():
                    accuracy[s].append(acc)
                acc = sum(acc_segs.values()) / len(acc_segs.values())
                accuracy['accuracy'].append(acc)
            else:
                loss["loss"].append(self.combine(embedding_batch, clabels_batch, **kwargs))
                acc = self.compute_heuristic_accuracy(embedding_batch, clabels_batch)
                accuracy['accuracy'].append(acc)

        # Averaged over batch at each layer
        loss = { key : sum(l) / float(len(l)) for key, l in loss.items() }
        accuracy = { key : sum(l) / float(len(l)) for key, l in accuracy.items() }
        return loss, accuracy


    def forward(self, out, semantic_labels, group_labels):
        '''
        Forward function for the Discriminative Loss Module.

        Inputs:
            out: output of UResNet; embedding-space coordinates.
            semantic_labels: ground-truth semantic labels
            group_labels: ground-truth instance labels
        Returns:
            (dict): A dictionary containing key-value pairs for
            loss, accuracy, etc.
        '''

        loss = defaultdict(list)
        accuracy = defaultdict(list)
        num_gpus = len(semantic_labels)
        num_layers = len(out['cluster_feature'][0])

        for i_gpu in range(num_gpus):
            batch_idx = semantic_labels[i_gpu][0][:, 3].detach().cpu().int().numpy()
            batch_idx = np.unique(batch_idx)
            batch_size = len(batch_idx)
            # Summing clustering loss over layers.
            for i, em in enumerate(out['cluster_feature'][i_gpu]):
                delta_var, delta_dist = self.intra_margins[i], self.inter_margins[i]
                loss_i, acc_i = self.compute_loss_layer(
                    em, semantic_labels[i_gpu][i], group_labels[i_gpu][i], batch_idx,
                    delta_var=delta_var, delta_dist=delta_dist)
                for key, val in loss_i.items():
                    loss[key].append(val)
                # Compute accuracy only at last layer.
                if i == 0:
                    acc_clustering = acc_i
            for key, acc in acc_clustering.items():
                # Batch Averaged Accuracy
                accuracy[key].append(acc)

        # Average over layers and num_gpus
        loss_avg = {}
        acc_avg = {}
        for key, val in loss.items():
            loss_avg[key] = sum(val) / len(val)
        for key, val in accuracy.items():
            acc_avg[key] = sum(val) / len(val)

        res = {}
        res.update(loss_avg)
        res.update(acc_avg)

        return res


class MultiScaleLoss2(MultiScaleLoss):
    '''
    Same as multi scale loss, but we include enemy loss in intra loss.
    '''
    def __init__(self, cfg, name='spice_loss'):
        super(MultiScaleLoss2, self).__init__(cfg, name=name)
        self.ally_weight = self.loss_config.get('ally_weight', 1.0)
        self.ally_margins = self.intra_margins
        self.enemy_weight = self.loss_config.get('enemy_weight', 10.0)
        self.loss_hyperparams['enemy_margin'] = self.loss_config.get('enemy_margin', 1.0)
        self.enemy_margins = self.loss_config.get('enemy_margins',
            [self.loss_hyperparams['enemy_margin'] for i in range(self.num_strides)])


    def intra_cluster_loss(self, features, labels, cluster_means,
                           ally_margin=0.5, enemy_margin=1.0):
        '''
        Intra-cluster loss, with per-voxel weighting and enemy loss.
        This variant of intra-cluster loss penalizes the distance
        from the centroid to its enemies in addition to pulling
        ally points towards the center.

        INPUTS:
            - ally_margin (float): centroid pulls all allied points
            inside this margin.
            - enemy_margin (float): centroid pushs all enemy points
            inside this margin.
            - weight:
        '''
        intra_loss = 0.0
        ally_loss, enemy_loss = 0.0, 0.0
        n_clusters = len(cluster_means)
        cluster_labels = labels.unique(sorted=True)
        for i, c in enumerate(cluster_labels):
            index = (labels == c)
            allies = torch.norm(features[index] - cluster_means[i] + 1e-8,
                               p=self.norm, dim=1)
            allies = torch.clamp(allies - ally_margin, min=0)
            x = self.ally_weight * torch.mean(torch.pow(allies, 2))
            intra_loss += x
            ally_loss += float(x)
            if index.all():
                continue
            enemies = torch.norm(features[~index] - cluster_means[i] + 1e-8,
                    p=self.norm, dim=1)
            enemies = torch.clamp(enemy_margin - enemies, min=0)
            x = self.enemy_weight * torch.sum(torch.pow(enemies, 2))
            intra_loss += x
            enemy_loss += float(x)

        intra_loss /= n_clusters
        ally_loss /= n_clusters
        enemy_loss /= n_clusters
        return intra_loss, ally_loss, enemy_loss


    def combine(self, features, labels, **kwargs):
        '''
        Wrapper function for combining different components of the loss function.
        Inputs:
            features (torch.Tensor): pixel embeddings
            labels (torch.Tensor): ground-truth instance labels
        Returns:
            loss: combined loss, in most cases over a given semantic class.
        '''
        # Clustering Loss Hyperparameters
        # We allow changing the parameters at each computation in order
        # to alter the margins at each spatial resolution in multi-scale losses.
        ally_margin = kwargs.get('ally_margin', 0.5)
        enemy_margin = kwargs.get('enemy_margin', 1.0)
        inter_margin = kwargs.get('inter_margin', 1.5)
        intra_weight = kwargs.get('intra_weight', 1.0)
        inter_weight = kwargs.get('inter_weight', 1.0)
        reg_weight = kwargs.get('reg_weight', 0.001)

        c_means = self.find_cluster_means(features, labels)
        inter_loss = self.inter_cluster_loss(c_means, margin=inter_margin)
        intra_loss, ally_loss, enemy_loss = self.intra_cluster_loss(features,
                                           labels,
                                           c_means,
                                           ally_margin=ally_margin,
                                           enemy_margin=enemy_margin)
        reg_loss = self.regularization(c_means)

        loss = intra_weight * intra_loss + inter_weight \
            * inter_loss + reg_weight * reg_loss

        return {
            'loss': loss,
            'intra_loss': intra_weight * float(intra_loss),
            'inter_loss': inter_weight * float(inter_loss),
            'reg_loss': reg_weight * float(reg_loss),
            'ally_loss': intra_weight * float(ally_loss),
            'enemy_loss': intra_weight * float(enemy_loss)
        }


    def forward(self, out, semantic_labels, group_labels):
        '''
        Forward function for the Discriminative Loss Module.

        Inputs:
            out: output of UResNet; embedding-space coordinates.
            semantic_labels: ground-truth semantic labels
            group_labels: ground-truth instance labels
        Returns:
            (dict): A dictionary containing key-value pairs for
            loss, accuracy, etc.
        '''

        loss = defaultdict(list)
        accuracy = defaultdict(list)
        num_gpus = len(semantic_labels)
        num_layers = len(out['cluster_feature'][0])

        for i_gpu in range(num_gpus):
            batch_idx = semantic_labels[i_gpu][0][:, 3].detach().cpu().int().numpy()
            batch_idx = np.unique(batch_idx)
            batch_size = len(batch_idx)
            # Summing clustering loss over layers.
            for i, em in enumerate(out['cluster_feature'][i_gpu]):
                delta_var, delta_dist = self.ally_margins[i], self.inter_margins[i]
                delta_enemy = self.enemy_margins[i]
                loss_i, acc_i = self.compute_loss_layer(
                    em, semantic_labels[i_gpu][i], group_labels[i_gpu][i], batch_idx,
                    ally_margin=delta_var, inter_margin=delta_dist, enemy_margin=delta_enemy)
                for key, val in loss_i.items():
                    loss[key].append(val)
                # Compute accuracy only at last layer.
                if i == 0:
                    acc_clustering = acc_i
            for key, acc in acc_clustering.items():
                # Batch Averaged Accuracy
                accuracy[key].append(acc)

        # Average over layers and num_gpus
        loss_avg = {}
        acc_avg = {}
        for key, val in loss.items():
            loss_avg[key] = sum(val) / len(val)
        for key, val in accuracy.items():
            acc_avg[key] = sum(val) / len(val)

        res = {}
        res.update(loss_avg)
        res.update(acc_avg)

        return res


class WeightedMultiLoss(MultiScaleLoss):
    '''
    Same as MultiScaleLoss, but with attention weighting.
    '''
    def __init__(self, cfg, name='spice_loss'):
        super(WeightedMultiLoss, self).__init__(cfg, name=name)
        self.attention_kernel = self.loss_config.get('attention_kernel', 1)
        if self.attention_kernel == 0:
            self.kernel_func = lambda x: (1.0 + torch.exp(-x)) / 2.0
        elif self.attention_kernel == 1:
            self.kernel_func = lambda x: 2.0 / (1 + torch.exp(-x))
        else:
            raise ValueError('Invalid weighting kernel function mode.')


    def intra_cluster_loss(self, features, labels, cluster_means, coords, margin=0.5):
        '''
        Intra-cluster loss, with per-voxel weighting and enemy loss.
        This variant of intra-cluster loss penalizes the distance
        from the centroid to its enemies in addition to pulling
        ally points towards the center.

        INPUTS:
            - ally_margin (float): centroid pulls all allied points
            inside this margin.
            - enemy_margin (float): centroid pushs all enemy points
            inside this margin.
            - weight: Tensor with features.shape[0] entries corresponding to
            attention weights.
        '''
        intra_loss = 0.0
        ally_loss, enemy_loss = 0.0, 0.0
        n_clusters = len(cluster_means)
        cluster_labels = labels.unique(sorted=True)
        with torch.no_grad():
            coords_mean = self.find_cluster_means(coords, labels)
        for i, c in enumerate(cluster_labels):
            index = (labels == c)
            allies = torch.norm(features[index] - cluster_means[i] + 1e-8,
                               p=self.norm, dim=1)
            allies = torch.clamp(allies - margin, min=0)
            with torch.no_grad():
                dists = torch.norm(coords[index] - coords_mean[i] + 1e-8,
                                    p=self.norm, dim=1)
                dists = dists / (dists.std(unbiased=False) + 1e-4)
                weight = self.kernel_func(dists)
            intra_loss += torch.mean(weight * torch.pow(allies, 2))

        intra_loss /= n_clusters
        return intra_loss


    def combine(self, features, labels, **kwargs):
        '''
        Wrapper function for combining different components of the loss function.
        Inputs:
            features (torch.Tensor): pixel embeddings
            labels (torch.Tensor): ground-truth instance labels
        Returns:
            loss: combined loss, in most cases over a given semantic class.
        '''
        # Clustering Loss Hyperparameters
        # We allow changing the parameters at each computation in order
        # to alter the margins at each spatial resolution in multi-scale losses.
        inter_margin = kwargs.get('inter_margin', 1.5)
        intra_margin = kwargs.get('intra_margin', 0.5)
        intra_weight = kwargs.get('intra_weight', 1.0)
        inter_weight = kwargs.get('inter_weight', 1.0)
        reg_weight = kwargs.get('reg_weight', 0.001)
        coords = kwargs['coords']

        c_means = self.find_cluster_means(features, labels)
        inter_loss = self.inter_cluster_loss(c_means, margin=inter_margin)
        intra_loss = self.intra_cluster_loss(features,
                                           labels,
                                           c_means,
                                           coords,
                                           margin=intra_margin)
        reg_loss = self.regularization(c_means)

        loss = intra_weight * intra_loss + inter_weight \
            * inter_loss + reg_weight * reg_loss

        return {
            'loss': loss,
            'intra_loss': intra_weight * float(intra_loss),
            'inter_loss': inter_weight * float(inter_loss),
            'reg_loss': reg_weight * float(reg_loss)
        }


    def forward(self, result, segment_label, cluster_label):
        '''
        Mostly borrowed from uresnet_clustering.py
        '''
        num_gpus = len(segment_label)
        loss, accuracy = defaultdict(list), defaultdict(list)

        # Loop first over scaled feature maps
        for i_gpu in range(num_gpus):
            for depth in range(self.depth):

                batch_ids = segment_label[i_gpu][depth][:, 3].detach().cpu().int().numpy()
                batch_ids = np.unique(batch_ids)
                batch_size = len(batch_ids)

                embedding = result['cluster_feature'][i_gpu][depth]
                clabels_depth = cluster_label[i_gpu][depth]
                slabels_depth = segment_label[i_gpu][depth]

                coords = embedding.get_spatial_locations()[:, :4]
                perm = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0], coords[:, 3]))
                coords = coords[perm].float().cuda()
                feature_map = embedding.features[perm]

                loss_event = defaultdict(list)
                acc_event = defaultdict(list)
                for bidx in batch_ids:
                    batch_mask = coords[:, 3] == bidx
                    hypercoordinates = feature_map[batch_mask]
                    slabels_event = slabels_depth[batch_mask]
                    clabels_event = clabels_depth[batch_mask]
                    coords_event = coords[batch_mask]

                    # Loop over semantic labels:
                    semantic_classes = slabels_event[:, -1].unique()
                    n_classes = len(semantic_classes)
                    loss_class = defaultdict(list)
                    acc_class = defaultdict(list)
                    acc_avg = 0.0
                    for class_ in semantic_classes:
                        k = int(class_)
                        class_mask = slabels_event[:, -1] == class_
                        embedding_class = hypercoordinates[class_mask]
                        cluster_class = clabels_event[class_mask][:, -2]
                        coords_class = coords_event[class_mask][:, :3]
                        # Clustering Loss
                        if depth == 0:
                            acc = self.compute_heuristic_accuracy(embedding_class,
                                                                cluster_class)
                            accuracy['accuracy_{}'.format(k)].append(acc)
                            acc_avg += acc / n_classes
                        closs = self.combine(embedding_class,
                                             cluster_class,
                            intra_margin=self.intra_margins[depth],
                            inter_margin=self.inter_margins[depth],
                            coords=coords_class)
                        for key, val in closs.items():
                            loss_class[key].append(val)
                    if depth == 0:
                        accuracy['accuracy'].append(acc_avg)
                    for key, val in loss_class.items():
                        loss_event[key].append(sum(val) / len(val))
                for key, val in loss_event.items():
                    loss[key].append(sum(val) / len(val))

        res = {}

        for key, val in loss.items():
            res[key] = sum(val) / len(val)

        for key, val in accuracy.items():
            res[key] = sum(val) / len(val)

        return res


class DistanceEstimationLoss(MultiScaleLoss):

    def __init__(self, cfg, name='spice_loss'):
        super().__init__(cfg)
        print('Distance Estimation')
        self.loss_config = cfg[name]
        self.huber_loss = torch.nn.SmoothL1Loss(reduction='mean')
        self.distance_estimate_weight = self.loss_config.get('distance_estimate_weight', 1.0)
        self.clustering_weight = self.loss_config.get('clustering_weight', 1.0)
        print(self)

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
            return nnMap


    def forward(self, result, segment_label, cluster_label):
        '''
        Mostly borrowed from uresnet_clustering.py
        '''
        num_gpus = len(segment_label)
        loss, accuracy = defaultdict(list), defaultdict(list)

        # Loop first over scaled feature maps
        for i_gpu in range(num_gpus):
            distance_estimate = result['distance_estimation'][i_gpu]
            dcoords = distance_estimate.get_spatial_locations()[:, :4]
            perm = np.lexsort((dcoords[:, 2], dcoords[:, 1], dcoords[:, 0], dcoords[:, 3]))
            distance_estimate = distance_estimate.features[perm]
            for depth in range(self.depth):

                batch_ids = segment_label[i_gpu][depth][:, 3].detach().cpu().int().numpy()
                batch_ids = np.unique(batch_ids)
                batch_size = len(batch_ids)

                embedding = result['cluster_feature'][i_gpu][depth]
                clabels_depth = cluster_label[i_gpu][depth]
                slabels_depth = segment_label[i_gpu][depth]

                coords = embedding.get_spatial_locations()[:, :4]
                perm = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0], coords[:, 3]))
                coords = coords[perm].float().cuda()
                feature_map = embedding.features[perm]

                loss_event = defaultdict(list)
                acc_event = defaultdict(list)
                for bidx in batch_ids:
                    batch_mask = coords[:, 3] == bidx
                    hypercoordinates = feature_map[batch_mask]
                    slabels_event = slabels_depth[batch_mask]
                    clabels_event = clabels_depth[batch_mask]
                    coords_event = coords[batch_mask]
                    if depth == 0:
                        distances_event = distance_estimate[batch_mask]

                    # Loop over semantic labels:
                    semantic_classes = slabels_event[:, -1].unique()
                    n_classes = len(semantic_classes)
                    loss_class = defaultdict(list)
                    acc_class = defaultdict(list)
                    acc_avg = 0.0
                    for class_ in semantic_classes:
                        k = int(class_)
                        class_mask = slabels_event[:, -1] == class_
                        # print('Slabels: ', slabels_event[class_mask])
                        embedding_class = hypercoordinates[class_mask]
                        cluster_class = clabels_event[class_mask][:, -2]
                        # print('Clabels: ', clabels_event[class_mask])
                        coords_class = coords_event[class_mask][:, :3]
                        # print('Coords: ', coords_class)
                        # assert False
                        if depth == 0:
                            distances_class = distances_event[class_mask]
                            acc = self.compute_heuristic_accuracy(embedding_class,
                                                                cluster_class)
                            accuracy['accuracy_{}'.format(k)].append(acc)
                            acc_avg += acc / n_classes
                            dMap = self.get_nn_map(embedding_class, cluster_class)
                            dloss = self.huber_loss(dMap, distances_class) * self.distance_estimate_weight
                            loss_class['loss_distance_estimation'].append(dloss)
                        # Clustering Loss
                        closs = self.combine(embedding_class,
                                             cluster_class,
                            intra_margin=self.intra_margins[depth],
                            inter_margin=self.inter_margins[depth])
                        for key, val in closs.items():
                            loss_class[key].append(val)
                    if depth == 0:
                        accuracy['accuracy'].append(acc_avg)
                    for key, val in loss_class.items():
                        loss_event[key].append(sum(val) / len(val))
                for key, val in loss_event.items():
                    loss[key].append(sum(val) / len(val))

        res = {}

        for key, val in loss.items():
            res[key] = sum(val) / len(val)
        res['loss'] += res['loss_distance_estimation']
        res['loss_distance_estimation'] = float(res['loss_distance_estimation'])

        for key, val in accuracy.items():
            res[key] = sum(val) / len(val)

        print(res)
        return res


class DistanceEstimationLoss2(MultiScaleLoss2):

    def __init__(self, cfg, name='spice_loss'):
        super(DistanceEstimationLoss2, self).__init__(cfg, name='uresnet')
        self.loss_config = cfg[name]
        self.huber_loss = torch.nn.SmoothL1Loss(reduction='mean')
        self.distance_estimate_weight = self.loss_config.get('distance_estimate_weight', 1.0)
        self.clustering_weight = self.loss_config.get('clustering_weight', 1.0)

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
            return nnMap


    def forward(self, result, segment_label, cluster_label):
        '''
        Mostly borrowed from uresnet_clustering.py
        '''
        num_gpus = len(segment_label)
        loss, accuracy = defaultdict(list), defaultdict(list)

        # Loop first over scaled feature maps
        for i_gpu in range(num_gpus):
            distance_estimate = result['distance_estimation'][i_gpu]
            dcoords = distance_estimate.get_spatial_locations()[:, :4]
            perm = np.lexsort((dcoords[:, 2], dcoords[:, 1], dcoords[:, 0], dcoords[:, 3]))
            distance_estimate = distance_estimate.features[perm]
            for depth in range(self.depth):

                batch_ids = segment_label[i_gpu][depth][:, 3].detach().cpu().int().numpy()
                batch_ids = np.unique(batch_ids)
                batch_size = len(batch_ids)

                embedding = result['cluster_feature'][i_gpu][depth]
                clabels_depth = cluster_label[i_gpu][depth]
                slabels_depth = segment_label[i_gpu][depth]

                coords = embedding.get_spatial_locations()[:, :4]
                perm = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0], coords[:, 3]))
                coords = coords[perm].float().cuda()
                feature_map = embedding.features[perm]

                loss_event = defaultdict(list)
                acc_event = defaultdict(list)
                for bidx in batch_ids:
                    batch_mask = coords[:, 3] == bidx
                    hypercoordinates = feature_map[batch_mask]
                    slabels_event = slabels_depth[batch_mask]
                    clabels_event = clabels_depth[batch_mask]
                    coords_event = coords[batch_mask]
                    if depth == 0:
                        distances_event = distance_estimate[batch_mask]

                    # Loop over semantic labels:
                    semantic_classes = slabels_event[:, -1].unique()
                    n_classes = len(semantic_classes)
                    loss_class = defaultdict(list)
                    acc_class = defaultdict(list)
                    acc_avg = 0.0
                    for class_ in semantic_classes:
                        k = int(class_)
                        class_mask = slabels_event[:, -1] == class_
                        embedding_class = hypercoordinates[class_mask]
                        cluster_class = clabels_event[class_mask][:, -2]
                        coords_class = coords_event[class_mask][:, :3]
                        if depth == 0:
                            distances_class = distances_event[class_mask]
                            acc = self.compute_heuristic_accuracy(embedding_class,
                                                                cluster_class)
                            accuracy['accuracy_{}'.format(k)].append(acc)
                            acc_avg += acc / n_classes
                            dMap = self.get_nn_map(embedding_class, cluster_class)
                            dloss = self.huber_loss(dMap, distances_class) * self.distance_estimate_weight
                            loss_class['loss_distance_estimation'].append(dloss)
                        # Clustering Loss
                        closs = self.combine(embedding_class,
                                             cluster_class,
                            intra_margin=self.intra_margins[depth],
                            inter_margin=self.inter_margins[depth])
                        for key, val in closs.items():
                            loss_class[key].append(val)
                    if depth == 0:
                        accuracy['accuracy'].append(acc_avg)
                    for key, val in loss_class.items():
                        loss_event[key].append(sum(val) / len(val))
                for key, val in loss_event.items():
                    loss[key].append(sum(val) / len(val))

        res = {}

        for key, val in loss.items():
            res[key] = sum(val) / len(val)
        res['loss'] += res['loss_distance_estimation']
        res['loss_distance_estimation'] = float(res['loss_distance_estimation'])

        for key, val in accuracy.items():
            res[key] = sum(val) / len(val)

        return res

class DistanceEstimationLoss3(WeightedMultiLoss):

    def __init__(self, cfg, name='spice_loss'):
        super(DistanceEstimationLoss3, self).__init__(cfg, name='uresnet')
        self.loss_config = cfg[name]
        self.huber_loss = torch.nn.SmoothL1Loss(reduction='mean')
        self.distance_estimate_weight = self.loss_config.get('distance_estimate_weight', 1.0)
        self.clustering_weight = self.loss_config.get('clustering_weight', 1.0)


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
            return nnMap


    def forward(self, result, segment_label, cluster_label):
        '''
        Mostly borrowed from uresnet_clustering.py
        '''
        num_gpus = len(segment_label)
        loss, accuracy = defaultdict(list), defaultdict(list)

        # Loop first over scaled feature maps
        for i_gpu in range(num_gpus):
            distance_estimate = result['distance_estimation'][i_gpu]
            dcoords = distance_estimate.get_spatial_locations()[:, :4]
            perm = np.lexsort((dcoords[:, 2], dcoords[:, 1], dcoords[:, 0], dcoords[:, 3]))
            distance_estimate = distance_estimate.features[perm]
            for depth in range(self.depth):

                batch_ids = segment_label[i_gpu][depth][:, 3].detach().cpu().int().numpy()
                batch_ids = np.unique(batch_ids)
                batch_size = len(batch_ids)

                embedding = result['cluster_feature'][i_gpu][depth]
                clabels_depth = cluster_label[i_gpu][depth]
                slabels_depth = segment_label[i_gpu][depth]

                coords = embedding.get_spatial_locations()[:, :4]
                perm = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0], coords[:, 3]))
                coords = coords[perm].float().cuda()
                feature_map = embedding.features[perm]

                loss_event = defaultdict(list)
                acc_event = defaultdict(list)
                for bidx in batch_ids:
                    batch_mask = coords[:, 3] == bidx
                    hypercoordinates = feature_map[batch_mask]
                    slabels_event = slabels_depth[batch_mask]
                    clabels_event = clabels_depth[batch_mask]
                    coords_event = coords[batch_mask]
                    if depth == 0:
                        distances_event = distance_estimate[batch_mask]

                    # Loop over semantic labels:
                    semantic_classes = slabels_event[:, -1].unique()
                    n_classes = len(semantic_classes)
                    loss_class = defaultdict(list)
                    acc_class = defaultdict(list)
                    acc_avg = 0.0
                    for class_ in semantic_classes:
                        k = int(class_)
                        class_mask = slabels_event[:, -1] == class_
                        embedding_class = hypercoordinates[class_mask]
                        cluster_class = clabels_event[class_mask][:, -2]
                        coords_class = coords_event[class_mask][:, :3]
                        if depth == 0:
                            distances_class = distances_event[class_mask]
                            acc = self.compute_heuristic_accuracy(embedding_class,
                                                                cluster_class)
                            accuracy['accuracy_{}'.format(k)].append(acc)
                            acc_avg += acc / n_classes
                            dMap = self.get_nn_map(embedding_class, cluster_class)
                            dloss = self.huber_loss(dMap, distances_class) * self.distance_estimate_weight
                            loss_class['loss_distance_estimation'].append(dloss)
                        # Clustering Loss
                        closs = self.combine(embedding_class,
                                             cluster_class,
                            intra_margin=self.intra_margins[depth],
                            inter_margin=self.inter_margins[depth],
                            coords=coords_class)
                        for key, val in closs.items():
                            loss_class[key].append(val)
                    if depth == 0:
                        accuracy['accuracy'].append(acc_avg)
                    for key, val in loss_class.items():
                        loss_event[key].append(sum(val) / len(val))
                for key, val in loss_event.items():
                    loss[key].append(sum(val) / len(val))

        res = {}

        for key, val in loss.items():
            res[key] = sum(val) / len(val)
        res['loss'] += res['loss_distance_estimation']
        res['loss_distance_estimation'] = float(res['loss_distance_estimation'])

        for key, val in accuracy.items():
            res[key] = sum(val) / len(val)

        return res


class DensityLoss(MultiScaleLoss2):
    '''
    Trainable Density Loss
    '''
    def __init__(self, cfg, name='spice_loss'):
        super(DensityLoss, self).__init__(cfg)
        self.density_radius = self.loss_config.get('density_radius', 0.1)
        self.sigma = self.density_radius / np.sqrt(2 * np.log(2))
        self.ally_density_weight = self.loss_config.get('ally_density_weight', 1.0)
        self.enemy_density_weight = self.loss_config.get('enemy_density_weight', 1.0)

    def intra_cluster_loss(self, features, labels, cluster_means,
                           ally_margin=0.5, enemy_margin=1.0):
        '''
        Intra-cluster loss, with per-voxel weighting and enemy loss.
        This variant of intra-cluster loss penalizes the distance
        from the centroid to its enemies in addition to pulling
        ally points towards the center.

        INPUTS:
            - ally_margin (float): centroid pulls all allied points
            inside this margin.
            - enemy_margin (float): centroid pushs all enemy points
            inside this margin.
        '''
        intra_loss = 0.0
        ally_loss, enemy_loss = 0.0, 0.0
        densityAloss, densityEloss = 0.0, 0.0
        n_clusters = len(cluster_means)
        cluster_labels = labels.unique(sorted=True)
        with torch.no_grad():
            dists = distance_matrix(features)
        for i, c in enumerate(cluster_labels):
            # Intra-Pull Loss
            index = (labels == c)
            allies = torch.norm(features[index] - cluster_means[i] + 1e-8,
                               p=self.norm, dim=1)
            allies = torch.clamp(allies - ally_margin, min=0)
            x = self.ally_weight * torch.mean(torch.pow(allies, 2))
            intra_loss += x
            ally_loss += float(x)
            # Ally Density Loss (Check that at least k allies exist)
            if sum(index) < 5:
                k = sum(index)
            else:
                k = 5
            _, idx_ally = dists[index, :][:, index].topk(k, dim=1, largest=False)
            x = torch.sum(torch.pow(
                features[index].unsqueeze(1) - features[index][idx_ally], 2), dim=2)
            x = torch.mean(torch.clamp(torch.exp(-x / (2 * self.sigma**2)), max=0.5))
            intra_loss += x
            densityAloss += float(x)
            if index.all():
                continue
            # Intra-Push Loss
            enemies = torch.norm(features[~index] - cluster_means[i] + 1e-8,
                    p=self.norm, dim=1)
            enemies = torch.clamp(enemy_margin - enemies, min=0)
            x = self.enemy_weight * torch.sum(torch.pow(enemies, 2))
            intra_loss += self.ally_density_weight * x
            enemy_loss += float(x)
            # Enemy Density Loss (Check that at least k enemies exist)
            if sum(~index) < 5:
                k = sum(~index)
            else:
                k = 5
            _, idx_enemy = dists[~index, :][:, ~index].topk(k, dim=1, largest=False)
            x = torch.sum(torch.pow(
                features[~index].unsqueeze(1) - features[~index][idx_enemy], 2), dim=2)
            x = torch.mean(torch.clamp(torch.exp(-x / (2 * self.sigma**2)), max=0.5))
            intra_loss += self.enemy_density_weight * x
            densityEloss += float(x)

        intra_loss /= n_clusters
        ally_loss /= n_clusters
        enemy_loss /= n_clusters
        densityAloss /= n_clusters
        densityEloss /= n_clusters
        return intra_loss, ally_loss, enemy_loss, densityAloss, densityEloss

    def combine(self, features, labels, **kwargs):
        '''
        Wrapper function for combining different components of the loss function.
        Inputs:
            features (torch.Tensor): pixel embeddings
            labels (torch.Tensor): ground-truth instance labels
        Returns:
            loss: combined loss, in most cases over a given semantic class.
        '''
        # Clustering Loss Hyperparameters
        # We allow changing the parameters at each computation in order
        # to alter the margins at each spatial resolution in multi-scale losses.
        ally_margin = kwargs.get('ally_margin', 0.5)
        enemy_margin = kwargs.get('enemy_margin', 1.0)
        inter_margin = kwargs.get('inter_margin', 1.5)
        intra_weight = kwargs.get('intra_weight', 1.0)
        inter_weight = kwargs.get('inter_weight', 1.0)
        reg_weight = kwargs.get('reg_weight', 0.001)

        c_means = self.find_cluster_means(features, labels)
        inter_loss = self.inter_cluster_loss(c_means, margin=inter_margin)
        intra_loss, ally_loss, enemy_loss, densityAloss, densityEloss = \
            self.intra_cluster_loss(features,
                                    labels,
                                    c_means,
                                    ally_margin=ally_margin,
                                    enemy_margin=enemy_margin)
        reg_loss = self.regularization(c_means)

        loss = intra_weight * intra_loss + inter_weight \
            * inter_loss + reg_weight * reg_loss

        res = {
            'loss': loss,
            'intra_loss': intra_weight * float(intra_loss),
            'inter_loss': inter_weight * float(inter_loss),
            'reg_loss': reg_weight * float(reg_loss),
            'ally_loss': intra_weight * float(ally_loss),
            'enemy_loss': intra_weight * float(enemy_loss),
            'ally_density_loss': densityAloss * self.ally_density_weight,
            'enemy_density_loss': densityEloss * self.enemy_density_weight
        }

        return res


class DensityDistanceEstimationLoss(DistanceEstimationLoss):

    def __init__(self, cfg, name='spice_loss'):
        super(DensityDistanceEstimationLoss, self).__init__(cfg)
        self.density_radius = self.loss_config.get('density_radius', 0.1)
        self.sigma = self.density_radius / np.sqrt(2 * np.log(2))
        self.ally_density_weight = self.loss_config.get('ally_density_weight', 1.0)
        self.enemy_density_weight = self.loss_config.get('enemy_density_weight', 1.0)
        self.ally_weight = self.loss_config.get('ally_weight', 1.0)
        self.enemy_weight = self.loss_config.get('enemy_weight', 1.0)

    def intra_cluster_loss(self, features, labels, cluster_means,
                           ally_margin=0.5, enemy_margin=1.0):
        '''
        Intra-cluster loss, with per-voxel weighting and enemy loss.
        This variant of intra-cluster loss penalizes the distance
        from the centroid to its enemies in addition to pulling
        ally points towards the center.

        INPUTS:
            - ally_margin (float): centroid pulls all allied points
            inside this margin.
            - enemy_margin (float): centroid pushs all enemy points
            inside this margin.
            - weight:
        '''
        intra_loss = 0.0
        ally_loss, enemy_loss = 0.0, 0.0
        n_clusters = len(cluster_means)
        cluster_labels = labels.unique(sorted=True)
        for i, c in enumerate(cluster_labels):
            index = (labels == c)
            allies = torch.norm(features[index] - cluster_means[i] + 1e-8,
                               p=self.norm, dim=1)
            allies = torch.clamp(allies - ally_margin, min=0)
            x = self.ally_weight * torch.mean(torch.pow(allies, 2))
            intra_loss += x
            ally_loss += float(x)
            if index.all():
                continue
            enemies = torch.norm(features[~index] - cluster_means[i] + 1e-8,
                    p=self.norm, dim=1)
            enemies = torch.clamp(enemy_margin - enemies, min=0)
            x = self.enemy_weight * torch.sum(torch.pow(enemies, 2))
            intra_loss += x
            enemy_loss += float(x)

        intra_loss /= n_clusters
        ally_loss /= n_clusters
        enemy_loss /= n_clusters
        return intra_loss, ally_loss, enemy_loss

    def intra_cluster_loss_density(self, features, labels, cluster_means,
                           ally_margin=0.5, enemy_margin=1.0):
        '''
        Intra-cluster loss, with per-voxel weighting and enemy loss.
        This variant of intra-cluster loss penalizes the distance
        from the centroid to its enemies in addition to pulling
        ally points towards the center.

        INPUTS:
            - ally_margin (float): centroid pulls all allied points
            inside this margin.
            - enemy_margin (float): centroid pushs all enemy points
            inside this margin.
        '''
        intra_loss = 0.0
        ally_loss, enemy_loss = 0.0, 0.0
        densityAloss, densityEloss = 0.0, 0.0
        n_clusters = len(cluster_means)
        cluster_labels = labels.unique(sorted=True)
        with torch.no_grad():
            dists = distance_matrix(features)
        for i, c in enumerate(cluster_labels):
            # Intra-Pull Loss
            index = (labels == c)
            allies = torch.norm(features[index] - cluster_means[i] + 1e-8,
                               p=self.norm, dim=1)
            allies = torch.clamp(allies - ally_margin, min=0)
            x = self.ally_weight * torch.mean(torch.pow(allies, 2))
            intra_loss += x
            ally_loss += float(x)
            # Ally Density Loss (Check that at least k allies exist)
            if sum(index) < 5:
                k = sum(index)
            else:
                k = 5
            _, idx_ally = dists[index, :][:, index].topk(k, dim=1, largest=False)
            x = torch.sum(torch.pow(
                features[index].unsqueeze(1) - features[index][idx_ally], 2), dim=2)
            x = torch.sum(torch.clamp(torch.exp(-x / (2 * self.sigma**2)) - 0.5, min=0))
            intra_loss += x
            densityAloss += float(x)
            if index.all():
                continue
            # Intra-Push Loss
            enemies = torch.norm(features[~index] - cluster_means[i] + 1e-8,
                    p=self.norm, dim=1)
            enemies = torch.clamp(enemy_margin - enemies, min=0)
            x = self.enemy_weight * torch.sum(torch.pow(enemies, 2))
            intra_loss += self.ally_density_weight * x
            enemy_loss += float(x)
            # Enemy Density Loss (Check that at least k enemies exist)
            if sum(~index) < 5:
                k = sum(~index)
            else:
                k = 5
            _, idx_enemy = dists[index, :][:, ~index].topk(k, dim=1, largest=False)
            x = torch.sum(torch.pow(
                features[index].unsqueeze(1) - features[~index][idx_enemy], 2), dim=2)
            x = torch.sum(torch.clamp(torch.exp(-x / (2 * self.sigma**2)) - 0.5, min=0))
            intra_loss += self.enemy_density_weight * x
            densityEloss += float(x)

        intra_loss /= n_clusters
        ally_loss /= n_clusters
        enemy_loss /= n_clusters
        densityAloss /= n_clusters
        densityEloss /= n_clusters
        return intra_loss, ally_loss, enemy_loss, densityAloss, densityEloss

    def combine(self, features, labels, **kwargs):
        '''
        Wrapper function for combining different components of the loss function.
        Inputs:
            features (torch.Tensor): pixel embeddings
            labels (torch.Tensor): ground-truth instance labels
        Returns:
            loss: combined loss, in most cases over a given semantic class.
        '''
        # Clustering Loss Hyperparameters
        # We allow changing the parameters at each computation in order
        # to alter the margins at each spatial resolution in multi-scale losses.
        ally_margin = kwargs.get('ally_margin', 0.5)
        enemy_margin = kwargs.get('enemy_margin', 1.0)
        inter_margin = kwargs.get('inter_margin', 1.5)
        intra_weight = kwargs.get('intra_weight', 1.0)
        inter_weight = kwargs.get('inter_weight', 1.0)
        reg_weight = kwargs.get('reg_weight', 0.001)

        c_means = self.find_cluster_means(features, labels)
        inter_loss = self.inter_cluster_loss(c_means, margin=inter_margin)
        intra_loss, ally_loss, enemy_loss = self.intra_cluster_loss(features,
                                    labels,
                                    c_means,
                                    ally_margin=ally_margin,
                                    enemy_margin=enemy_margin)
        reg_loss = self.regularization(c_means)

        loss = intra_weight * intra_loss + inter_weight \
            * inter_loss + reg_weight * reg_loss

        res = {
            'loss': loss,
            'intra_loss': intra_weight * float(intra_loss),
            'inter_loss': inter_weight * float(inter_loss),
            'reg_loss': reg_weight * float(reg_loss),
            'ally_loss': intra_weight * float(ally_loss),
            'enemy_loss': intra_weight * float(enemy_loss)
        }

        return res

    def combine_density(self, features, labels, **kwargs):
        '''
        Wrapper function for combining different components of the loss function.
        Inputs:
            features (torch.Tensor): pixel embeddings
            labels (torch.Tensor): ground-truth instance labels
        Returns:
            loss: combined loss, in most cases over a given semantic class.
        '''
        # Clustering Loss Hyperparameters
        # We allow changing the parameters at each computation in order
        # to alter the margins at each spatial resolution in multi-scale losses.
        ally_margin = kwargs.get('ally_margin', 0.5)
        enemy_margin = kwargs.get('enemy_margin', 1.0)
        inter_margin = kwargs.get('inter_margin', 1.5)
        intra_weight = kwargs.get('intra_weight', 1.0)
        inter_weight = kwargs.get('inter_weight', 1.0)
        reg_weight = kwargs.get('reg_weight', 0.001)

        c_means = self.find_cluster_means(features, labels)
        inter_loss = self.inter_cluster_loss(c_means, margin=inter_margin)
        intra_loss, ally_loss, enemy_loss, densityAloss, densityEloss = \
            self.intra_cluster_loss_density(features,
                                    labels,
                                    c_means,
                                    ally_margin=ally_margin,
                                    enemy_margin=enemy_margin)
        reg_loss = self.regularization(c_means)

        loss = intra_weight * intra_loss + inter_weight \
            * inter_loss + reg_weight * reg_loss

        res = {
            'loss': loss,
            'intra_loss': intra_weight * float(intra_loss),
            'inter_loss': inter_weight * float(inter_loss),
            'reg_loss': reg_weight * float(reg_loss),
            'ally_loss': intra_weight * float(ally_loss),
            'enemy_loss': intra_weight * float(enemy_loss),
            'ally_density_loss': densityAloss * self.ally_density_weight,
            'enemy_density_loss': densityEloss * self.enemy_density_weight
        }

        return res

    def forward(self, result, segment_label, cluster_label):
        '''
        Mostly borrowed from uresnet_clustering.py
        '''
        num_gpus = len(segment_label)
        loss, accuracy = defaultdict(list), defaultdict(list)

        # Loop first over scaled feature maps
        for i_gpu in range(num_gpus):
            distance_estimate = result['distance_estimation'][i_gpu]
            dcoords = distance_estimate.get_spatial_locations()[:, :4]
            perm = np.lexsort((dcoords[:, 2], dcoords[:, 1], dcoords[:, 0], dcoords[:, 3]))
            distance_estimate = distance_estimate.features[perm]
            for depth in range(self.depth):

                batch_ids = segment_label[i_gpu][depth][:, 3].detach().cpu().int().numpy()
                batch_ids = np.unique(batch_ids)
                batch_size = len(batch_ids)

                embedding = result['cluster_feature'][i_gpu][depth]
                clabels_depth = cluster_label[i_gpu][depth]
                slabels_depth = segment_label[i_gpu][depth]

                coords = embedding.get_spatial_locations()[:, :4]
                perm = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0], coords[:, 3]))
                coords = coords[perm].float().cuda()
                feature_map = embedding.features[perm]

                loss_event = defaultdict(list)
                acc_event = defaultdict(list)
                for bidx in batch_ids:
                    batch_mask = coords[:, 3] == bidx
                    hypercoordinates = feature_map[batch_mask]
                    slabels_event = slabels_depth[batch_mask]
                    clabels_event = clabels_depth[batch_mask]
                    coords_event = coords[batch_mask]
                    if depth == 0:
                        distances_event = distance_estimate[batch_mask]

                    # Loop over semantic labels:
                    semantic_classes = slabels_event[:, -1].unique()
                    n_classes = len(semantic_classes)
                    loss_class = defaultdict(list)
                    acc_class = defaultdict(list)
                    acc_avg = 0.0
                    for class_ in semantic_classes:
                        k = int(class_)
                        class_mask = slabels_event[:, -1] == class_
                        embedding_class = hypercoordinates[class_mask]
                        cluster_class = clabels_event[class_mask][:, -2]
                        coords_class = coords_event[class_mask][:, :3]
                        if depth == 0:
                            distances_class = distances_event[class_mask]
                            acc = self.compute_heuristic_accuracy(embedding_class,
                                                                cluster_class)
                            accuracy['accuracy_{}'.format(k)].append(acc)
                            acc_avg += acc / n_classes
                            dMap = self.get_nn_map(embedding_class, cluster_class)
                            dloss = self.huber_loss(dMap, distances_class) * self.distance_estimate_weight
                            loss_class['distance_estimation'].append(dloss)
                            closs = self.combine_density(embedding_class,
                                                         cluster_class,
                                                         intra_margin=self.intra_margins[depth],
                                                         inter_margin=self.inter_margins[depth])
                        # Clustering Loss
                        else:
                            closs = self.combine(embedding_class,
                                                cluster_class,
                                intra_margin=self.intra_margins[depth],
                                inter_margin=self.inter_margins[depth])
                        for key, val in closs.items():
                            loss_class[key].append(val)
                    if depth == 0:
                        accuracy['accuracy'].append(acc_avg)
                    for key, val in loss_class.items():
                        loss_event[key].append(sum(val) / len(val))
                for key, val in loss_event.items():
                    loss[key].append(sum(val) / len(val))

        res = {}

        for key, val in loss.items():
            res[key] = sum(val) / len(val)
        res['loss'] += res['distance_estimation']
        res['distance_estimation'] = float(res['distance_estimation'])

        for key, val in accuracy.items():
            res[key] = sum(val) / len(val)

        for key, val in res.items():
            print("{}: {}".format(key, val))

        return res
