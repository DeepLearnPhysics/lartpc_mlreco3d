import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import sparseconvnet as scn

from collections import defaultdict
from mlreco.cluster_cnn.utils import distance_matrix, pairwise_distances
from .single_layers import DiscriminativeLoss


class NeighborLoss(MultiScaleLoss):
    '''
    Distance to Neighboring Ally and Enemy Loss

    NOTE: This function has HUGE memory footprint and training
    will crash under current implementation.
    '''
    def __init__(self, cfg):
        super(NeighborLoss, self).__init__(cfg)
        self.loss_config = cfg['modules']['clustering_loss']

        # Huber Loss for Team Loss
        self.huber_loss = torch.nn.SmoothL1Loss(reduction='mean')

        # Density Loss Parameters
        self.estimate_teams = self.loss_config.get('estimate_teams', False)
        self.ally_est_weight = self.loss_config.get('ally_est_weight', 1.0)
        self.enemy_est_weight = self.loss_config.get('enemy_est_weight', 1.0)

        # Minimum Required Distance^2 to Closest Ally
        self.targetAlly = self.loss_config.get('target_friends', 1.0)
        # Maximum Required Distance^2 to Closest Enemy
        self.targetEnemy = self.loss_config.get('target_enemies', 5.0)

        self.ally_margins = self.loss_config.get('ally_margins', 
            [self.targetAlly / 2**i for i in range(self.num_strides)])
        self.enemy_margins = self.loss_config.get('enemy_margins',
            [self.targetEnemy / 2**i for i in range(self.num_strides)])

        self.ally_weight = self.loss_config.get('ally_weight', 1.0)
        self.enemy_weight = self.loss_config.get('enemy_weight', 1.0)
        self.affinity_weight = self.loss_config.get('affinity_weight', 1.0)
        self.clustering_weight = self.loss_config.get('clustering_weight', 1.0)


    def compute_neighbor_loss(self, embedding_class, cluster_class,
            ally_margin=0.25, enemy_margin=10.0):
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
        loss = 0.0
        ally_loss, enemy_loss = 0.0, 0.0
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
            lossA = self.ally_weight *  torch.mean(
                torch.clamp(allies - ally_margin, min=0))
            loss += lossA
            ally_loss += float(lossA)
            del lossA
            if index.all(): 
                # Skip if there are no enemies
                continue
            enemies, _ = torch.min(dist[index, :][:, ~index], dim=1)
            lossE = self.enemy_weight * torch.mean(
                torch.clamp(enemy_margin - enemies, min=0))
            loss += lossE
            enemy_loss += float(lossE)
            del lossE
        
        loss /= num_clusters
        ally_loss /= num_clusters
        enemy_loss /= num_clusters
        return loss, ally_loss, enemy_loss


    def forward(self, result, segment_label, cluster_label):
        '''
        Mostly borrowed from uresnet_clustering.py
        '''
        data = ForwardData()
        num_gpus = len(segment_label)
        loss = 0.0
        clustering_loss, affinity_loss = 0.0, 0.0
        count = 0

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
                coords = coords[perm]
                feature_map = embedding.features[perm]

                for bidx in batch_ids:
                    batch_mask = coords[:, 3] == bidx
                    hypercoordinates = feature_map[batch_mask]
                    slabels_event = slabels_depth[batch_mask]
                    clabels_event = clabels_depth[batch_mask]
                    coords_event = coords[batch_mask]

                    # Loop over semantic labels:
                    semantic_classes = slabels_event[:, -1].unique()
                    n_classes = len(semantic_classes)
                    for class_ in semantic_classes:
                        k = int(class_)
                        class_mask = slabels_event[:, -1] == class_
                        embedding_class = hypercoordinates[class_mask]
                        cluster_class = clabels_event[class_mask][:, -1]
                        # Clustering Loss
                        acc = self.compute_heuristic_accuracy(embedding_class,
                                                              cluster_class)
                        closs = self.combine(embedding_class,
                                             cluster_class,
                            intra_margin=self.intra_margins[depth],
                            inter_margin=self.inter_margins[depth])
                        dloss, dlossF, dlossE = self.compute_neighbor_loss(
                            embedding_class, cluster_class,
                            ally_margin=self.ally_margins[depth],
                            enemy_margin=self.enemy_margins[depth])
                        # Informations to be saved in log file (loss/accuracy). 
                        data.update_mean('accuracy', acc)
                        data.update_mean('intra_loss', closs['intra_loss'])
                        data.update_mean('inter_loss', closs['inter_loss'])
                        data.update_mean('reg_loss', closs['reg_loss'])
                        data.update_mean('ally_loss', dlossF)
                        data.update_mean('enemy_loss', dlossE)
                        data.update_mean('accuracy_{}'.format(class_), acc)
                        data.update_mean('intra_loss_{}'.format(class_), closs['intra_loss'])
                        data.update_mean('inter_loss_{}'.format(class_), closs['inter_loss'])
                        data.update_mean('ally_loss_{}'.format(class_), dlossF)
                        data.update_mean('enemy_loss_{}'.format(class_), dlossE)
                        clustering_loss += self.clustering_weight * closs['loss']
                        affinity_loss += self.affinity_weight * dloss
                        count += 1

        res = data.as_dict()
        res['loss'] = (clustering_loss + affinity_loss) / count
        return res


class EnhancedEmbeddingLoss(MultiScaleLoss):

    def __init__(self, cfg, name='clustering_loss'):
        super(EnhancedEmbeddingLoss, self).__init__(cfg)
        self.spatial_size = self.loss_config.get('spatial_size', 512)
        self.ally_weight = self.loss_config.get('ally_weight', 1.0)
        self.enemy_weight = self.loss_config.get('enemy_weight', 0.0)
        self.attention_kernel = self.loss_config.get('attention_kernel', 1)
        self.compute_enemy_loss = self.loss_config.get('compute_enemy_loss', True)
        self.compute_attention_weights = self.loss_config.get('compute_attention_weights', True)
        if self.attention_kernel == 0:
            self.kernel_func = lambda x: 1.0 + torch.exp(-x)
        elif self.attention_kernel == 1:
            self.kernel_func = lambda x: 2.0 / (1 + torch.exp(-x))
        else:
            raise ValueError('Invalid weighting kernel function mode.')

    def compute_attention_weight(self, coords, labels):
        '''
        Computes the per-voxel intra-cluster loss weights from
        distances to cluster centroids in coordinate space.

        INPUTS:
            - coords (N x 2,3): spatial coordinates of N voxels
            in image space.
            - labels (N x 1): cluster labels for N voxels.

        RETURNS:
            - weights (N x 1): computed attention weights for 
            N voxels.
        '''
        with torch.no_grad():
            weights = torch.zeros(labels.shape)
            if torch.cuda.is_available():
                weights = weights.cuda()
            centroids = self.find_cluster_means(coords, labels)
            cluster_labels = labels.unique(sorted=True)
            for i, c in enumerate(cluster_labels):
                index = labels == c
                dists = torch.norm(coords[index] - centroids[i] + 1e-8,
                                    p=self.norm, dim=1) / self.spatial_size
                weights[index] = self.kernel_func(dists)
        return weights


    def intra_cluster_loss(self, features, labels, cluster_means,
                           ally_margin=0.5, enemy_margin=1.0, weight=1.0):
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
            x = self.ally_weight * torch.mean(weight[index] * torch.pow(allies, 2))
            intra_loss += x
            ally_loss += float(x)
            if index.all() or self.compute_enemy_loss:
                continue
            enemies = torch.norm(features[~index] - cluster_means[i] + 1e-8,
                    p=self.norm, dim=1)
            enemies = torch.clamp(enemy_margin - enemies, min=0)
            x = self.enemy_weight * torch.sum(torch.pow(enemies, 2))
            intra_loss += x
            enemy_loss += x

        intra_loss /= n_clusters
        ally_loss /= n_clusters
        enemy_loss / n_clusters
        return intra_loss, ally_loss, enemy_loss


    # def inter_cluster_loss(self, cluster_means, margin=1.5):
    #     '''
    #     Inter-cluster loss, vectorized with BLAS/LAPACK distance
    #     matrix computation.
    #
    #     NOTE: This function causes NaNs during backward. 
    #     '''
    #     inter_loss = 0.0
    #     n_clusters = len(cluster_means)
    #     if n_clusters < 2:
    #         # Inter-cluster loss is zero if there only one instance exists for
    #         # a semantic label.
    #         return 0.0
    #     else:
    #         inter_loss = torch.pow(torch.clamp(2.0 * margin - \
    #             torch.sqrt(distance_matrix(cluster_means) + 1e-8), min=0), 2)
    #         inter_loss = torch.triu(inter_loss, diagonal=1)
    #         inter_loss = 2 * torch.sum(inter_loss) / float((n_clusters - 1) * n_clusters)
    #         return inter_loss


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
        attention_weights = kwargs.get('attention_weights', 1.0)

        c_means = self.find_cluster_means(features, labels)
        inter_loss = self.inter_cluster_loss(c_means, margin=inter_margin)
        intra_loss, ally_loss, enemy_loss = self.intra_cluster_loss(features,
                                           labels,
                                           c_means,
                                           ally_margin=ally_margin,
                                           enemy_margin=enemy_margin,
                                           weight=attention_weights)
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


    def combine_multiclass(self, features, slabels, clabels,
            attention_weight=1.0, **kwargs):
        '''
        Wrapper function for combining different components of the loss, 
        in particular when clustering must be done PER SEMANTIC CLASS. 

        NOTE: When there are multiple semantic classes, we compute the DLoss
        by first masking out by each semantic segmentation (ground-truth/prediction)
        and then compute the clustering loss over each masked point cloud. 

        INPUTS: 
            features (torch.Tensor): pixel embeddings
            slabels (torch.Tensor): semantic labels
            clabels (torch.Tensor): group/instance/cluster labels

        OUTPUT:
            loss_segs (list): list of computed loss values for each semantic class. 
            loss[i] = computed DLoss for semantic class <i>. 
            acc_segs (list): list of computed clustering accuracy for each semantic class. 
        '''
        loss, accuracy = defaultdict(float), defaultdict(float)
        semantic_classes = slabels.unique()
        nClasses = len(semantic_classes)
        avg_acc = 0.0
        compute_accuracy = kwargs.get('compute_accuracy', False)
        for sc in semantic_classes:
            index = (slabels == sc)
            num_clusters = len(clabels[index].unique())
            loss_blob = self.combine(features[index], clabels[index],
                attention_weight=attention_weight[index], **kwargs)
            loss['loss'] += loss_blob['loss'] / nClasses
            loss['intra_loss'] += loss_blob['intra_loss'] / nClasses
            loss['inter_loss'] += loss_blob['inter_loss'] / nClasses
            loss['reg_loss'] += loss_blob['reg_loss'] / nClasses
            loss['ally_loss'] += loss_blob['ally_loss'] / nClasses
            loss['enemy_loss'] += loss_blob['enemy_loss'] / nClasses
            if compute_accuracy:
                acc = self.compute_heuristic_accuracy(features[index], clabels[index])
                accuracy['accuracy_{}'.format(sc.item())] = acc
                avg_acc += acc / nClasses
        accuracy['accuracy'] = avg_acc
        return loss, accuracy


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
        loss = ForwardData()
        accuracy = ForwardData()

        coords = embedding_scn.get_spatial_locations()
        coords_np = coords.numpy()
        perm = np.lexsort((coords_np[:, 2], coords_np[:, 1],
                           coords_np[:, 0], coords_np[:, 3]))
        embedding = embedding_scn.features[perm]
        coords = coords[perm].float()
        if torch.cuda.is_available():
            coords = coords.cuda()

        for bidx in batch_idx:
            index = slabels[:, 3].int() == bidx
            embedding_batch = embedding[index]
            slabels_batch = slabels[index][:, -1]
            clabels_batch = clabels[index][:, -1]
            coords_batch = coords[index][:, :3]
            attention_weights = self.compute_attention_weight(coords_batch, clabels_batch)
            loss_dict, acc_dict = self.combine_multiclass(
                embedding_batch, slabels_batch, clabels_batch,
                attention_weights=attention_weights, **kwargs)
            loss.update_dict(loss_dict)
            accuracy.update_dict(acc_dict)

        return loss.as_dict(), accuracy.as_dict()


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

        data = ForwardData()
        for i_gpu in range(len(semantic_labels)):
            batch_idx = semantic_labels[i_gpu][0][:, 3].detach().cpu().int().numpy()
            batch_idx = np.unique(batch_idx)
            batch_size = len(batch_idx)
            # Summing clustering loss over layers.
            for i, em in enumerate(out['cluster_feature'][i_gpu]):
                # Get scaled margins for each layer.
                delta_var, delta_dist = self.intra_margins[i], self.inter_margins[i]
                # Compute accuracy at last layer.
                if i == 0:
                    em = out['final_embedding'][i_gpu]
                    loss_i, acc_i = self.compute_loss_layer(
                        em, semantic_labels[i_gpu][i], group_labels[i_gpu][i], batch_idx,
                        delta_var=delta_var, delta_dist=delta_dist,
                        compute_accuracy=True)
                    data.update_dict(loss_i)
                    data.update_dict(acc_i)
                else:
                    loss_i, acc_i = self.compute_loss_layer(
                        em, semantic_labels[i_gpu][i], group_labels[i_gpu][i], batch_idx,
                        delta_var=delta_var, delta_dist=delta_dist,
                        compute_accuracy=False)
                    data.update_dict(loss_i)

        res = data.as_dict()
        return res