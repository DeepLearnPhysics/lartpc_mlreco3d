import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import sparseconvnet as scn

from .misc import *
from collections import defaultdict


class SPICELoss(nn.Module):
    '''
    Loss function for Sparse Spatial Embeddings Model, with fixed
    centroids and symmetric gaussian kernels.
    '''
    def __init__(self, cfg, name='spice_loss'):
        super(SPICELoss, self).__init__()
        self.loss_config = cfg[name]
        self.seediness_weight = self.loss_config.get('seediness_weight', 1.0)
        self.embedding_weight = self.loss_config.get('embedding_weight', 1.0)
        self.smoothing_weight = self.loss_config.get('smoothing_weight', 1.0)
        self.spatial_size = self.loss_config.get('spatial_size', 768)

        self.mask_loss_fn = self.loss_config.get('mask_loss_fn', 'BCE')
        self.seed_loss_fn = self.loss_config.get('seed_loss_fn', 'L1')

        if self.mask_loss_fn == 'BCE':
            self.mask_loss = nn.BCEWithLogitsLoss(reduction='none')
        elif self.mask_loss_fn == 'lovasz_hinge':
            self.mask_loss = LovaszHingeLoss()
        elif self.mask_loss_fn == 'focal':
            raise NotImplementedError
        else:
            raise ValueError(
            'Invalid loss scheme: {}'.format(self.loss_scheme))

        # L2 Loss for Seediness
        if self.seed_loss_fn == 'L1':
            self.seed_loss = torch.nn.L1Loss(reduction='mean')
        elif self.seed_loss_fn == 'L1':
            self.seed_loss = torch.nn.MSELoss(reduction='mean')
        elif self.seed_loss_fn == 'huber':
            self.seed_loss = torch.nn.SmoothL1Loss(reduction='mean')
        else:
            raise ValueError(
            'Invalid loss scheme: {}'.format(self.loss_scheme))

    def find_cluster_means(self, features, labels):
        '''
        For a given image, compute the centroids mu_c for each
        cluster label in the embedding space.
        Inputs:
            features (torch.Tensor): the pixel embeddings, shape=(N, d) where
            N is the number of pixels and d is the embedding space dimension.
            labels (torch.Tensor): ground-truth group labels, shape=(N, )
        Returns:
            cluster_means (torch.Tensor): (n_c, d) tensor where n_c is the number of
            distinct instances. Each row is a (1,d) vector corresponding to
            the coordinates of the i-th centroid.
        '''
        cluster_means = find_cluster_means(features, labels)
        return cluster_means

    def get_per_class_probabilities(self, embeddings, margins, labels, eps=1e-6):
        '''
        Computes binary foreground/background loss.
        '''
        device = embeddings.device
        n = labels.shape[0]
        centroids = self.find_cluster_means(embeddings, labels)
        sigma = scatter_mean(margins.squeeze(), labels)
        num_clusters = labels.unique().shape[0]

        # Compute spatial term
        em = embeddings[:, None, :]
        centroids = centroids[None, :, :]
        sqdists = ((em - centroids)**2).sum(-1)

        p = sqdists / (2.0 * sigma.view(1, -1)**2)
        p = torch.clamp(torch.exp(-p), min=eps, max=1-eps)
        logits = logit_fn(p, eps=eps)
        eye = torch.eye(len(labels.unique()), dtype=torch.float32, device=device)
        targets = eye[labels]
        loss_tensor = self.mask_loss(logits, targets)
        loss = loss_tensor.mean(dim=0).mean()
        with torch.no_grad():
            acc = iou_batch(logits > 0, targets.bool())
        smoothing_loss = margin_smoothing_loss(margins.squeeze(), sigma.detach(), labels, margin=0)
        p = torch.gather(p, 1, labels.view(-1, 1))
        return loss, smoothing_loss, p.squeeze(), acc


    def combine_multiclass(self, embeddings, margins, seediness, slabels, clabels, coords):
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
        loss = defaultdict(list)
        accuracy = defaultdict(float)
        semantic_classes = slabels.unique()
        #print(semantic_classes)
        for sc in semantic_classes:
            if int(sc) == 4:
                continue
            index = (slabels == sc)
            clabels_unique, _ = unique_label_torch(clabels[index])
            mask_loss, smoothing_loss, probs, acc = self.get_per_class_probabilities(
                embeddings[index], margins[index], clabels_unique)
            prob_truth = probs.detach()
            seed_loss = self.seed_loss(prob_truth, seediness[index].squeeze(1))
            total_loss = self.embedding_weight * mask_loss \
                       + self.seediness_weight * seed_loss \
                       + self.smoothing_weight * smoothing_loss
            loss['loss'].append(total_loss)
            loss['mask_loss'].append(float(self.embedding_weight * mask_loss))
            loss['seed_loss'].append(float(self.seediness_weight * seed_loss))
            loss['smoothing_loss'].append(float(self.smoothing_weight * smoothing_loss))
            loss['mask_loss_{}'.format(int(sc))].append(float(mask_loss))
            loss['seed_loss_{}'.format(int(sc))].append(float(seed_loss))
            accuracy['accuracy_{}'.format(int(sc))] = acc

        return loss, accuracy

    def forward(self, out, segment_label, group_label):

        num_gpus = len(segment_label)
        loss = defaultdict(list)
        accuracy = defaultdict(list)

        for i in range(num_gpus):
            slabels = segment_label[i][:, -1]
            #coords = segment_label[i][:, :3].float()
            #if torch.cuda.is_available():
            #    coords = coords.cuda()
            slabels = slabels.int()
            clabels = group_label[i][:, -1]
            batch_idx = segment_label[i][:, 3]
            embedding = out['embeddings'][i]
            seediness = out['seediness'][i]
            margins = out['margins'][i]
            nbatch = batch_idx.unique().shape[0]

            for bidx in batch_idx.unique(sorted=True):
                embedding_batch = embedding[batch_idx == bidx]
                slabels_batch = slabels[batch_idx == bidx]
                clabels_batch = clabels[batch_idx == bidx]
                seed_batch = seediness[batch_idx == bidx]
                margins_batch = margins[batch_idx == bidx]

                loss_class, acc_class = self.combine_multiclass(
                    embedding_batch, margins_batch,
                    seed_batch, slabels_batch, clabels_batch)
                for key, val in loss_class.items():
                    loss[key].append(sum(val) / len(val))
                for s, acc in acc_class.items():
                    accuracy[s].append(acc)
                acc = sum(acc_class.values()) / len(acc_class.values())
                accuracy['accuracy'].append(acc)

        loss_avg = {}
        acc_avg = defaultdict(float)

        for key, val in loss.items():
            loss_avg[key] = sum(val) / len(val)
        for key, val in accuracy.items():
            acc_avg[key] = sum(val) / len(val)

        res = {}
        res.update(loss_avg)
        res.update(acc_avg)

        return res


class SPICEInterLoss(SPICELoss):

    def __init__(self, cfg, name='spice_loss'):
        super(SPICEInterLoss, self).__init__(cfg, name)
        self.inter_weight = self.loss_config.get('inter_weight', 1.0)
        self.inter_margin = self.loss_config.get('inter_margin', 0.2)
        self.norm = 2
        self._min_voxels = self.loss_config.get('min_voxels', 2)


    def regularization(self, cluster_means):
        '''
        Implementation of regularization loss in Discriminative Loss
        Inputs:
            cluster_means (torch.Tensor): output from find_cluster_means
        Returns:
            reg_loss (float): computed regularization loss (see paper).
        '''
        reg_loss = regularization_loss(cluster_means)
        return reg_loss


    def inter_cluster_loss(self, cluster_means, margin=0.2):
        '''
        Implementation of distance loss in Discriminative Loss.
        Inputs:
            cluster_means (torch.Tensor): output from find_cluster_means
            margin (float/int): the magnitude of the margin delta_d in the paper.
            Think of it as the distance between each separate clusters in
            embedding space.
        Returns:
            inter_loss (float): computed cross-centroid distance loss (see paper).
            Factor of 2 is included for proper normalization.
        '''
        inter_loss = inter_cluster_loss(cluster_means, margin=margin)
        return inter_loss


    def get_per_class_probabilities(self, embeddings, margins, labels, eps=1e-6):
        '''
        Computes binary foreground/background loss.
        '''
        device = embeddings.device
        n = labels.shape[0]
        centroids = self.find_cluster_means(embeddings, labels)
        sigma = scatter_mean(margins.squeeze(), labels)
        num_clusters = labels.unique().shape[0]
        inter_loss = self.inter_cluster_loss(centroids, margin=self.inter_margin)

        # Compute spatial term
        em = embeddings[:, None, :]
        centroids = centroids[None, :, :]
        sqdists = ((em - centroids)**2).sum(-1)

        p = sqdists / (2.0 * sigma.view(1, -1)**2)
        p = torch.clamp(torch.exp(-p), min=eps, max=1-eps)
        logits = logit_fn(p, eps=eps)
        eye = torch.eye(len(labels.unique()), dtype=torch.float32, device=device)
        targets = eye[labels]
        loss_tensor = self.mask_loss(logits, targets)
        loss = loss_tensor.mean(dim=0).mean()
        with torch.no_grad():
            acc = iou_batch(logits > 0, targets.bool())
        smoothing_loss = margin_smoothing_loss(margins.squeeze(), sigma.detach(), labels, margin=0)
        p = torch.gather(p, 1, labels.view(-1, 1))
        loss += inter_loss
        return loss, smoothing_loss, float(inter_loss), p.squeeze(), acc

    def combine_multiclass(self, embeddings, margins, seediness, slabels, clabels):
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
        loss = defaultdict(list)
        accuracy = defaultdict(float)
        semantic_classes = slabels.unique()
        for sc in semantic_classes:
            if int(sc) == 4:
                continue
            index = (slabels == sc)
            if len(embeddings[index]) < self._min_voxels:
                continue
            clabels_unique, _ = unique_label_torch(clabels[index])
            mask_loss, smoothing_loss, inter_loss, probs, acc = self.get_per_class_probabilities(
                embeddings[index], margins[index], clabels_unique)
            prob_truth = probs.detach()
            seed_loss = self.seed_loss(prob_truth, seediness[index].squeeze(1))
            total_loss = self.embedding_weight * mask_loss \
                       + self.seediness_weight * seed_loss \
                       + self.smoothing_weight * smoothing_loss
            loss['loss'].append(total_loss)
            loss['mask_loss'].append(
                float(self.embedding_weight * mask_loss))
            loss['seed_loss'].append(
                float(self.seediness_weight * seed_loss))
            loss['smoothing_loss'].append(
                float(self.smoothing_weight * smoothing_loss))
            loss['inter_loss'].append(
                float(self.inter_weight * inter_loss))
            loss['mask_loss_{}'.format(int(sc))].append(float(mask_loss))
            loss['seed_loss_{}'.format(int(sc))].append(float(seed_loss))
            accuracy['accuracy_{}'.format(int(sc))] = acc

        return loss, accuracy
