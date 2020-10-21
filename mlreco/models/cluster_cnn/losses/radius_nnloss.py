import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import sparseconvnet as scn

from .lovasz import mean, lovasz_hinge_flat, StableBCELoss, iou_binary
from .misc import FocalLoss, WeightedFocalLoss
from collections import defaultdict
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score as ari

from torch_cluster import knn, radius
from torch_scatter import scatter_mean, scatter_add


class DensityBasedNNLoss(torch.nn.modules.loss._Loss):

    def __init__(self, cfg, name='density_loss'):
        super(DensityBasedNNLoss, self).__init__()
        self.eps1 = 0.2
        self.eps2 = 1.0
        self.minpts = 10
        self.ally_loss_weight = 1.0
        self.enemy_loss_weight = 1.0
        self.dbscan = DBSCAN(eps=self.eps1, min_samples=self.minpts)

    def radius_neighbor_loss(self, features, labels, minPoints=5, eps1=1.999, eps2=1.999, compute_accuracy=False):

        loss = []

        ally_loss_list, enemy_loss_list = [], []

        for c in labels.unique():

            allies = features[labels == c]
            enemies = features[labels != c]

            if allies.shape[0] < self.minpts:
                continue

            index = knn(allies, allies, minPoints)
            dist = torch.norm(allies[index[0, :]] - allies[index[1, :]], dim=1)
            dist = dist[index[0, :] != index[1, :]]
            ally_loss = torch.pow(dist, 2)
            scatter_index = index[0, :][index[0, :] != index[1, :]]
            ally_loss = scatter_add(ally_loss, scatter_index)
            ally_len = ally_loss.shape[0]
            ally_loss = torch.mean(ally_loss)
            ally_loss_list.append(float(ally_loss))

            if enemies.shape[0] == 0:
                loss.append(self.ally_loss_weight * ally_loss)
                continue

            index = knn(enemies, allies, minPoints)
            dist = torch.norm(allies[index[0, :]] - enemies[index[1, :]], dim=1)
            enemy_loss = torch.clamp(1.0 - torch.exp(-dist**2), min=0.001, max=1-0.001)
            enemy_loss = -torch.log(enemy_loss)
            scatter_index = index[0, :]
            enemy_loss = scatter_add(enemy_loss, scatter_index)
            enemy_len = enemy_loss.shape[0]
            assert(ally_len == enemy_len)
            enemy_loss = torch.mean(enemy_loss)
            enemy_loss_list.append(float(enemy_loss))

            l = self.ally_loss_weight * ally_loss + \
                self.enemy_loss_weight * enemy_loss

            loss.append(l)

        if len(loss) == 0:
            return 0.0, 0.0, 0.0, 0.0

        loss = sum(loss) / len(loss)

        ally_loss, enemy_loss = 0, 0

        if len(ally_loss_list) > 0:
            ally_loss = sum(ally_loss_list) / len(ally_loss_list)
        if len(enemy_loss_list) > 0:
            enemy_loss = sum(enemy_loss_list) / len(enemy_loss_list)

        pred = self.dbscan.fit_predict(features.detach().cpu().numpy())
        acc = ari(pred, labels.cpu().numpy())

        return loss, acc, ally_loss, enemy_loss


    def combine_multiclass(self, features, slabels, clabels, **kwargs):
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
        minpts = kwargs['minPoints']
        eps1 = kwargs['eps1']
        eps2 = kwargs['eps2']

        loss, accuracy = {}, {}
        total_loss = []
        semantic_classes = slabels.unique()
        for sc in semantic_classes:
            if (int(sc) == 4):
                continue
            index = (slabels == sc)
            l, acc, ally_loss, enemy_loss = self.radius_neighbor_loss(features[index], clabels[index], minpts, eps1, eps2)
            total_loss.append(l)
            loss['ally_loss'] = ally_loss
            loss['enemy_loss'] = enemy_loss
            loss['loss_{}'.format(int(sc))] = float(l)
            accuracy['acc_{}'.format(int(sc))] = float(acc)

        loss['loss'] = sum(total_loss) / len(total_loss)

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
        num_gpus = len(semantic_labels)
        loss = defaultdict(list)
        accuracy = defaultdict(list)

        for i in range(num_gpus):
            slabels = semantic_labels[i][:, -1]
            slabels = slabels.int()
            clabels = group_labels[i][:, -1]
            batch_idx = semantic_labels[i][:, 3]
            embedding = out['embeddings'][i]
            nbatch = batch_idx.unique().shape[0]

            for bidx in batch_idx.unique(sorted=True):
                embedding_batch = embedding[batch_idx == bidx]
                slabels_batch = slabels[batch_idx == bidx]
                clabels_batch = clabels[batch_idx == bidx]

                loss_dict, acc_dict = self.combine_multiclass(
                    embedding_batch, slabels_batch, clabels_batch, minPoints=self.minpts, eps1=self.eps1, eps2=self.eps2)

                for key, val in loss_dict.items():
                    loss[key].append(val)
                for s, acc in acc_dict.items():
                    accuracy[s].append(acc)
                acc = sum(acc_dict.values()) / len(acc_dict.values())
                accuracy['accuracy'].append(acc)

        loss_avg = {}
        acc_avg = defaultdict(float)

        for key, val in loss.items():
            loss_avg[key] = sum(val) / len(val)

        print(loss_avg)

        for key, val in accuracy.items():
            acc_avg[key] = sum(val) / len(val)

        res = {}
        res.update(loss_avg)
        res.update(acc_avg)

        return res
