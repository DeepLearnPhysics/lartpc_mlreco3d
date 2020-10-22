import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import sparseconvnet as scn

from .lovasz import mean, lovasz_hinge_flat, StableBCELoss, iou_binary
from .misc import *
from collections import defaultdict
from torch_scatter import scatter_mean


class OccuSegLoss(nn.Module):
    '''
    Loss function for Sparse Spatial Embeddings Model, with fixed
    centroids and symmetric gaussian kernels.
    '''
    def __init__(self, cfg, name='spice_loss'):
        super(OccuSegLoss, self).__init__()
        self.loss_config = cfg[name]
        self.batch_column = self.loss_config.get('batch_column', 3)

        self.ft_interloss = self.loss_config.get('ft_interloss_margin', 1.5)
        self.sp_interloss = self.loss_config.get('sp_interloss_margin', 0.2)

        self.ft_intraloss = self.loss_config.get('ft_intraloss_margin', 1.0)
        self.sp_intraloss = self.loss_config.get('sp_intraloss_margin', 0.0)

        self.eps = self.loss_config.get('eps', 0.001)

        self.ft_loss_params = self.loss_config.get('ft_loss_params', dict(inter=1.0, intra=1.0, reg=0.1))
        self.sp_loss_params = self.loss_config.get('sp_loss_params', dict(inter=1.0, intra=1.0))

        self.kernel_lossfn_name = self.loss_config.get('kernel_lossfn', 'BCE')
        if self.kernel_lossfn_name == 'BCE':
            self.kernel_lossfn = nn.BCEWithLogitsLoss(reduction='mean')
        elif self.kernel_lossfn_name == 'lovasz_hinge':
            self.kernel_lossfn = LovaszHingeLoss(reduction='none')
        else:
            self.kernel_lossfn = nn.BCEWithLogitsLoss(reduction='none')

        self.seg_lossfn_name = self.loss_config.get('seg_lossfn', 'CE')
        if self.seg_lossfn_name == 'CE':
            self.seg_loss_fn = nn.CrossEntropyLoss(reduction='mean')
        elif self.seg_lossfn_name == 'lovasz_softmax':
            self.seg_loss_fn = LovaszSoftmaxWithLogitsLoss(reduction='mean')
        else:
            self.seg_loss_fn = nn.CrossEntropyLoss(reduction='mean')


    def feature_embedding_loss(self, ft_emb, groups, ft_centroids):
        '''
        Compute discriminative feature embedding loss.

        INPUTS:
            - ft_emb (N x F)
            - groups (N)
            - ft_centroids (N_c X F)
        '''
        intercluster_loss = inter_cluster_loss(ft_centroids, margin=self.ft_interloss)
        intracluster_loss = intra_cluster_loss(ft_emb, ft_centroids, groups, margin=self.ft_intraloss)
        reg_loss = torch.mean(torch.norm(ft_centroids, dim=1))
        out = {}
        out['intercluster_loss'] = float(intercluster_loss)
        out['intracluster_loss'] = float(intracluster_loss)
        out['regularization_loss'] = float(reg_loss)
        out['loss'] = self.ft_loss_params['inter'] * intercluster_loss + \
                      self.ft_loss_params['intra'] * intracluster_loss + \
                      self.ft_loss_params['reg'] * reg_loss
        return out

    def spatial_embedding_loss(self, sp_emb, groups, sp_centroids):
        '''
        Compute spatial centroid regression loss.

        INPUTS:
            - sp_emb (N x D)
            - groups (N)
            - ft_centroids (N_c X F)
        '''
        out = {}
        intercluster_loss = inter_cluster_loss(sp_centroids, margin=self.sp_interloss)
        intracluster_loss = intra_cluster_loss(sp_emb, sp_centroids, groups, margin=self.sp_intraloss)
        out['intercluster_loss'] = float(intercluster_loss)
        out['intracluster_loss'] = float(intracluster_loss)
        out['loss'] = self.sp_loss_params['inter'] * intercluster_loss + \
                      self.sp_loss_params['intra'] * intracluster_loss

        return out

    def covariance_loss(self, sp_emb, ft_emb, cov, groups,
                        sp_centroids, ft_centroids, eps=0.001):

        device = sp_emb.device
        cov_means = find_cluster_means(cov, groups)

        # Compute spatial term
        sp_emb_tmp = sp_emb[:, None, :]
        sp_centroids_tmp = sp_centroids[None, :, :]
        sp_cov = torch.clamp(cov_means[:, 0][None, :], min=eps)
        sp_sqdists = ((sp_emb_tmp - sp_centroids_tmp)**2).sum(-1) / (sp_cov**2)

        # Compute feature term
        ft_emb_tmp = ft_emb[:, None, :]
        ft_centroids_tmp = ft_centroids[None, :, :]
        ft_cov = torch.clamp(cov_means[:, 0][None, :], min=eps)
        ft_sqdists = ((ft_emb_tmp - ft_centroids_tmp)**2).sum(-1) / (ft_cov**2)

        # Compute joint kernel score
        pvec = torch.exp(-sp_sqdists - ft_sqdists)
        # probs = (1-pvec).index_put((torch.arange(groups.shape[0]), groups),
        #     torch.gather(pvec, 1, groups.view(-1, 1)).squeeze())
        logits = logit_fn(pvec)
        eye = torch.eye(len(groups.unique()), dtype=torch.float32, device=device)
        targets = eye[groups]

        acc = iou_batch(logits > 0, targets.bool())
        # Compute kernel score loss
        cov_loss = self.kernel_lossfn(logits, targets)
        return cov_loss, acc


    def occupancy_loss(self, occ, groups):
        '''
        INPUTS:
            - occ (N x 1)
            - groups (N)
        '''
        bincounts = torch.bincount(groups).float()
        bincounts[bincounts == 0] = 1
        occ_truth = torch.log(bincounts)
        occ_loss = torch.abs(torch.gather(occ - occ_truth[None, :], 1, groups.view(-1, 1)))
        occ_loss = scatter_mean(occ_loss.squeeze(), groups)
        # occ_loss = occ_loss[occ_loss > 0]

        return occ_loss.mean()


    def combine_multiclass(self, sp_embeddings, ft_embeddings, covariance,
            occupancy, slabels, clabels):
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
        counts = 0
        for sc in semantic_classes:
            if int(sc) == 4:
                continue
            index = (slabels == sc)
            sp_emb = sp_embeddings[index]
            ft_emb = ft_embeddings[index]
            cov = covariance[index]
            occ = occupancy[index]
            groups = clabels[index]
            groups_unique, _ = unique_label_torch(groups)
            sp_centroids = find_cluster_means(sp_emb, groups_unique)
            ft_centroids = find_cluster_means(ft_emb, groups_unique)
            # Get different loss components
            ft_out = self.feature_embedding_loss(
                ft_emb, groups_unique, ft_centroids)
            sp_out = self.spatial_embedding_loss(
                sp_emb, groups_unique, sp_centroids)
            cov_loss, acc = self.covariance_loss(
                sp_emb, ft_emb, cov, groups_unique,
                sp_centroids, ft_centroids, eps=self.eps)
            occ_loss = self.occupancy_loss(occ, groups_unique)
            # TODO: Combine loss with weighting, keep track for logging
            loss['ft_intra'].append(ft_out['intracluster_loss'])
            loss['ft_inter'].append(ft_out['intercluster_loss'])
            loss['ft_reg'].append(ft_out['regularization_loss'])
            loss['sp_intra'].append(sp_out['intracluster_loss'])
            loss['sp_inter'].append(sp_out['intercluster_loss'])
            loss['cov_loss'].append(float(cov_loss))
            loss['occ_loss'].append(float(occ_loss))
            loss['loss'].append(
                ft_out['loss'] + sp_out['loss'] + cov_loss + occ_loss)
            # TODO: Implement train-time accuracy estimation
            accuracy['acc_{}'.format(int(sc))] = acc
            accuracy['accuracy'] += acc
            counts += 1

            # for key, val in ft_out.items():
            #     if val != val:
            #         print('ft_loss: {} = NaN ({})'.format(key, val))
            #
            # for key, val in sp_out.items():
            #     if val != val:
            #         print('sp_loss: {} = NaN ({})'.format(key, val))
            #
            # if cov_loss != cov_loss:
            #     print('cov_loss: {}'.format(cov_loss))
            #
            # if occ_loss != occ_loss:
            #     print('occ_loss: {}'.format(occ_loss))

        accuracy['accuracy'] /= counts
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
            slabels = slabels.long()
            clabels = group_label[i][:, -1].long()
            batch_idx = segment_label[i][:, self.batch_column]
            sp_embedding = out['spatial_embeddings'][i]
            ft_embedding = out['feature_embeddings'][i]
            covariance = out['covariance'][i]
            occupancy = out['occupancy'][i]
            segmentation = out['segmentation'][i]
            nbatch = batch_idx.unique().shape[0]

            for bidx in batch_idx.unique(sorted=True):
                batch_mask = batch_idx == bidx
                sp_embedding_batch = sp_embedding[batch_mask]
                ft_embedding_batch = ft_embedding[batch_mask]
                segmentation_batch = segmentation[batch_mask]
                slabels_batch = slabels[batch_mask]
                clabels_batch = clabels[batch_mask]
                covariance_batch = covariance[batch_mask]
                occupancy_batch = occupancy[batch_mask]

                loss_seg = self.seg_loss_fn(segmentation_batch, slabels_batch)
                acc_seg = float(torch.sum(torch.argmax(
                    segmentation_batch, dim=1) == slabels_batch)) / float(segmentation_batch.shape[0])

                loss_class, acc_class = self.combine_multiclass(
                    sp_embedding_batch, ft_embedding_batch,
                    covariance_batch, occupancy_batch,
                    slabels_batch, clabels_batch)
                for key, val in loss_class.items():
                    loss[key].append(sum(val) / len(val))
                for s, acc in acc_class.items():
                    accuracy[s].append(acc)

                loss['loss_seg'].append(loss_seg)
                accuracy['acc_seg'].append(acc_seg)

        loss_avg = {}
        acc_avg = defaultdict(float)

        for key, val in loss.items():
            loss_avg[key] = sum(val) / len(val)
        loss_avg['loss'] += loss_avg['loss_seg']
        for key, val in accuracy.items():
            acc_avg[key] = sum(val) / len(val)

        res = {}
        res.update(loss_avg)
        res.update(acc_avg)

        return res
