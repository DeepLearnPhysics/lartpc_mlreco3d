import torch
import torch.nn as nn

from mlreco.utils import local_cdist
from mlreco.models.layers.cluster_cnn.losses.lovasz import lovasz_hinge_flat
from mlreco.models.layers.cluster_cnn.losses.lovasz import StableBCELoss
from collections import defaultdict


def logit(input, eps=1e-6):
    x = torch.clamp(input, min=eps, max=1-eps)
    return torch.log(x / (1 - x))


class EmbeddingLoss(nn.Module):
    '''
    Loss function for Sparse Spatial Embeddings Model, with fixed
    centroids and symmetric gaussian kernels.
    '''
    def __init__(self, cfg, name='clustering_loss'):
        super(EmbeddingLoss, self).__init__()
        self.loss_config = cfg[name]
        self.embedding_weight = self.loss_config.get('embedding_weight', 1.0)
        self.smoothing_weight = self.loss_config.get('smoothing_weight', 1.0)
        self.spatial_size = self.loss_config.get('spatial_size', 512)

        self.embedding_loss_name = self.loss_config.get(
            'embedding_loss_name', 'BCE')

        # BCELoss for Embedding Loss
        if self.embedding_loss_name == 'BCE':
            self.embedding_loss_fn = StableBCELoss()
        elif self.embedding_loss_name == 'lovasz':
            self.embedding_loss_fn = lovasz_hinge_flat
        else:
            raise ValueError('Loss function name {} does not correspond \
                to available options'.format(self.embedding_loss_name))


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
        device = features.device
        bincount = torch.bincount(labels)
        zero_bins = bincount > 0
        bincount[bincount == 0] = 1.0
        numerator = torch.zeros(bincount.shape[0], features.shape[1]).to(device)
        numerator = numerator.index_add(0, labels, features)
        centroids = numerator / bincount.view(-1, 1)
        centroids = centroids[zero_bins]
        return centroids


    def inter_cluster_loss(self, cluster_means, margin=0.2):
        inter_loss = 0.0
        n_clusters = len(cluster_means)
        if n_clusters < 2:
            # Inter-cluster loss is zero if there only one instance exists for
            # a semantic label.
            return 0.0
        else:
            indices = torch.triu_indices(cluster_means.shape[0],
                                         cluster_means.shape[0], 1)
            dist = local_cdist(cluster_means, cluster_means)
            return torch.pow(torch.clamp(2.0 * margin - dist[indices[0, :], \
                indices[1, :]], min=0), 2).mean()


    def get_per_class_probabilities(self, embeddings, margins, labels):
        '''
        Computes binary foreground/background loss.
        '''
        device = embeddings.device
        loss = 0.0
        smoothing_loss = 0.0
        centroids = self.find_cluster_means(embeddings, labels.to(dtype=torch.int64))
        n_clusters = len(centroids)
        cluster_labels = labels.unique(sorted=True)
        probs = torch.zeros(embeddings.shape[0]).float().to(device)
        acc = 0.0


        for i, c in enumerate(cluster_labels):
            index = (labels == c)
            mask = torch.zeros(embeddings.shape[0]).to(device)
            mask[index] = 1.0
            mask[~index] = 0.0
            sigma = torch.mean(margins[index], dim=0)
            dists = torch.sum(torch.pow(embeddings - centroids[i], 2), dim=1)
            p = torch.clamp(
                torch.exp(-dists / (2 * torch.pow(sigma, 2) + 1e-8)), min=0, max=1)
            probs[index] = p[index]
            logits = logit(p, eps=1e-6)
            loss += self.embedding_loss_fn(logits, mask)
            acc += float((mask.bool() & (p > 0.5)).sum()) \
                 / float((mask.bool() | (p > 0.5)).sum())
            sigma_detach = sigma.detach()
            smoothing_loss += torch.sum(torch.pow(margins[index] - sigma_detach, 2))

        loss /= n_clusters
        smoothing_loss /= n_clusters
        acc /= n_clusters

        return loss, smoothing_loss, probs, acc


    def combine_multiclass(self, embeddings, margins, slabels, clabels):
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
            mask_loss, smoothing_loss, probs, acc = self.get_per_class_probabilities(
                embeddings[index], margins[index], clabels[index])
            total_loss = self.embedding_weight * mask_loss \
                       + self.smoothing_weight * smoothing_loss
            loss['loss'].append(total_loss)
            loss['mask_loss'].append(float(self.embedding_weight * mask_loss))
            loss['smoothing_loss'].append(float(self.smoothing_weight * smoothing_loss))
            loss['mask_loss_{}'.format(int(sc))].append(float(mask_loss))
            accuracy['accuracy_{}'.format(int(sc))] = acc

        return loss, accuracy


    def forward(self, out, segment_label, group_label):

        num_gpus = len(segment_label)
        loss = defaultdict(list)
        accuracy = defaultdict(list)

        for i in range(num_gpus):
            slabels = segment_label[i][:, -1]
            coords = segment_label[i][:, 1:4].float()
            if torch.cuda.is_available():
                coords = coords.cuda()
            slabels = slabels.int()
            clabels = group_label[i][:, -1]
            batch_idx = segment_label[i][:, 0]
            embedding = out['embeddings'][i]
            margins = out['margins'][i]
            nbatch = batch_idx.unique().shape[0]

            for bidx in batch_idx.unique(sorted=True):
                embedding_batch = embedding[batch_idx == bidx]
                slabels_batch = slabels[batch_idx == bidx]
                clabels_batch = clabels[batch_idx == bidx]
                margins_batch = margins[batch_idx == bidx]

                loss_class, acc_class = self.combine_multiclass(
                    embedding_batch, margins_batch,
                    slabels_batch, clabels_batch)
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
