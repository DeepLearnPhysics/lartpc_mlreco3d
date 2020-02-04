import torch
import torch.nn as nn
import numpy as np
import sparseconvnet as scn

from collections import defaultdict
from sklearn.metrics import adjusted_rand_score


class UResNet(torch.nn.Module):
    '''
    UResNet Backbone architecture. Nothing has changed from uresnet.py
    '''
    def __init__(self, cfg, name='discriminative_loss'):
        import sparseconvnet as scn
        super(UResNet, self).__init__()
        model_config = cfg[name]
        dimension = model_config['data_dim']
        self.spatial_size = model_config['spatial_size']
        reps = model_config.get('reps', 2)  # Conv block repetition factor
        kernel_size = 2  # Use input_spatial_size method for other values?
        m = model_config['filters']  # Unet number of features
        nPlanes = [i * m for i in range(1, model_config['num_strides'] + 1)]
        nInputFeatures = 1
        self._coordConv = model_config.get('coordConv', False)
        if self._coordConv:
            nInputFeatures += 3
        self.sparseModel = scn.Sequential().add(
            scn.InputLayer(dimension, model_config['spatial_size'], mode=3)).add(
            scn.SubmanifoldConvolution(dimension, nInputFeatures, m, 3,False)).add(
                   # Kernel size 3, no bias
            scn.UNet(dimension, reps, nPlanes, residual_blocks=True,
                     downsample=[kernel_size, 2], leakiness=0.25)).add(
                   # downsample = [filter size, filter stride]
            scn.BatchNormReLU(m)).add(scn.OutputLayer(dimension))
        if self._coordConv:
            self.linear = torch.nn.Linear(m, model_config['num_classes'])
        else:
            self.linear = torch.nn.Linear(m, model_config['num_classes'])

    def forward(self, input):
        """
        point_cloud is a list of length minibatch size (assumes mbs = 1)
        point_cloud[0] has 3 spatial coordinates + 1 batch coordinate + 1 feature
        shape of point_cloud[0] = (N, 4)
        """
        #print(input)
        point_cloud, = input
        coords = point_cloud[:, :-2].float()
        normalized_coords = (coords - self.spatial_size / 2) / float(self.spatial_size / 2)
        features = point_cloud[:, -1][:, None].float()
        if self._coordConv:
            features = torch.cat([normalized_coords, features], dim=1)
            emb = self.sparseModel((coords, features))
        else:
            emb = self.sparseModel((coords, features))
        x = self.linear(emb)
        return {
            'cluster_feature': [x]
        }


class DiscriminativeLoss(torch.nn.Module):
    '''
    Implementation of the Discriminative Loss Function in Pytorch.
    https://arxiv.org/pdf/1708.02551.pdf
    Note that there are many other implementations in Github, yet here
    we tailor it for use in conjuction with Sparse UResNet.
    '''

    def __init__(self, cfg, reduction='sum', name='discriminative_loss'):
        super(DiscriminativeLoss, self).__init__()
        self._cfg = cfg[name]

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
        clabels = labels.unique(sorted=True)
        cluster_means = []
        for c in clabels:
            index = (labels == c)
            #print(index)
            #index = (labels == c).squeeze(1).nonzero()
            #index = index.squeeze(1)
            mu_c = features[index].mean(0)
            cluster_means.append(mu_c)
        cluster_means = torch.stack(cluster_means)
        return cluster_means

    def intra_cluster_loss(self, features, labels, cluster_means, margin=0.5):
        '''
        Implementation of variance loss in Discriminative Loss.
        Inputs:
            features (torch.Tensor): pixel embedding, same as in find_cluster_means.
            labels (torch.Tensor): ground truth instance labels
            cluster_means (torch.Tensor): output from find_cluster_means
            margin (float/int): constant used to specify delta_v in paper. Think of it
            as the size of each clusters in embedding space.
        Returns:
            var_loss: (float) variance loss (see paper).
        '''
        var_loss = 0.0
        n_clusters = len(cluster_means)
        cluster_labels = labels.unique(sorted=True)
        for i, c in enumerate(cluster_labels):
            #index = (labels == c).squeeze(1).nonzero()
            #index = index.squeeze()
            index = (labels == c)
            dists = torch.norm(features[index] - cluster_means[i] + 1e-8,
                               p=self._cfg['norm'],
                               dim=1)
            hinge = torch.clamp(dists - margin, min=0)
            l = torch.mean(torch.pow(hinge, 2))
            var_loss += l
        var_loss /= n_clusters
        return var_loss

    def inter_cluster_loss(self, cluster_means, margin=1.5):
        '''
        Implementation of distance loss in Discriminative Loss.
        Inputs:
            cluster_means (torch.Tensor): output from find_cluster_means
            margin (float/int): the magnitude of the margin delta_d in the paper.
            Think of it as the distance between each separate clusters in
            embedding space.
        Returns:
            dist_loss (float): computed cross-centroid distance loss (see paper).
            Factor of 2 is included for proper normalization.
        '''
        dist_loss = 0.0
        n_clusters = len(cluster_means)
        if n_clusters < 2:
            # Inter-cluster loss is zero if there only one instance exists for
            # a semantic label.
            return 0.0
        else:
            for i, c1 in enumerate(cluster_means):
                for j, c2 in enumerate(cluster_means):
                    if i != j:
                        dist = torch.norm(c1 - c2 + 1e-8, p=self._cfg['norm'])
                        hinge = torch.clamp(2.0 * margin - dist, min=0)
                        dist_loss += torch.pow(hinge, 2)
            dist_loss /= float((n_clusters - 1) * n_clusters)
            return dist_loss

    def regularization(self, cluster_means):
        '''
        Implementation of regularization loss in Discriminative Loss
        Inputs:
            cluster_means (torch.Tensor): output from find_cluster_means
        Returns:
            reg_loss (float): computed regularization loss (see paper).
        '''
        reg_loss = 0.0
        n_clusters, _ = cluster_means.shape
        for i in range(n_clusters):
            reg_loss += torch.norm(cluster_means[i, :] + 1e-8, p=self._cfg['norm'])
        reg_loss /= float(n_clusters)
        return reg_loss

    def acc_DUResNet(self, embedding, truth, bandwidth=0.5):
        '''
        Compute Adjusted Rand Index Score for given embedding coordinates,
        where predicted cluster labels are obtained from distance to closest
        centroid (computes heuristic accuracy).

        Inputs:
            embedding (torch.Tensor): (N, d) Tensor where 'd' is the embedding dimension.
            truth (torch.Tensor): (N, ) Tensor for the ground truth clustering labels.
        Returns:
            score (float): Computed ARI Score
            clustering (array): the predicted cluster labels.
        '''
        from sklearn.metrics import adjusted_rand_score
        nearest = []
        with torch.no_grad():
            cmeans = self.find_cluster_means(embedding, truth)
            for centroid in cmeans:
                dists = torch.sum((embedding - centroid)**2, dim=1)
                dists = dists.view(-1, 1)
                nearest.append(dists)
            nearest = torch.cat(nearest, dim=1)
            nearest = torch.argmin(nearest, dim=1)
            pred = nearest.cpu().numpy()
            grd = truth.cpu().numpy()
            score = adjusted_rand_score(pred, grd)
        return score

    def combine(self, features, labels, verbose=True):
        '''
        Wrapper function for combining different components of the loss function.
        Inputs:
            features (torch.Tensor): pixel embeddings
            labels (torch.Tensor): ground-truth instance labels
        Returns:
            loss: combined loss, in most cases over a given semantic class.
        '''
        delta_var = self._cfg['delta_var']
        delta_dist = self._cfg['delta_dist']
        c_means = self.find_cluster_means(features, labels)
        loss_dist = self.inter_cluster_loss(c_means, margin=delta_dist)
        loss_var = self.intra_cluster_loss(features,
                                           labels,
                                           c_means,
                                           margin=delta_var)
        loss_reg = self.regularization(c_means)

        loss = self._cfg['alpha'] * loss_var + self._cfg[
            'beta'] * loss_dist + self._cfg['gamma'] * loss_reg
        if verbose:
            return [loss, float(loss_var), float(loss_dist), float(loss_reg)]
        else:
            return [loss]


    def combine_multiclass(self, features, slabels, clabels):
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
        loss, acc_segs = defaultdict(list), defaultdict(float)
        semantic_classes = slabels.unique()
        for sc in semantic_classes:
            index = (slabels == sc)
            num_clusters = len(clabels[index].unique())
            # FIXME:
            # Need faster clustering (maybe switch to DBSCAN?) or faster
            # estimates of clustering accuracy.
            loss_blob = self.combine(features[index], clabels[index])
            loss['total_loss'].append(loss_blob[0])
            loss['var_loss'].append(loss_blob[1])
            loss['dist_loss'].append(loss_blob[2])
            loss['reg_loss'].append(loss_blob[3])
            acc = self.acc_DUResNet(features[index], clabels[index])
            #print(acc)
            acc_segs[sc.item()] = acc
        return loss, acc_segs

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
        slabels = semantic_labels[0][:, 4]
        slabels = slabels.type(torch.LongTensor)
        clabels = group_labels[0][:, 4]
        batch_idx = semantic_labels[0][:, 3]
        embedding = out['cluster_feature'][0]
        loss = defaultdict(list)
        accuracy = defaultdict(list)
        nbatch = int(batch_idx.unique().shape[0])
        # Loop over each minibatch instance event
        for bidx in batch_idx.unique(sorted=True):
            embedding_batch = embedding[batch_idx == bidx]
            slabels_batch = slabels[batch_idx == bidx]
            clabels_batch = clabels[batch_idx == bidx]

            # Computing the Discriminative Loss
            if self._cfg['multiclass']:
                loss_dict, acc_segs = self.combine_multiclass(
                    embedding_batch, slabels_batch, clabels_batch)
                #print(acc_segs)
                loss["total_loss"].append(
                    sum(loss_dict["total_loss"]) / float(len(loss_dict["total_loss"])))
                loss["var_loss"].append(
                    sum(loss_dict["var_loss"]) / float(len(loss_dict["var_loss"])))
                loss["dist_loss"].append(
                    sum(loss_dict["dist_loss"]) / float(len(loss_dict["dist_loss"])))
                loss["reg_loss"].append(
                    sum(loss_dict["reg_loss"]) / float(len(loss_dict["reg_loss"])))
                for s, acc in acc_segs.items():
                    accuracy[s].append(acc)
            else:
                # DEPRECATED
                loss["total_loss"].append(self.combine(embedding_batch, clabels_batch))
                acc, _ = self.acc_DUResNet(embedding_batch, clabels_batch)
                accuracy.append(acc)

        total_loss = sum(loss["total_loss"]) / nbatch
        var_loss = sum(loss["var_loss"]) / nbatch
        dist_loss = sum(loss["dist_loss"]) / nbatch
        reg_loss = sum(loss["reg_loss"]) / nbatch
        acc_segs = defaultdict(float)
        acc_avg = []
        for i in range(5):
            if accuracy[i]:
                acc_segs[i] = sum(accuracy[i]) / float(len(accuracy[i]))
                acc_avg.append(acc_segs[i])
            else:
                acc_segs[i] = 0.0
        acc_avg = sum(acc_avg) / float(len(acc_avg))


        return {
            "loss": total_loss,
            "var_loss": var_loss,
            "dist_loss": dist_loss,
            "reg_loss": reg_loss,
            "accuracy": acc_avg,
            "acc_0": acc_segs[0],
            "acc_1": acc_segs[1],
            "acc_2": acc_segs[2],
            "acc_3": acc_segs[3],
            "acc_4": acc_segs[4]
        }
