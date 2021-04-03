from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import numpy as np


class UResNet(torch.nn.Module):
    """
    UResNet

    For semantic segmentation, using sparse convolutions from SCN, but not the
    ready-made UNet from SCN library. The option `ghost` allows to train at the
    same time for semantic segmentation between N classes (e.g. particle types)
    and ghost points masking.

    Can also be used in a chain, for example stacking PPN layers on top.

    Configuration
    -------------
    num_strides : int
        Depth of UResNet, also corresponds to how many times we down/upsample.
    filters : int
        Number of filters in the first convolution of UResNet.
        Will increase linearly with depth.
    num_classes : int
        Should be number of classes (+1 if we include ghost points directly)
    data_dim : int
        Dimension 2 or 3
    spatial_size : int
        Size of the cube containing the data, e.g. 192, 512 or 768px.
    reps : int, optional
        Convolution block repetition factor
    kernel_size : int, optional
        Kernel size for the SC (sparse convolutions for down/upsample).
    features: int, optional
        How many features are given to the network initially.

    Returns
    -------
    list
        In order:
        - segmentation scores (N, num_classes)
        - feature maps of encoding path
        - feature maps of decoding path
    """
    INPUT_SCHEMA = [
        ["parse_sparse3d_scn", (float,), (3, 1)]
    ]

    MODULES = ['uresnet_clustering']

    def __init__(self, cfg, name="uresnet_clustering"):
        super(UResNet, self).__init__()
        import sparseconvnet as scn
        self._model_config = cfg[name]

        # Whether to compute ghost mask separately or not
        self._ghost = self._model_config.get('ghost', False)
        self._dimension = self._model_config.get('data_dim', 3)
        reps = self._model_config.get('reps', 2)  # Conv block repetition factor
        kernel_size = self._model_config.get('kernel_size', 2)
        num_strides = self._model_config.get('num_strides', 5)
        m = self._model_config.get('filters', 16)  # Unet number of features
        nInputFeatures = self._model_config.get('features', 1)
        spatial_size = self._model_config.get('spatial_size', 768)
        num_classes = self._model_config.get('num_classes', 5)
        self._N = self._model_config.get('num_cluster_conv', 0)
        self._simpleN = self._model_config.get('simple_conv', True)
        self._add_coordinates = self._model_config.get('cluster_add_coords', False)
        self._density_estimate = self._model_config.get('density_estimate', False)

        nPlanes = [i*m for i in range(1, num_strides+1)]  # UNet number of features per level
        downsample = [kernel_size, 2]  # [filter size, filter stride]
        self.last = None
        leakiness = 0

        def block(m, a, b):  # ResNet style blocks
            m.add(scn.ConcatTable()
                  .add(scn.Identity() if a == b else scn.NetworkInNetwork(a, b, False))
                  .add(scn.Sequential()
                    .add(scn.BatchNormLeakyReLU(a, leakiness=leakiness))
                    .add(scn.SubmanifoldConvolution(self._dimension, a, b, 3, False))
                    .add(scn.BatchNormLeakyReLU(b, leakiness=leakiness))
                    .add(scn.SubmanifoldConvolution(self._dimension, b, b, 3, False)))
             ).add(scn.AddTable())

        self.input = scn.Sequential().add(
           scn.InputLayer(self._dimension, spatial_size, mode=3)).add(
           scn.SubmanifoldConvolution(self._dimension, nInputFeatures, m, 3, False)) # Kernel size 3, no bias
        self.concat = scn.JoinTable()
        # Encoding
        self.bn = scn.BatchNormLeakyReLU(nPlanes[0], leakiness=leakiness)
        self.encoding_block = scn.Sequential()
        self.encoding_conv = scn.Sequential()
        module = scn.Sequential()
        for i in range(num_strides):
            module = scn.Sequential()
            for _ in range(reps):
                block(module, nPlanes[i], nPlanes[i])
            self.encoding_block.add(module)
            module2 = scn.Sequential()
            if i < num_strides-1:
                module2.add(
                    scn.BatchNormLeakyReLU(nPlanes[i], leakiness=leakiness)).add(
                    scn.Convolution(self._dimension, nPlanes[i], nPlanes[i+1],
                        downsample[0], downsample[1], False))
            self.encoding_conv.add(module2)
        self.encoding = module

        # Decoding
        self.decoding_conv, self.decoding_blocks = scn.Sequential(), scn.Sequential()
        for i in range(num_strides-2, -1, -1):
            inFeatures = nPlanes[i+1] * (2 if (self._N > 0 and i < num_strides-2) else 1)
            module1 = scn.Sequential().add(
                scn.BatchNormLeakyReLU(inFeatures, leakiness=leakiness)).add(
                scn.Deconvolution(self._dimension, inFeatures, nPlanes[i],
                    downsample[0], downsample[1], False))
            self.decoding_conv.add(module1)
            module2 = scn.Sequential()
            for j in range(reps):
                block(module2, nPlanes[i] * (2 if j == 0 else 1), nPlanes[i])
            self.decoding_blocks.add(module2)

        # Clustering convolutions
        if self._N > 0:
            self.clustering_conv = scn.Sequential()
            for i in range(num_strides-2, -1, -1):
                conv = scn.Sequential()
                for j in range(self._N):
                    if self._simpleN:
                        conv.add(scn.SubmanifoldConvolution(self._dimension, nPlanes[i] + (4 if j == 0 and self._add_coordinates else 0), nPlanes[i], 3, False))
                        conv.add(scn.BatchNormLeakyReLU(nPlanes[i], leakiness=leakiness))
                    else:
                        block(conv, nPlanes[i] + (4 if j == 0 and self._add_coordinates else 0), nPlanes[i])
                self.clustering_conv.add(conv)

        outFeatures = m * (2 if self._N > 0 else 1)
        self.output = scn.Sequential().add(
           scn.BatchNormReLU(outFeatures)).add(
           scn.OutputLayer(self._dimension))

        self.linear = torch.nn.Linear(outFeatures, num_classes)
        if self._density_estimate:
            self._density_layer = []
            for i in range(num_strides-2, -1, -1):
                self._density_layer.append(torch.nn.Linear(nPlanes[i], 2))
            self._density_layer = torch.nn.Sequential(*self._density_layer)


    def forward(self, input):
        """
        point_cloud is a list of length minibatch size (assumes mbs = 1)
        point_cloud[0] has 3 spatial coordinates + 1 batch coordinate + 1 feature
        label has shape (point_cloud.shape[0] + 5*num_labels, 1)
        label contains segmentation labels for each point + coords of gt points
        """
        point_cloud, = input
        coords = point_cloud[:, 0:self._dimension+1].float()
        features = point_cloud[:, self._dimension+1:].float()
        x = self.input((coords, features))
        feature_maps = [x]
        feature_ppn = [x]
        for i, layer in enumerate(self.encoding_block):
            x = self.encoding_block[i](x)
            feature_maps.append(x)
            x = self.encoding_conv[i](x)
            feature_ppn.append(x)

        # U-ResNet decoding
        feature_ppn2 = [x]
        feature_clustering = [x]
        feature_density = []
        for i, layer in enumerate(self.decoding_conv):
            encoding_block = feature_maps[-i-2]
            x = layer(x)
            x = self.concat([encoding_block, x])
            x = self.decoding_blocks[i](x)
            feature_ppn2.append(x)
            if self._N > 0:
                # Optionally add 3D coordinates before N convolutions for clustering
                if self._add_coordinates:
                    x.features = torch.cat([x.get_spatial_locations(), x.features], dim=1)
                x = self.clustering_conv[i](x)
            feature_clustering.append(x)
            if self._density_estimate:
                feature_density.append(self._density_layer[i](x.features))
            if self._N > 0:
                x = self.concat([feature_clustering[-1], feature_ppn2[-1]])

        x = self.output(x)
        x_seg = self.linear(x)  # Output of UResNet

        res = {
            'segmentation'     : [x_seg],
            'ppn_feature_enc' : [feature_ppn],
            'ppn_feature_dec' : [feature_ppn2],
            'cluster_feature'  : [feature_clustering]
            }

        if self._density_estimate:
            res['density_feature'] = [feature_density]

        return res


class SegmentationLoss(torch.nn.modules.loss._Loss):
    """
    Loss definition for UResNet.
    Instance clustering flavor

    Configuration
    -------------
    In addition to the network parameters, those specific to the loss include:

    alpha: float
        Weight for intracluster loss
    beta: float
        Weight for intercluster loss
    gamma: float
        Weight for regularization loss
    delta: float
        Weight for real distance loss
    intracluster_margin: float or list
        Margin for intracluster loss. If list, one value per depth level.
    intercluster_margin: float or list
        Margin for intercluster loss. If list, one value per depth level.
    uresnet_weight: float
        Weight for uresnet loss
    density_weight: float
        Weight for the density estimate loss. If a single value is provided,
        will weight the overall density loss.  If a list is provided, it will
        be weights for [overall density loss, density loss A, density loss B].
    radius: list
        Radius at which we compute neighbors density. List of float
    target_density_intercluster: float or list
    target_density_intracluster: float or list
    """
    # INPUT_SCHEMA = [
    #     ["parse_sparse3d_scn_scales", (int,), [(3, 1)]*5],
    #     ["parse_cluster3d_scales", (int,), [(3, 1)]*5]
    # ]

    def __init__(self, cfg, reduction='sum'):
        super(SegmentationLoss, self).__init__(reduction=reduction)
        self._cfg = cfg['uresnet_clustering']
        self._num_classes = self._cfg.get('num_classes', 5)
        self._depth = self._cfg.get('stride', 5)
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')

        self._alpha = self._cfg.get('alpha', 1)
        self._beta = self._cfg.get('beta', 1)
        self._gamma = self._cfg.get('gamma', 0.001)
        self._delta = self._cfg.get('delta', 0.)
        self._uresnet_weight = self._cfg.get('uresnet_weight', 1.0)
        self._dimension = self._cfg.get('data_dim', 3)

        self._intra_cluster_margin = self._cfg.get('intracluster_margin', 0.5)
        self._inter_cluster_margin = self._cfg.get('intercluster_margin', 1.5)

        if isinstance(self._intra_cluster_margin, float):
            self._intra_cluster_margin = [self._intra_cluster_margin] * self._depth
        if isinstance(self._inter_cluster_margin, float):
            self._inter_cluster_margin = [self._inter_cluster_margin] * self._depth
        if isinstance(self._intra_cluster_margin, list) and len(self._intra_cluster_margin) != self._depth:
            raise Exception("Expected list of size %d, got list of size %d for intracluster margin parameter." % (self._depth, self._intra_cluster_margin))
        if isinstance(self._inter_cluster_margin, list) and len(self._inter_cluster_margin) != self._depth:
            raise Exception("Expected list of size %d, got list of size %d for intercluster margin parameter." % (self._depth, self._inter_cluster_margin))

        # Density estimation configuration parameters
        self._density_estimate = self._cfg.get('density_estimate', False)
        self._density_weight = self._cfg.get('density_weight', 0.001)
        self._density_weightA = 1.
        self._density_weightB = 1.
        if isinstance(self._density_weight, list):
            if len(self._density_weight) != 3:
                raise Exception("Expected list of size 3, got list of size %d for density weight parameter." % len(self._density_weight))
            self._density_weightA = self._density_weight[1]
            self._density_weightB = self._density_weight[2]
            self._density_weight = self._density_weight[0]

        self._target_densityA = self._cfg.get('target_density_intracluster', 0.9)
        self._target_densityB = self._cfg.get('target_density_intercluster', 0.1)

        self._radius = self._cfg.get('radius', [2.0])
        if not isinstance(self._radius, list):
            raise Exception("Expected list for radius parameter.")
        if isinstance(self._target_densityA, float) or isinstance(self._target_densityA, int):
            self._target_densityA = [self._target_densityA] * len(self._radius)
        if isinstance(self._target_densityB, float) or isinstance(self._target_densityB, int):
            self._target_densityB = [self._target_densityB] * len(self._radius)
        if isinstance(self._target_densityA, list) and len(self._target_densityA) != len(self._radius):
            raise Exception("Expected list of size %d, got list of size %d for target densityA parameter." % (len(self._radius), len(self._target_densityA)))
        if isinstance(self._target_densityB, list) and len(self._target_densityB) != len(self._radius):
            raise Exception("Expected list of size %d, got list of size %d for target densityB parameter." % (len(self._radius), len(self._target_densityB)))

    def distances(self, v1, v2):
        """
        Simple method to compute distances from points in v1 to points in v2.
        """
        v1_2 = v1.unsqueeze(1).expand(v1.size(0), v2.size(0), v1.size(1))
        v2_2 = v2.unsqueeze(0).expand(v1.size(0), v2.size(0), v1.size(1))
        return torch.sqrt(torch.pow(v2_2 - v1_2, 2).sum(2) + 0.000000001)

    def distances2(self, points):
        """
        Uses BLAS/LAPACK operations to efficiently compute pairwise distances.
        """
        M = points
        transpose  = M.permute([0, 2, 1])
        zeros = torch.zeros(1, 1, 1)
        if torch.cuda.is_available():
            zeros = zeros.cuda()
        inner_prod = torch.baddbmm(zeros, M, transpose, alpha=-2.0, beta=0.0)
        squared    = torch.sum(torch.mul(M, M), dim=-1, keepdim=True)
        squared_tranpose = squared.permute([0, 2, 1])
        inner_prod += squared
        inner_prod += squared_tranpose
        return inner_prod

    def forward(self, result, label, cluster_label):
        """
        result[0], label and weight are lists of size #gpus = batch_size.
        segmentation has as many elements as UResNet returns.
        label[0] has shape (N, 1) where N is #pts across minibatch_size events.
        """
        assert len(result['segmentation']) == len(label)
        batch_ids = [d[0][:, -2] for d in label]
        uresnet_loss, uresnet_acc = 0., 0.

        cluster_intracluster_loss = 0.
        cluster_intercluster_loss = 0.
        cluster_reg_loss = 0.
        cluster_real_distance_loss = 0.
        cluster_total_loss = 0.
        cluster_intracluster_loss_per_class = [0.] * self._num_classes
        cluster_intercluster_loss_per_class = [0.] * self._num_classes
        cluster_reg_loss_per_class = [0.] * self._num_classes
        cluster_real_distance_loss_per_class = [0.] * self._num_classes
        cluster_total_loss_per_class = [0.] * self._num_classes
        density_loss = 0.
        density_lossA_estimate, density_lossA_target = 0., 0.
        density_lossB_estimate, density_lossB_target = 0., 0.
        density_accA = [0.] * len(self._radius)
        density_accB = [0.] * len(self._radius)
        accuracy = []
        for j in range(self._depth):
            accuracy.append([0.] * self._num_classes)

        for i in range(len(label)):
            max_depth = len(cluster_label[i])
            # for j, feature_map in enumerate(segmentation[3][i]):
            #     hypercoordinates = feature_map.features
            #     self.distances2(hypercoordinates)
            for b in batch_ids[i].unique():
                batch_index = batch_ids[i] == b

                event_segmentation = result['segmentation'][i][batch_index]  # (N, num_classes)
                event_label = label[i][0][batch_index][:, -1][:, None]  # (N, 1)
                event_label = torch.squeeze(event_label, dim=-1).long()

                # Reorder event_segmentation to match event_label
                data_coords = result['cluster_feature'][i][-1].get_spatial_locations()[batch_index][:, :-1]
                perm = np.lexsort((data_coords[:, 2], data_coords[:, 1], data_coords[:, 0]))
                event_segmentation = event_segmentation[perm]

                # Loss for semantic segmentation
                loss_seg = self.cross_entropy(event_segmentation, event_label)
                uresnet_loss += torch.mean(loss_seg)

                # Accuracy for semantic segmentation
                predicted_labels = torch.argmax(event_segmentation, dim=-1)
                acc = predicted_labels.eq_(event_label).sum().item() / float(predicted_labels.nelement())
                uresnet_acc += acc

                # Loss for clustering
                # Loop first over feature maps, starting from coarsest one
                # Note: We have to be careful with coordinates sorting.
                # TODO
                # - density estimation for points within 0.1, 0.3, 0.5, outside of 0.5 of cluster center
                for j, feature_map in enumerate(result['cluster_feature'][i]):
                    if torch.cuda.is_available():
                        batch_index = feature_map.get_spatial_locations()[:, -1].cuda() == b.long()
                    else:
                        batch_index = feature_map.get_spatial_locations()[:, -1] == b.long()
                    hypercoordinates = feature_map.features[batch_index]
                    coordinates = feature_map.get_spatial_locations()[batch_index][:, :-1]
                    clusters = cluster_label[i][-(j+1+(max_depth-self._depth))][cluster_label[i][-(j+1+(max_depth-self._depth))][:, self._dimension] == b]
                    # clusters_coordinates = clusters[:, :self._dimension]
                    clusters_labels = clusters[:, -1:]
                    semantic_labels = label[i][-(j+1+(max_depth-self._depth))][label[i][-(j+1+(max_depth-self._depth))][:, self._dimension] == b]
                    # Sort coordinates in lexicographic order
                    x = coordinates.cpu().detach().numpy()
                    perm = np.lexsort((x[:, 2], x[:, 1], x[:, 0]))
                    coordinates = coordinates[perm]
                    hypercoordinates = hypercoordinates[perm]

                    # Density estimate loss
                    if self._density_estimate and j > 0:
                        density_estimate = result['density_feature'][i][j-1][batch_index][perm]
                        clusters_id = clusters_labels.unique()
                        lossA_estimate, lossA_target, lossB_estimate, lossB_target = 0., 0., 0., 0.
                        total_densityA = [0.] * len(self._radius)
                        total_densityB = [0.] * len(self._radius)
                        distances = self.distances2(hypercoordinates[None,...][..., :3]).squeeze(0)
                        for c in clusters_id:
                            cluster_idx = (clusters_labels == c).squeeze(1)
                            cluster = hypercoordinates[cluster_idx]
                            estimate = density_estimate[cluster_idx]
                            for k, r in enumerate(self._radius):
                                # d = (self.distances(cluster, hypercoordinates) < r)
                                d = (distances[cluster_idx, :] < r)
                                # d = self.radius(cluster, hypercoordinates, r)
                                densityA = d[:, cluster_idx].sum(dim=1)
                                densityB = d[:, ~cluster_idx].sum(dim=1)
                                # total = (densityA + densityB).float()
                                densityA = densityA.float() #/ total
                                densityB = densityB.float() #/ total
                                total_densityA[k] += densityA.mean()
                                total_densityB[k] += densityB.mean()
                                lossA_estimate += torch.pow(estimate[:, 0] - densityA, 2) .mean()
                                lossB_estimate += torch.pow(estimate[:, 1] - densityB, 2).mean()
                                lossA_target += torch.pow(torch.clamp(self._target_densityA[k] - densityA, min=0), 2).mean()
                                lossB_target += torch.pow(torch.clamp(densityB - self._target_densityB[k], min=0), 2).mean()
                                # print("densityA", j, c, densityA.mean())
                                #print(torch.clamp(self._target_densityA - densityA, min=0))
                                #print("densityB", densityB)
                                #print(torch.clamp(densityB - self._target_densityB, min=0))

                        lossA_estimate /= clusters_id.size(0) * len(self._radius)
                        lossA_target /= clusters_id.size(0) * len(self._radius)
                        lossB_estimate /= clusters_id.size(0) * len(self._radius)
                        lossB_target /= clusters_id.size(0) * len(self._radius)

                        density_loss += self._density_weightA * (lossA_estimate + lossA_target) + self._density_weightB * (lossB_estimate + lossB_target)
                        density_lossA_estimate += lossA_estimate
                        density_lossB_estimate += lossB_estimate
                        density_lossA_target += lossA_target
                        density_lossB_target += lossB_target
                        for k in range(len(self._radius)):
                            total_densityA[k] /= clusters_id.size(0) * len(self._radius)
                            density_accA[k] += total_densityA[k]
                            total_densityB[k] /= clusters_id.size(0) * len(self._radius)
                            density_accB[k] += total_densityB[k]
                        # print(density_lossA_estimate, density_lossA_target, density_lossB_estimate, density_lossB_target)

                    # Loop over semantic classes
                    for class_ in range(self._num_classes):
                        class_index = semantic_labels[:, -1] == class_

                        # 0. Identify label clusters
                        clusters_id = clusters_labels[class_index].unique()
                        hyperclusters = []  # Hypercoordinates for each true cluster
                        realclusters = []  # Real coordinates of centroids of each true cluster
                        for c in clusters_id:
                            cluster_idx = (clusters_labels[class_index] == c).squeeze(1)
                            hyperclusters.append(hypercoordinates[class_index][cluster_idx])
                            realclusters.append(coordinates[class_index][cluster_idx].float())

                        # 1. Loop over clusters, define intra-cluster loss
                        #
                        # Also define real cluster loss = mean distance in real
                        # coordinates from a point to the centroid of the
                        # predicted cluster. This should avoid clustering
                        # together points that are far away in the real life.
                        intra_cluster_loss = 0.
                        real_distance_loss = 0.
                        means = []
                        realmeans = []
                        zero = torch.tensor(0.)
                        if torch.cuda.is_available(): zero = zero.cuda()
                        C = len(hyperclusters)
                        if C > 0:
                            for x, cluster in enumerate(hyperclusters):
                                mean = cluster.mean(dim=0)
                                means.append(mean)
                                realmean = realclusters[x].mean(dim=0)
                                realmeans.append(realmean)
                                # intra_cluster_loss += torch.max(((mean - cluster).pow(2).sum(dim=1) + 0.000000001).sqrt() - self._intra_cluster_margin, zero).pow(2).mean()
                                intra_cluster_loss += torch.mean(torch.pow(torch.clamp(torch.norm(cluster-mean, dim=1)- self._intra_cluster_margin[j], min=0), 2))
                            intra_cluster_loss /= C
                            means = torch.stack(means)
                            realmeans = torch.stack(realmeans)
                            # Now compute real cluster loss
                            for x, cluster in enumerate(hyperclusters):
                                # Assign each point to a predicted centroid
                                predicted_assignments = torch.argmin(self.distances(cluster, means), dim=1)
                                # Distance to this centroid in real space
                                real_distance_loss += torch.mean(torch.pow(torch.norm(realmeans[predicted_assignments] - realclusters[x], dim=1), 2))
                            real_distance_loss /= C
                            # compute accuracy based on this heuristic cluster
                            # prediction assignments
                            predicted_assignments = torch.argmin(self.distances(hypercoordinates[class_index], means), dim=1)
                            predicted_assignments = clusters_id[predicted_assignments]
                            accuracy[j][class_] += predicted_assignments.eq_(clusters_labels[class_index].squeeze(1)).sum().item() / float(predicted_assignments.nelement())
                        # 2. Define inter-cluster loss
                        inter_cluster_loss = 0.
                        if C > 1:
                            d = torch.max(2 * self._inter_cluster_margin[j] - self.distances(means, means), zero).pow(2)
                            inter_cluster_loss = d[np.triu_indices(d.size(1), k=1)].sum() * 2.0 / (C * (C-1))
                        # 3. Add regularization term
                        reg_loss = 0.
                        if C > 0:
                            # reg_loss = (means.pow(2).sum(dim=1) + 0.000000001).sqrt().mean()
                            reg_loss = torch.norm(means, dim=1).mean()

                        # Compute final loss
                        total_loss = self._alpha * intra_cluster_loss + self._beta * inter_cluster_loss + self._gamma * reg_loss + self._delta * real_distance_loss
                        cluster_intracluster_loss += self._alpha * intra_cluster_loss
                        cluster_intercluster_loss += self._beta * inter_cluster_loss
                        cluster_reg_loss += self._gamma * reg_loss
                        cluster_real_distance_loss += self._delta * real_distance_loss
                        cluster_total_loss += total_loss
                        cluster_intracluster_loss_per_class[class_] += self._alpha * intra_cluster_loss
                        cluster_intercluster_loss_per_class[class_] += self._beta * inter_cluster_loss
                        cluster_reg_loss_per_class[class_] += self._gamma * reg_loss
                        cluster_real_distance_loss_per_class[class_] += self._delta * real_distance_loss
                        cluster_total_loss_per_class[class_] += total_loss
                        # print(feature_map.features.shape, feature_map.spatial_size)
                        # print(j, class_, "Intra =", torch.tensor(self._alpha * intra_cluster_loss).item())
                        # print(j, class_, "Inter =", torch.tensor(self._beta * inter_cluster_loss).item())
                        # print(j, class_, "Reg = ", torch.tensor(self._gamma * reg_loss).item())
                        # print(j, class_, "Intra =", torch.tensor(self._alpha * intra_cluster_loss/float(self._num_classes)).item())
                        # print(j, class_, "Inter =", torch.tensor(self._beta * inter_cluster_loss/float(self._num_classes)).item())
                        # print(j, class_, "Reg = ", torch.tensor(self._gamma * reg_loss/float(self._num_classes)).item())

        batch_size = len(batch_ids[i].unique())
        # cluster_intracluster_loss /= self._num_classes
        # cluster_intercluster_loss /= self._num_classes
        # cluster_reg_loss /= self._num_classes
        # cluster_total_loss /= self._num_classes
        # cluster_intracluster_loss_per_class = [x/batch_size for x in cluster_intracluster_loss_per_class]
        # cluster_intercluster_loss_per_class = [x/batch_size for x in cluster_intercluster_loss_per_class]
        # cluster_reg_loss_per_class = [x/batch_size for x in cluster_reg_loss_per_class]
        # cluster_total_loss_per_class = [x/batch_size for x in cluster_total_loss_per_class]
        # print("Intra = ", cluster_intracluster_loss.item())
        # print("Inter = ", cluster_intercluster_loss.item())
        # print("Reg = ", cluster_reg_loss.item())

        results = {
            'accuracy': uresnet_acc / float(batch_size),
            'loss': (self._uresnet_weight * uresnet_loss + cluster_total_loss + self._density_weight * density_loss) / float(batch_size),
            'uresnet_loss': self._uresnet_weight * uresnet_loss / float(batch_size),
            'uresnet_acc': uresnet_acc / float(batch_size),
            'intracluster_loss': cluster_intracluster_loss / float(batch_size),
            'intercluster_loss': cluster_intercluster_loss / float(batch_size),
            'reg_loss': cluster_reg_loss / float(batch_size),
            'real_distance_loss': cluster_real_distance_loss / float(batch_size),
            'total_cluster_loss': cluster_total_loss / float(batch_size),
            'density_loss': self._density_weight * density_loss / float(batch_size),
            'density_lossA_estimate': self._density_weight * self._density_weightA * density_lossA_estimate / float(batch_size),
            'density_lossB_estimate': self._density_weight * self._density_weightB * density_lossB_estimate / float(batch_size),
            'density_lossA_target': self._density_weight * self._density_weightA * density_lossA_target / float(batch_size),
            'density_lossB_target': self._density_weight * self._density_weightB * density_lossB_target / float(batch_size),
        }
        for i, r in enumerate(self._radius):
            results['density_accA_%.2f' % r] = density_accA[i] / float(batch_size)
            results['density_accB_%.2f' % r] = density_accB[i] / float(batch_size)

        for class_ in range(self._num_classes):
            results['intracluster_loss_%d' % class_] = cluster_intracluster_loss_per_class[class_] / float(batch_size)
            results['intercluster_loss_%d' % class_] = cluster_intercluster_loss_per_class[class_] / float(batch_size)
            results['reg_loss_%d' % class_] = cluster_reg_loss_per_class[class_] / float(batch_size)
            results['real_distance_loss_%d' % class_] = cluster_real_distance_loss_per_class[class_] / float(batch_size)
            results['total_cluster_loss_%d' % class_] = cluster_total_loss_per_class[class_] / float(batch_size)
            for j in range(self._depth):
                results['acc_%d_%d' % (j, class_)] = accuracy[j][class_] / float(batch_size)

        return results
