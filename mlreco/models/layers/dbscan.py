from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import numpy as np
import sklearn
from mlreco.utils.track_clustering import track_clustering

class DBSCANFragmenter(torch.nn.Module):
    """
    DBSCAN Layer that uses sklearn's DBSCAN implementation
    to fragment each of the particle classes into dense instances.
    - Runs pure DBSCAN for showers, Michel and Delta
    - Runs DBSCAN on PPN-masked voxels for tracks, associates leftovers based on proximity

    Args:
        data ([torch.tensor]): (N,5) [x, y, z, batchid, sem_type]
        output (dict)        : Dictionary that contains the UResNet+PPN output
    Returns:
        (torch.tensor): [(C_0^0, C_0^1, ..., C_0^N_0), ...] List of list of clusters (one per class)
    """
    def __init__(self, cfg, name='dbscan_frag', cluster_classes=None):
        super(DBSCANFragmenter, self).__init__()
        self.cfg = cfg['dbscan_frag']
        self.dim = self.cfg.get('dim', 3)
        self.eps = self.cfg.get('eps', [1.999, 1.999, 1.999, 1.999])
        self.min_samples = self.cfg.get('min_samples', 1)
        self.min_size = self.cfg.get('min_size', [10,3,3,3])
        self.num_classes = self.cfg.get('num_classes', 4)
        
        # cluster_classes determines which semantic classes will undergo DBSCAN clustering
        # Priority to set this parameter (from top to bottom):
        # - configuration of DBScanFragmenter (so you can exclude LE for example)
        # - otherwise will be set by `ghost_chain_2` model for example
        #   (complementary set to cluster_classes for CNN clustering, which might include LE class)
        # - last default option is to cluster all classes defined by num_classes
        cluster_classes = self.cfg.get('cluster_classes', None)
        self.cluster_classes = range(self.num_classes) if cluster_classes is None else cluster_classes

        self.track_label = self.cfg.get('track_label', 1)
        self.michel_label = self.cfg.get('michel_label', 2)
        self.delta_label = self.cfg.get('delta_label', 3)
        self.track_clustering_method = self.cfg.get('track_clustering_method', 'masked_dbscan')
        self.ppn_score_threshold = self.cfg.get('ppn_score_threshold', 0.5)
        self.ppn_type_threshold = self.cfg.get('ppn_type_threshold', 1.999)
        self.ppn_type_score_threshold = self.cfg.get('ppn_type_score_threshold', 0.5)
        self.ppn_mask_radius = self.cfg.get('ppn_mask_radius', 5)

    def forward(self, data, output):

        from mlreco.utils.ppn import uresnet_ppn_type_point_selector
        from scipy.spatial.distance import cdist

        # Output one list of fragments
        clusts = []

        # Get the track points from the PPN output
        data = data.detach().cpu().numpy()
        numpy_output = {'segmentation':[output['segmentation'][0].detach().cpu().numpy()],
                        'points':      [output['points'][0].detach().cpu().numpy()],
                        'mask_ppn2':   [output['mask_ppn2'][0].detach().cpu().numpy()]}
        points =  uresnet_ppn_type_point_selector(data, numpy_output,
                                                  score_threshold = self.ppn_score_threshold,
                                                  type_threshold = self.ppn_type_threshold,
                                                  type_score_threshold = self.ppn_type_score_threshold)
        point_labels = points[:,-1]
        track_points = points[(point_labels == self.track_label) | (point_labels == self.michel_label),:self.dim+1]

        # Break down the input data to its components
        bids = np.unique(data[:,self.dim])
        segmentation = data[:,-1]
        data = data[:,:-1]

        # Loop over batch and semantic classes
        for bid in bids:
            batch_mask = data[:,self.dim] == bid
            for s in self.cluster_classes:
                # Run DBSCAN
                mask = batch_mask & (segmentation == s)
                if s == self.track_label:
                    mask = batch_mask & ((segmentation == s) | (segmentation == self.delta_label))
                selection = np.where(mask)[0]
                if not len(selection):
                    continue

                voxels = data[selection, :self.dim]
                if s == self.track_label:
                    labels = track_clustering(voxels = voxels,
                                              points = track_points[track_points[:,self.dim] == bid,:3],
                                              method = self.track_clustering_method,
                                              eps = self.eps[s],
                                              min_samples = self.min_samples,
                                              mask_radius = self.ppn_mask_radius)
                else:
                    labels = sklearn.cluster.DBSCAN(eps=self.eps[s], min_samples=self.min_samples).fit(voxels).labels_

                # Build clusters for this class
                if s == self.track_label:
                    labels[segmentation[selection] == self.delta_label] = -1
                cls_idx = [selection[np.where(labels == i)[0]] for i in np.unique(labels) if (i > -1 and np.sum(labels == i) >= self.min_size[s])]
                clusts.extend(cls_idx)

        return np.array(clusts, dtype=object)


class DBScanClusts(torch.nn.Module):
    """
    DBSCAN Layer that uses sklearn's DBSCAN implementation
    expects input that is of form
        x, y, z, batch_id, features, classes
        classes should be one-hot encoded

    forward:
        INPUT:
        x - torch.floatTensor
            x.shape = (N, dim + batch_index + feature + num_classes)
        OUTPUT:
        inds - list of torch.longTensor indices for each cluster
    """
    def __init__(self, cfg):
        super(DBScanClusts, self).__init__()
        self._cfg = cfg['dbscan']
        self.epsilon = self._cfg.get('epsilon', 15)
        self.minPoints = self._cfg.get('minPoints', 5)
        self.num_classes = self._cfg.get('num_classes', 5)
        self.dim = self._cfg.get('data_dim', 3)

    def forward(self, x, onehot=True):
        # output clusters
        clusts = []
        # none of this is differentiable.  Detach for call to numpy
        x = x.detach()
        # move to CPU if on gpu
        if x.is_cuda:
            x = x.cpu()
        bids = torch.unique(x[:,self.dim])
        if onehot:
            data = x[:,:-self.num_classes]
        else:
            data = x[:,:-1]
        if onehot:
            segmentation = x[:, -self.num_classes:]
        else:
            segmentation = x[:,-1] # labels
        # loop over batch
        for bid in bids:
            # batch indices
            binds = data[:,self.dim] == bid
            # loop over classes
            for c in range(self.num_classes):
                if onehot:
                    cinds = segmentation[:,c] == 1
                else:
                    cinds = segmentation == c
                print('class',c,cinds.sum(),'points')
                # batch = bid and class = c
                bcinds = torch.all(torch.stack([binds, cinds]), dim=0)
                selection = np.where(bcinds == 1)[0]
                if len(selection) == 0:
                    continue
                print('selection length',len(selection))
                # perform DBSCAN
                sel_vox = data[bcinds, :self.dim]
                print('sel vox length',sel_vox.size())
                res=sklearn.cluster.DBSCAN(eps=self.epsilon,
                                           min_samples=self.minPoints,
                                           metric='euclidean'
                                          ).fit(sel_vox)
                cls_idx = [ selection[np.where(res.labels_ == i)[0]] for i in range(np.max(res.labels_)+1) ]
                clusts.extend(cls_idx)

        return np.array(clusts)

class DBScanClusts2(torch.nn.Module):
    """
    DBSCAN Layer that uses sklearn's DBSCAN implementation
    expects input that is of form
        x, y, z, batch_id, features, classes
        classes should be one-hot encoded

    forward:
        INPUT:
        x - torch.floatTensor
            x.shape = (N, dim + batch_index + feature + num_classes)
        OUTPUT:
        inds - list of torch.longTensor indices for each cluster
    """
    def __init__(self, cfg):
        super(DBScanClusts2, self).__init__()
        self._cfg = cfg['dbscan']
        self.epsilon = self._cfg.get('epsilon', 15)
        self.minPoints = self._cfg.get('minPoints', 5)
        self.num_classes = self._cfg.get('num_classes', 5)
        self.dim = self._cfg.get('data_dim', 3)

    def forward(self, x, onehot=True):
        # output clusters
        clusts = []
        # none of this is differentiable.  Detach for call to numpy
        x = x.detach()
        # move to CPU if on gpu
        if x.is_cuda:
            x = x.cpu()
        bids = torch.unique(x[:,self.dim])
        if onehot:
            data = x[:,:-self.num_classes]
        else:
            data = x[:,:-1]
        if onehot:
            segmentation = x[:, -self.num_classes:]
        else:
            segmentation = x[:,-1] # labels
        for i in range(self.num_classes):
            clusts.append([])
        # loop over batch
        for bid in bids:
            # batch indices
            binds = data[:,self.dim] == bid
            # loop over classes
            for c in range(self.num_classes):
                if onehot:
                    cinds = segmentation[:,c] == 1
                else:
                    cinds = segmentation == c
                #print('class',c,cinds.sum(),'points')
                # batch = bid and class = c
                bcinds = torch.all(torch.stack([binds, cinds]), dim=0)
                selection = np.where(bcinds == 1)[0]
                if len(selection) == 0:
                    #clusts[c].append([])
                    continue
                #print('selection length',len(selection))
                # perform DBSCAN
                sel_vox = data[bcinds, :self.dim]
                #print('sel vox length',sel_vox.size())
                res=sklearn.cluster.DBSCAN(eps=self.epsilon,
                                           min_samples=self.minPoints,
                                           metric='euclidean'
                                          ).fit(sel_vox)
                cls_idx = [ selection[np.where(res.labels_ == i)[0]] for i in range(np.max(res.labels_)+1) ]
                #for idx in cls_idx:
                #    print('cluster',len(idx),'points')
                if not len(cls_idx):
                    continue
                clusts[c].extend(cls_idx)

        return clusts


class DBScan2(torch.nn.Module):
    """
    DBSCAN Layer that uses sklearn's DBSCAN implementation
    expects input that is of form
        x, y, z, batch_id, features, classes
        classes should be one-hot encoded

    forward:
        INPUT:
        x - torch.floatTensor
            x.shape = (N, dim + batch_index + feature + num_classes)
        OUTPUT:
        same as DBScan module
    """
    def __init__(self, cfg):
        super(DBScan2, self).__init__()
        self.dbclusts = DBScanClusts(cfg)
        self._cfg = cfg['dbscan']
        self.num_classes = self._cfg.get('num_classes', 5)

    def forward(self, x):
        # get cluster index sets
        clusts = self.dbclusts(x)

        ret = []
        for cinds in clusts:
            datac = x[cinds,:-self.num_classes]
            labelc = torch.argmax(x[cinds, -self.num_classes:], dim=1)
            ret.append(torch.cat([datac, labelc.double().view(-1,1)], dim=1))

        return ret


class DBScanFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, epsilon, minPoints, num_classes, dim):
        """
        input.shape = (N, dim + batch_index + feature + num_classes)
        epsilon, minPoints: parameters of DBScan
        num_classes: semantic segmentation classes
        dim: 2D or 3D
        """
        keep = (input, )  # Variables we keep for backward pass
        ctx.num_classes = num_classes  # Save this also (integer, not variable)
        ctx.dim = dim

        data = input[:, :-num_classes]  # (N, dim + batch_index + feature)
        segmentation = input[:, -num_classes:]  # (N, num_classes)
        class_index = torch.argmax(segmentation, dim=1)  # (N,)
        batch_indices = torch.unique(data[:, dim])
        keep += (class_index, batch_indices, )

        output = []
        for b in batch_indices:
            batch_index = data[:, dim] == b
            # For each class, run DBScan and record clusters
            for class_id in range(num_classes):
                mask = class_index[batch_index] == class_id
                labels = dbscan(data[batch_index][mask][:, :dim], epsilon, minPoints)
                labels = labels.reshape((-1,))
                keep += (labels, )
                print(b, class_id)
                # Now loop over clusters identified by DBScan, append class_id
                clusters = []
                unique_labels, _ = torch.sort(torch.unique(labels))
                for cluster_id in unique_labels:
                    if cluster_id >= 0:  # Ignore noise
                        cluster = data[batch_index][mask][labels == cluster_id]
                        # cluster = torch.cat([cluster[:, :-2], cluster[:, -1][:, None]], dim=1)
                        # cluster = torch.nn.functional.pad(cluster, (0, 1, 0, 0), mode='constant', value=b)
                        cluster = torch.nn.functional.pad(cluster, (0, 1, 0, 0), mode='constant', value=class_id)
                        clusters.append(cluster)  # (N_cluster, dim + batch_id + feature + class_id)
                output.extend(clusters)

        ctx.save_for_backward(*keep)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grad_out):
        """
        len(*grad_out) = number of clusters (outputs from forward)
        """
        input = ctx.saved_variables[0]
        segmentation = ctx.saved_variables[1]
        batch_indices = ctx.saved_variables[2]

        # For each class retrieve dbscan labels from forward
        # We do similar loops as in forward to retrieve labels in same order
        # as clusters were returned
        labels = {}  # Indexed by batch id and class id
        cluster_ids = []  # Unique cluster ids sorted, for each class/batch id
        class_ids = []  # Contains class_id shape = (N,)
        batch_ids = []  # Contains batch_id shape = (N,)
        i = 0
        for b in batch_indices:
            for class_id in range(ctx.num_classes):
                l = ctx.saved_variables[i+3]
                labels[(b, class_id)] = l
                cluster_ids.append(torch.sort(torch.unique(l[l>=0]))[0])
                class_ids.extend([class_id] * len(cluster_ids[-1]))
                batch_ids.extend([b] * len(cluster_ids[-1]))
                i += 1

        cluster_ids = torch.cat(cluster_ids)
        # Gradient must have same shape as input, we start with zeros
        grad_input = input.clone().fill_(0.0)
        # We know that the order of grad_cluster in grad_out corresponds
        # to the order in labels, cluster_ids, class_ids and batch_ids
        for i, grad_cluster in enumerate(grad_out):
            class_id = class_ids[i]
            batch_id = batch_ids[i]
            cluster_id = cluster_ids[i]
            batch_mask = input[:, ctx.dim] == batch_id  # Isolate this batch
            mask_class = segmentation == class_id  # Isolate this class
            mask_cluster = labels[(batch_id, class_id)] == cluster_id  # Isolate this cluster
            # We find the rows of input which belong to class_id,
            # then among these the rows which belong to cluster_id
            # Also we don't compute gradient for semantic segmentation scores
            # nor for class_id information (last column of grad_cluster)
            grad_input[(batch_mask & mask_class).nonzero()[mask_cluster].reshape((-1,)), :-ctx.num_classes] = grad_cluster[:, :-1]

        # As many outputs as inputs to forward
        return grad_input, None, None, None, None


class DBScan(torch.nn.Module):
    def __init__(self, cfg):
        super(DBScan, self).__init__()
        self.function = DBScanFunction.apply
        self._cfg = cfg['dbscan']
        self.epsilon = self._cfg.get('epsilon', 15)
        self.minPoints = self._cfg.get('minPoints', 5)
        self.num_classes = self._cfg.get('num_classes', 5)
        self.dim = self._cfg.get('data_dim', 3)

    def forward(self, x):
        """
        wrapper layer that incorporates additional things
        (like DBScan per semantic, and wrapper backward() to apply the combined
        mask from individual semantic's DBScan
        x.shape = (N, dim + batch_index + feature + num_classes)
        """
        return self.function(x, self.epsilon, self.minPoints, self.num_classes, self.dim)


def dbscan(points, epsilon, minPoints):
    """
    points.shape = [N, dim]
    labels: noise = -1, default fill = -2, labels id start at 0
    """
    num_points = points.size()[0]
    labels = torch.ones((num_points,)) * -2
    cluster_id = 0
    for p in range(num_points):
        if not (labels[p]) == -2:
            continue
        neighbors = region_query(epsilon, points, points[p])
        if neighbors.size()[0] < minPoints:  # Noise
            labels[p] = -1
        else:
            grow_cluster(epsilon, minPoints, points, labels, p, neighbors, cluster_id)
            cluster_id += 1
    return labels.reshape((-1, 1))


def grow_cluster(epsilon, minPoints, all_points, labels, point_id, neighbors, cluster_id):
    labels[point_id] = cluster_id
    i = 0
    while i < len(neighbors):
        point = neighbors[i]
        if labels[point] == -1:
            labels[point] = cluster_id
        elif labels[point] == -2:
            labels[point] = cluster_id
            new_neighbors = region_query(epsilon, all_points, all_points[point])
            if len(new_neighbors) >= minPoints:
                neighbors = torch.cat([neighbors, new_neighbors])
        i += 1


def distances(v1, v2):
    v1_2 = v1.unsqueeze(1).expand(v1.size(0), v2.size(0), v1.size(1)).double()
    v2_2 = v2.unsqueeze(0).expand(v1.size(0), v2.size(0), v1.size(1)).double()
    return torch.sqrt(torch.pow(v2_2 - v1_2, 2).sum(2))


def region_query(epsilon, all_points, point):
    """
    Assumes all_points.shape = (N, dim) and point = (dim, )
    Returns shape (N_neighbors,) (indexes in all_points)
    """
    d = distances(all_points, point[None, ...])
    return (d.reshape((-1,)) < epsilon).nonzero().reshape((-1,))


def dbscan_test():
    module = torch.nn.Sequential(
        torch.nn.Linear(3, 3),
        DBScan(),
        torch.nn.Linear(1, 2)
    )
    MSELoss = torch.nn.MSELoss(reduction='none')
    module.train()
    optimizer = torch.optim.Adam(module.parameters())
    x = torch.Tensor([
        [0.5, 0.5, 0.5],
        [1.0, 0.0, 1.0]
    ])
    y = module(x)
    labels = torch.Tensor([
        [-1, -1],
        [-1, -1]
    ])
    loss = MSELoss(labels, y).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


if __name__ == '__main__':
    wrapper = DBScan(num_classes=2, minPoints=1, epsilon=0.5)
    layer0 = torch.nn.Linear(in_features=6, out_features=2)
    layer1 = torch.nn.Linear(in_features=6, out_features=2)
    MSELoss = torch.nn.MSELoss(reduction='none')

    optimizer = torch.optim.Adam(list(layer0.parameters()) + list(layer1.parameters()))

    data = torch.tensor([
        [0.5, 0.5, 0.5, 0, 0.0003, 0.2, 0.8],
        [0.5, 0.6, 0.5, 0, 0.007, 0.3, 0.7],
        [1.0, 0.0, 2.0, 0, 0.02, 0.25, 0.75],
        [0.7, 3.0, 0.2, 1, 0.015, 0.6, 0.4],
        [0.9, 0.8, 0.7, 1, 0.1, 0.9, 0.1]
    ], requires_grad=True)
    labels0 = torch.tensor([
        [1, 1],
        [2, 2]
    ], dtype=torch.float)
    labels1 = torch.tensor([
        [3, 3],
        [4, 4]
    ], dtype=torch.float)
    print('Data: ', data)

    clusters = wrapper(data)
    print('Clusters: ', clusters)
    cluster0 = layer0(clusters[0])
    cluster1 = layer1(clusters[1])

    loss = MSELoss(labels0, cluster0).sum() + MSELoss(labels1, cluster1).sum()
    print('Loss: ', loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # print(len(clusters), "clusters")
    # cluster0.backward(torch.ones_like(cluster0))
    print(data.grad)
