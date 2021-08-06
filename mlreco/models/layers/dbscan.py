import torch
import numpy as np
import sklearn
from mlreco.utils.track_clustering import track_clustering

class DBSCANFragmenter(torch.nn.Module):
    """
    DBSCAN Layer that uses sklearn's DBSCAN implementation
    to fragment each of the particle classes into dense instances.
    - Runs pure DBSCAN for showers, Michel and Delta
    - Runs DBSCAN on PPN-masked voxels for tracks, associates leftovers based on proximity.
      Alternatively, uses a graph-based method to cluster tracks based on PPN vertices

    Args:
        data ([np.array]): (N,5) [x, y, z, batchid, sem_type]
        output (dict)    : Dictionary that contains the UResNet+PPN output
    Returns:
        (torch.tensor): [(C_0^0, C_0^1, ..., C_0^N_0), ...] List of list of clusters (one per class)
    """

    def __init__(self, cfg, name='dbscan_frag', batch_col=0, coords_col=(1, 4)):
        super(DBSCANFragmenter, self).__init__()

        model_cfg = cfg[name]

        self.batch_col = batch_col
        self.coords_col = coords_col

        # Global DBSCAN clustering parameters
        self.dim             = model_cfg.get('dim', 3)
        self.eps             = model_cfg.get('eps', 1.999)
        self.min_samples     = model_cfg.get('min_samples', 1)
        self.min_size        = model_cfg.get('min_size', 3)
        self.num_classes     = model_cfg.get('num_classes', 4)
        self.cluster_classes = model_cfg.get('cluster_classes', list(np.arange(self.num_classes)))

        # Track breaking parameters
        self.break_tracks             = model_cfg.get('break_tracks', True)
        self.track_label              = model_cfg.get('track_label', 1)
        self.michel_label             = model_cfg.get('michel_label', 2)
        self.delta_label              = model_cfg.get('delta_label', 3)
        self.track_clustering_method  = model_cfg.get('track_clustering_method', 'masked_dbscan')
        self.ppn_score_threshold      = model_cfg.get('ppn_score_threshold', 0.5)
        self.ppn_type_threshold       = model_cfg.get('ppn_type_threshold', 1.999)
        self.ppn_type_score_threshold = model_cfg.get('ppn_type_score_threshold', 0.5)
        self.ppn_mask_radius          = model_cfg.get('ppn_mask_radius', 5)

        # Assert consistency between parameter sizes
        if not isinstance(self.cluster_classes, list):
            self.cluster_classes = [self.cluster_classes]
        if not isinstance(self.eps, list):
            self.eps = [self.eps for _ in self.cluster_classes]
        if not isinstance(self.min_samples, list):
            self.min_samples = [self.min_samples for _ in self.cluster_classes]
        if not isinstance(self.min_size, list):
            self.min_size = [self.min_size for _ in self.cluster_classes]

        assert len(self.eps) == len(self.min_samples) == len(self.min_size) == len(self.cluster_classes)


    def get_clusts(self, data, bids, segmentation, track_points=None):
        # Loop over batch and semantic classes
        clusts = []
        for bid in bids:
            batch_mask = data[:, self.batch_col] == bid
            for k, s in enumerate(self.cluster_classes):
                # Run DBSCAN
                mask = batch_mask & (segmentation == s)
                if self.break_tracks and s == self.track_label:
                    mask = batch_mask & ((segmentation == s) | (segmentation == self.delta_label))
                selection = np.where(mask)[0]
                if not len(selection):
                    continue

                voxels = data[selection, self.coords_col[0]:self.coords_col[1]]
                if self.break_tracks and s == self.track_label:
                    assert track_points is not None
                    labels = track_clustering(voxels      = voxels,
                                              points      = track_points[track_points[:, self.batch_col] == bid,
                                                                         self.coords_col[0]:self.coords_col[1]],
                                              method      = self.track_clustering_method,
                                              eps         = self.eps[k],
                                              min_samples = self.min_samples[k],
                                              mask_radius = self.ppn_mask_radius)
                else:
                    labels = sklearn.cluster.DBSCAN(eps=self.eps[k],
                                                    min_samples=self.min_samples[k]).fit(voxels).labels_

                # Build clusters for this class
                if self.break_tracks and s == self.track_label:
                    labels[segmentation[selection] == self.delta_label] = -1
                cls_idx = [selection[np.where(labels == i)[0]] \
                    for i in np.unique(labels) \
                    if (i > -1 and np.sum(labels == i) >= self.min_size[k])]
                clusts.extend(cls_idx)

        same_length = np.all([len(c) == len(clusts[0]) for c in clusts])
        clusts = np.array(clusts, dtype=object if not same_length else np.int64)

        return clusts


    def forward(self, data, output=None, points=None):

        from mlreco.utils.ppn import uresnet_ppn_type_point_selector

        # If tracks are clustered, get the track points from the PPN output
        data = data.detach().cpu().numpy()
        track_points = None
        if self.break_tracks and self.track_label in self.cluster_classes:
            assert output is not None or points is not None
            if points is None:

                numpy_output = {'segmentation': [output['segmentation'][0].detach().cpu().numpy()],
                                'points'      : [output['points'][0].detach().cpu().numpy()],
                                'mask_ppn2'   : [output['mask_ppn2'][0].detach().cpu().numpy()]}

                points =  uresnet_ppn_type_point_selector(data, numpy_output,
                                                          score_threshold      = self.ppn_score_threshold,
                                                          type_threshold       = self.ppn_type_threshold,
                                                          type_score_threshold = self.ppn_type_score_threshold)
            point_labels = points[:,-1]
            track_points = points[(point_labels == self.track_label) | \
                                  (point_labels == self.michel_label),:self.dim+1]

        # Break down the input data to its components
        bids = np.unique(data[:, self.batch_col])
        segmentation = data[:,-1]
        data = data[:,:-1]

        clusts = self.get_clusts(data, bids, segmentation, track_points)
        return clusts


# def distances(v1, v2):
#     #print(v1.shape, v2.shape)
#     v1_2 = v1.unsqueeze(1).expand(v1.size(0), v2.size(0), v1.size(1)).double()
#     v2_2 = v2.unsqueeze(0).expand(v1.size(0), v2.size(0), v1.size(1)).double()
#     return torch.sqrt(torch.pow(v2_2 - v1_2, 2).sum(2))

def distances(v1, v2, eps=1e-6):
    v1_2 = v1.unsqueeze(1).expand(v1.size(0), v2.size(0), v1.size(1)).double()
    v2_2 = v2.unsqueeze(0).expand(v1.size(0), v2.size(0), v1.size(1)).double()
    return torch.sqrt(torch.clamp(torch.pow(v2_2 - v1_2, 2).sum(2), min=eps))


class MinkDBSCANFragmenter(DBSCANFragmenter):
    """
    DBSCANFragmenter for ME Backend PPN
    """

    def __init__(self, cfg, name='dbscan_frag'):
        super(MinkDBSCANFragmenter, self).__init__(cfg, batch_col=0, coords_col=(1,4))
        self.batch_col = 0
        self.coords_col = (1, 4)

    def forward(self, data, output):

        from mlreco.utils.ppn import mink_ppn_selector

        # If tracks are clustered, get the track points from the PPN output
        data = data.detach().cpu().numpy()
        track_points = None
        if self.track_label in self.cluster_classes:
            # FIXME ppn_score not in output?
            numpy_output = {'segmentation': [output['segmentation'][0].detach().cpu().numpy()],
                            'points'      : [output['points'][0].detach().cpu().numpy()],
                            'mask_ppn'    : [output['mask_ppn'][0].detach().cpu().numpy()],
                            'ppn_score'   : [output['ppn_score'][0].detach().cpu().numpy()]}

            points =  mink_ppn_selector(data, numpy_output,
                                        score_threshold      = self.ppn_score_threshold,
                                        type_threshold       = self.ppn_type_threshold,
                                        type_score_threshold = self.ppn_type_score_threshold)
            point_labels = points[:,-1]
            track_points = points[(point_labels == self.track_label) | \
                                  (point_labels == self.michel_label),:self.dim+1]

        # Break down the input data to its components
        bids = np.unique(data[:, self.batch_col].astype(int))
        segmentation = data[:,-1]
        data = data[:,:-1]

        clusts = self.get_clusts(data, bids, segmentation, track_points)
        return clusts
