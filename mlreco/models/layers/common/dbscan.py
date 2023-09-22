import torch
import numpy as np
import sklearn
from larcv import larcv
from mlreco.utils.track_clustering import track_clustering


class DBSCANFragmenter(torch.nn.Module):
    """
    DBSCAN Layer that uses sklearn's DBSCAN implementation
    to fragment each of the particle classes into dense instances.
    Runs DBSCAN on each requested class separately, in one of three ways:
    - Run pure DBSCAN on all the voxels in that class
    - Runs DBSCAN on PPN point-masked voxels, associates leftovers based on proximity
    - Use a graph-based method to cluster tracks based on PPN vertices (track only)

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
        self.metric          = model_cfg.get('metric', 'euclidean')
        self.min_samples     = model_cfg.get('min_samples', 1)
        self.min_size        = model_cfg.get('min_size', 3)
        self.num_classes     = model_cfg.get('num_classes', 4)
        self.cluster_classes = model_cfg.get('cluster_classes', list(np.arange(self.num_classes)))

        # Instance breaking parameters
        self.break_classes            = model_cfg.get('break_classes', [1])
        self.track_include_delta      = model_cfg.get('track_include_delta', False)
        self.track_clustering_method  = model_cfg.get('track_clustering_method', 'masked_dbscan')
        self.ppn_score_threshold      = model_cfg.get('ppn_score_threshold', 0.5)
        self.ppn_type_threshold       = model_cfg.get('ppn_type_threshold', 1.999)
        self.ppn_type_score_threshold = model_cfg.get('ppn_type_score_threshold', 0.5)
        self.ppn_mask_radius          = model_cfg.get('ppn_mask_radius', 5)

        # Assert consistency between parameter sizes
        if 'break_tracks' in model_cfg: # Deprecated, only kept for backward compatibility
            assert 'break_classes' not in model_cfg, 'break_tracks is deprecated, only specify break_classes'
            self.break_classes = model_cfg['break_tracks']*[1]
        if not isinstance(self.cluster_classes, list):
            self.cluster_classes = [self.cluster_classes]
        if not isinstance(self.eps, list):
            self.eps = [self.eps for _ in self.cluster_classes]
        if not isinstance(self.min_samples, list):
            self.min_samples = [self.min_samples for _ in self.cluster_classes]
        if not isinstance(self.min_size, list):
            self.min_size = [self.min_size for _ in self.cluster_classes]
        if not isinstance(self.break_classes, list):
            self.break_classes = [self.break_classes]

        assert len(self.eps) == len(self.min_samples) == len(self.min_size) == len(self.cluster_classes)


    def get_clusts(self, data, bids, segmentation, break_points=None):
        # Loop over batch and semantic classes
        clusts = []
        for bid in bids:
            # Batch mask
            batch_mask = data[:, self.batch_col] == bid
            for k, s in enumerate(self.cluster_classes):
                # Batch and segmentation mask
                mask = batch_mask & (segmentation == s)
                if self.track_include_delta and s == larcv.kShapeTrack and s in self.break_classes:
                    mask = batch_mask & ((segmentation == s) | (segmentation == larcv.kShapeDelta))
                selection = np.where(mask)[0]
                if not len(selection):
                    continue

                # Restrict voxel set, run clustering
                voxels = data[selection, self.coords_col[0]:self.coords_col[1]]
                if s in self.break_classes:
                    assert break_points is not None
                    points_mask = break_points[:, self.batch_col] == bid
                    breaking_method = self.track_clustering_method if s==larcv.kShapeTrack else 'masked_dbscan'
                    labels = track_clustering(voxels      = voxels,
                                              points      = break_points[points_mask, self.coords_col[0]:self.coords_col[1]],
                                              method      = breaking_method,
                                              eps         = self.eps[k],
                                              min_samples = self.min_samples[k],
                                              metric      = self.metric,
                                              mask_radius = self.ppn_mask_radius)
                else:
                    labels = sklearn.cluster.DBSCAN(eps=self.eps[k],
                                                    min_samples=self.min_samples[k],
                                                    metric=self.metric).fit(voxels).labels_

                # Build clusters for this class
                if self.track_include_delta and s == larcv.kShapeTrack and s in self.break_classes:
                    labels[segmentation[selection] == larcv.kShapeDelta] = -1
                cls_idx = [selection[np.where(labels == i)[0]] \
                    for i in np.unique(labels) \
                    if (i > -1 and np.sum(labels == i) >= self.min_size[k])]
                clusts.extend(cls_idx)

        clusts_nb    = np.empty(len(clusts), dtype=object)
        clusts_nb[:] = clusts

        return clusts_nb


    def forward(self, data, output=None, points=None):

        # If instances are to be broken up, either provide a set of points or get them from the PPN output
        break_points = None
        if isinstance(data, torch.Tensor): data = data.detach().cpu().numpy()
        if points is not None and isinstance(points, torch.Tensor): points = points.detach().cpu().numpy()
        if len(self.break_classes):
            assert output is not None or points is not None
            if points is None:
                from mlreco.utils.ppn import get_ppn_predictions
                numpy_output = {'segmentation': [output['segmentation'][0].detach().cpu().numpy()],
                                'ppn_points'  : [output['ppn_points'][0].detach().cpu().numpy()],
                                'ppn_masks'   : [x.detach().cpu().numpy() for x in output['ppn_masks'][0]],
                                'ppn_coords'  : [x.detach().cpu().numpy() for x in output['ppn_coords'][0]]}

                points =  get_ppn_predictions(data, numpy_output,
                                              score_threshold      = self.ppn_score_threshold,
                                              type_threshold       = self.ppn_type_threshold,
                                              type_score_threshold = self.ppn_type_score_threshold)
                point_labels = points[:, 12]
            else:
                point_labels = points[:, -1]
            break_points = points[point_labels != larcv.kShapeDelta, :self.dim+1] # Do not include delta points

        # Break down the input data to its components
        bids = np.unique(data[:, self.batch_col].astype(int))
        segmentation = data[:,-1]
        data = data[:,:-1]

        clusts = self.get_clusts(data, bids, segmentation, break_points)
        return clusts
