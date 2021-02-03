import numpy as np
from scipy.spatial.distance import cdist
import scipy
import os
from mlreco.utils import CSVData
from mlreco.utils.dbscan import dbscan_points
from mlreco.utils.ppn import uresnet_ppn_point_selector, uresnet_ppn_type_point_selector

def ppn_simple(cfg, data_blob, res, logdir, iteration):
    # UResNet prediction

    method_cfg = cfg['post_processing']['ppn_simple']

    index        = data_blob['index']
    points       = res['points']
    attention    = res['ppn_scores']
    num_classes = 4
    points_label = data_blob['particles_label']
    # print(points_label)
    cluster_label     = data_blob['cluster_label']

    batch_column = cluster_label[0][:, 0].int()
    nbatches = len(cluster_label[0][:, 0].int().unique())

    clusters = [cluster_label[0][batch_column == i] for i in range(nbatches)]
    input_data = [cluster_label[0][:, [0, 1, 2, 3, 4]][batch_column == i] for i in range(nbatches)]
    points_label = [points_label[0][points_label[0][:, 0] == i] for i in range(nbatches)]
    # print([t.shape for t in points[0]])
    batch_column = points[0][-1][:, 0].astype(int)
    points = [points[0][-1][batch_column == i] for i in range(nbatches)]
    attention = [attention[0][-1][batch_column == i] for i in range(nbatches)]

    store_per_iteration = True
    if method_cfg is not None and method_cfg.get('store_method',None) is not None:
        assert(method_cfg['store_method'] in ['per-iteration','per-event'])
        store_per_iteration = method_cfg['store_method'] == 'per-iteration'
    fout=None
    if store_per_iteration:
        fout_gt=CSVData(os.path.join(logdir, 'ppn-metrics-gt-iter-%07d.csv' % iteration))
        fout_pred=CSVData(os.path.join(logdir, 'ppn-metrics-pred-iter-%07d.csv' % iteration))

    for data_idx, tree_idx in enumerate(index):

        if not store_per_iteration:
            fout_gt=CSVData(os.path.join(logdir, 'ppn-metrics-gt-event-%07d.csv' % tree_idx))
            fout_pred=CSVData(os.path.join(logdir, 'ppn-metrics-pred-event-%07d.csv' % tree_idx))

        # Store PPN metrics
        # fout_all.record(('idx', 'acc_ppn1', 'acc_ppn2'),
        #                 (data_idx, res['acc_ppn1'][data_idx], res['acc_ppn2'][data_idx]))
        # fout_all.write()

        # UResNet output
        label = clusters[data_idx][:, -1]

        # Remove deltas from true points
        delta = 3
        #points_label_idx = points_label[data_idx][points_label[data_idx][:, -1] != delta]
        points_label_idx = points_label[data_idx]
        # print(points[data_idx])
        # print(attention[data_idx])
        # type and idx in this order = -2, -1
        # print(np.unique(points_label_idx[:, -2]), np.unique(points_label_idx[:, -1]))

        ppn_voxels = points[data_idx][:, 1:4]
        ppn_score  = attention[data_idx].squeeze()
        ppn_mask = (ppn_score > 0.6)

        ppn_voxels = ppn_voxels[ppn_mask]
        ppn_score  = ppn_score[ppn_mask]

        d = cdist(ppn_voxels, points_label_idx[:, 1:4])
        distance_to_closest_true_point = d.min(axis=1)

        for i in range(ppn_voxels.shape[0]):
            fout_pred.record(('idx', 'distance_to_closest_true_point'),
                (tree_idx, distance_to_closest_true_point[i]))
            fout_pred.write()

        # Distance to closest pred point (regardless of type)
        d = cdist(ppn_voxels, points_label_idx[:, :3])
        #print(d.shape)
        distance_to_closest_pred_point = d.min(axis=0)
        for i in range(points_label_idx.shape[0]):
            fout_pred.record(('idx', 'distance_to_closest_pred_point'),
                (tree_idx, distance_to_closest_true_point[i]))
            fout_pred.write()

        if not store_per_iteration:
            fout_gt.close()
            fout_pred.close()
    if store_per_iteration:
        fout_gt.close()
        fout_pred.close()
