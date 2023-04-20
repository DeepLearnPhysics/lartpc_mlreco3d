import numpy as np
from scipy.spatial.distance import cdist
import scipy
import os
from mlreco.post_processing import post_processing
from mlreco.utils import local_cdist, CSVData
from mlreco.utils.dbscan import dbscan_points
from mlreco.utils.ppn import uresnet_ppn_type_point_selector


@post_processing(['ppn-metrics-gt', 'ppn-metrics-pred'],
                ['segment_label', 'points_label', 'particles_label'],
                ['segmentation', 'points', 'mask_ppn', 'ppn_layers', 'ppn_coords'])
def ppn_simple(cfg, processor_cfg, data_blob, result, logdir, iteration,
                data_idx=None, segment_label=None, points_label=None, particles_label=None,
                ppn_layers=None, ppn_coords=None, points=None, segmentation=None,
                mask_ppn=None, **kwargs):
    import torch
    num_classes = processor_cfg.get('num_classes', 5)
    # clust_data = clust_data[data_idx]

    # segment_label = segment_label[data_idx]
    # segmentation = segmentation[data_idx]
    points = np.array(points)

    rows_gt_names, rows_gt_values = [], []
    rows_pred_names, rows_pred_values = [], []

    slabels         = segment_label[data_idx][:, -1]
    # clabels         = clust_data

    # Initialize log if one per event
    segmentation_batch = segmentation[data_idx]
    pred_seg = np.argmax(segmentation_batch, axis=1).astype(int)

    ppn = uresnet_ppn_type_point_selector(data_blob['input_data'][data_idx], result, entry=data_idx)
    if ppn.shape[0] == 0:
        return [(rows_gt_names, rows_gt_values), (rows_pred_names, rows_pred_values)]

    ppn_voxels = ppn[:, 1:4]
    ppn_score = ppn[:, 5]
    ppn_type = ppn[:, -2]
    ppn_endpoint_type = np.argmax(ppn[:, 13:], axis=1)

    d = cdist(points_label[data_idx][:, 1:4], ppn_voxels)

    # print(d.shape, points_label[data_idx].shape, ppn_voxels.shape)

    d_pred_to_closest_true = d.min(axis=0)
    pred_to_closest_true_coords = points_label[data_idx][d.argmin(axis=0)]
    d_true_to_closest_pred = d.min(axis=1)
    true_seg_voxels = segment_label[data_idx]
    true_mip_voxels = true_seg_voxels[slabels == 1]

    # Loop over true ppn points
    for i, true_point in enumerate(points_label[data_idx]):

        true_point_coord = true_point[1:4]
        true_point_type = true_point[4]
        true_point_idx = true_point[5].astype(np.int64)
        # p = particles_label[data_idx][true_point_idx]

        rows_gt_names.append(('Class',
                        # 'pdg',
                        'min_distance',
                        'x',
                        'y',
                        'z'))
        rows_gt_values.append((int(true_point_type),
                            #  p.pdg_code(),
                             d_true_to_closest_pred[i],
                             true_point_coord[0],
                             true_point_coord[1],
                             true_point_coord[2]))

    for i, pred_point in enumerate(ppn_voxels):
        pred_point_type, pred_point_score = ppn_type[i], ppn_score[i]
        closest_x, closest_y, closest_z = pred_to_closest_true_coords[i][1:4]
        closest_true_endpoint_type = pred_to_closest_true_coords[i][-1]
        pred_endpoint_type = ppn_endpoint_type[i]
        segmentation_voxels = segment_label[data_idx][:, 1:4][pred_seg == pred_point_type]
        if segmentation_voxels.shape[0] > 0:
            d_same_type = local_cdist(
                torch.Tensor(pred_point).view(1, -1),
                torch.Tensor(segmentation_voxels)).numpy()
            d_same_type_closest = d_same_type.min(axis=1)[0]
        else:
            d_same_type_closest = -1
        if true_mip_voxels.shape[0] > 0:
            d_mip = local_cdist(
                torch.Tensor(pred_point).view(1, -1),
                torch.Tensor(true_mip_voxels[:, 1:4])).numpy()
            d_closest_mip = d_mip.min(axis=1)[0]

        else:
            d_closest_mip = -1
        rows_pred_names.append(('Class',
                          'Score',
                          'min_distance',
                          'd_pred_to_same_type',
                          'd_true_mip',
                          'x',
                          'y',
                          'z',
                          'closest_x', 'closest_y', 'closest_z') \
                          + tuple(['score_type_%d' % c for c in range(num_classes)]) \
                          + tuple(['classify_endpoints_true', 'classify_endpoints_pred']))
        rows_pred_values.append((int(pred_point_type),
                             float(pred_point_score),
                             float(d_pred_to_closest_true[i]),
                             float(d_same_type_closest),
                             float(d_closest_mip),
                             ppn_voxels[i][0],
                             ppn_voxels[i][1],
                             ppn_voxels[i][2],
                             closest_x, closest_y, closest_z) \
                             + tuple(ppn[i, 7:12]) \
                             + tuple([closest_true_endpoint_type, pred_endpoint_type]))

    return [(rows_gt_names, rows_gt_values), (rows_pred_names, rows_pred_values)]
