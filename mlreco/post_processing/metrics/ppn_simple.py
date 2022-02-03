import numpy as np
from scipy.spatial.distance import cdist
import scipy
import os
from mlreco.post_processing import post_processing
from mlreco.utils import CSVData
from mlreco.utils.dbscan import dbscan_points
from mlreco.utils.ppn import uresnet_ppn_type_point_selector


def pairwise_distances(v1, v2):
    import torch
    v1_2 = v1.unsqueeze(1).expand(v1.size(0), v2.size(0), v1.size(1)).double()
    v2_2 = v2.unsqueeze(0).expand(v1.size(0), v2.size(0), v1.size(1)).double()
    return torch.sqrt(torch.pow(v2_2 - v1_2, 2).sum(2))

@post_processing(['ppn-metrics-gt', 'ppn-metrics-pred'],
                ['seg_label', 'points_label', 'clust_data', 'particles'],
                ['segmentation', 'points', 'mask_ppn', 'ppn_layers', 'ppn_coords'])
def ppn_simple(cfg, processor_cfg, data_blob, result, logdir, iteration,
                data_idx=None, seg_label=None, points_label=None, clust_data=None, particles=None,
                ppn_layers=None, ppn_coords=None, points=None, segmentation=None,
                mask_ppn=None, **kwargs):
    import torch
    num_classes = processor_cfg.get('num_classes', 5)
    clust_data = clust_data[data_idx]

    # ppn_layers   = result['ppn_layers'][0]
    # ppn_coords   = result['ppn_coords']
    # points       = result['points']#[0]
    seg_label = seg_label[data_idx]
    segmentation = segmentation[data_idx]
    points = np.array(points)
    # mask_ppn     = result['mask_ppn']
    batch_index  = ppn_coords[-1][:, 0]

    rows_gt_names, rows_gt_values = [], []
    rows_pred_names, rows_pred_values = [], []

    batch_mask      = batch_index == data_idx
    slabels         = seg_label[:, -1]#.int().numpy()
    clabels         = clust_data#[batch_mask]#.float().numpy()
    lowE_mask       = slabels != 4
    ppn_score_layer = ppn_layers[-1][batch_index == data_idx]
    coords_layer    = ppn_coords[-1][batch_index == data_idx]

    pixel_pred   = points[batch_index == data_idx][:,  0:3] \
                 + coords_layer[:, 1:4]
    pixel_score  = points[batch_index == data_idx][:, -1]
    pixel_logits = points[batch_index == data_idx][:,  3:8]
    points_label = points_label[data_idx]#[particles[:, 0] == data_idx]

    # Initialize log if one per event
    points_batch = points[batch_mask]
    segmentation_batch = segmentation#[batch_mask]
    mask_ppn_batch = mask_ppn[-1][batch_mask]

    res = {
        'points': [points_batch],
        'mask_ppn': [[mask_ppn_batch]],
        'segmentation': [segmentation_batch],
        #'ppn_score': scipy.special.expit(pixel_score)
    }
    # if 'ghost' in result:
    #     res['ghost'] = result['ghost']

    pred_seg = np.argmax(segmentation_batch, axis=1).astype(int)
    acc_seg  = np.sum(pred_seg == slabels) \
             / float(segmentation_batch.shape[0])

    #ppn = uresnet_ppn_type_point_selector(clabels, res)
    #ppn = uresnet_ppn_type_point_selector(seg_label, res)
    ppn = uresnet_ppn_type_point_selector(data_blob['input_data'][data_idx], result, entry=data_idx)
    if ppn.shape[0] == 0:
        return [(rows_gt_names, rows_gt_values), (rows_pred_names, rows_pred_values)]

    ppn_voxels = ppn[:, 1:4]
    #ppn_score = ppn[:, 4]
    ppn_score = ppn[:, 5]
    #ppn_occupancy = ppn[:, 5]
    ppn_occupancy = ppn[:, 6]
    #ppn_type = ppn[:, 11]
    ppn_type = ppn[:, 12]

    d = cdist(points_label[:, 1:4], ppn_voxels)

    d_pred_to_closest_true = d.min(axis=0)
    pred_to_closest_true_coords = points_label[d.argmin(axis=0)]
    d_true_to_closest_pred = d.min(axis=1)
    true_seg_voxels = seg_label#[batch_mask]#.numpy()
    true_mip_voxels = true_seg_voxels[slabels == 1]

    # Loop over true ppn points
    for i, true_point in enumerate(points_label):

        true_point_coord = true_point[1:4]#.cpu().numpy()
        true_point_type = true_point[4]
        true_point_idx = true_point[5].astype(np.int64)
        # print(len(particles[data_idx]), true_point_idx, true_point)
        p = particles[data_idx][true_point_idx]

        rows_gt_names.append(('Class',
                        'pdg',
                        'min_distance',
                        'x',
                        'y',
                        'z'))
        rows_gt_values.append((int(true_point_type),
                             p.pdg_code(),
                             d_true_to_closest_pred[i],
                             true_point_coord[0],
                             true_point_coord[1],
                             true_point_coord[2]))

    for i, pred_point in enumerate(ppn_voxels):
        pred_point_type, pred_point_score = ppn_type[i], ppn_score[i]
        x, y, z = pred_point
        closest_x, closest_y, closest_z = pred_to_closest_true_coords[i][1:4]
        segmentation_voxels = clabels[:, 1:4][pred_seg == pred_point_type]
        if segmentation_voxels.shape[0] > 0:
            d_same_type = pairwise_distances(
                torch.Tensor(pred_point).view(1, -1),
                torch.Tensor(segmentation_voxels)).numpy()
            d_same_type_closest = d_same_type.min(axis=1)[0]
        else:
            d_same_type_closest = -1
        # points_label_track = points_label[points_label[:, 4] == 1]
        if true_mip_voxels.shape[0] > 0:
            d_mip = pairwise_distances(
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
                          'closest_x', 'closest_y', 'closest_z') + tuple(['score_type_%d' % c for c in range(num_classes)]))
        rows_pred_values.append((int(pred_point_type),
                             float(pred_point_score),
                             float(d_pred_to_closest_true[i]),
                             float(d_same_type_closest),
                             float(d_closest_mip),
                             ppn_voxels[i][0],
                             ppn_voxels[i][1],
                             ppn_voxels[i][2],
                             closest_x, closest_y, closest_z) + tuple(ppn[i, 7:12]))

    return [(rows_gt_names, rows_gt_values), (rows_pred_names, rows_pred_values)]
