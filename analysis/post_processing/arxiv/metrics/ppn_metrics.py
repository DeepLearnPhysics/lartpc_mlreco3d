import numpy as np
from scipy.spatial.distance import cdist
import scipy

from mlreco.post_processing import post_processing
from mlreco.utils.dbscan import dbscan_points
from mlreco.utils.ppn import uresnet_ppn_type_point_selector


@post_processing(['ppn-metrics-gt', 'ppn-metrics-pred'],
                ['input_data', 'seg_label', 'points_label', 'particles', 'clust_data'],
                ['segmentation', 'points', 'mask_ppn2'])
def ppn_metrics(cfg, module_cfg, data_blob, res, logdir, iteration,
                data_idx=None, input_data=None, seg_label=None, points_label=None, particles=None,
                clust_data=None, seg_prediction=None, points=None, mask_ppn2=None, **kwargs):
    # UResNet prediction
    if not 'segmentation' in res: return (), ()
    if not 'points' in res: return (), ()

    rows_gt_names, rows_gt_values = [], []
    rows_pred_names, rows_pred_values = [], []

    segment_label = seg_label
    clusters = clust_data
    attention = mask_ppn2
    num_classes = module_cfg.get('num_classes', 5)
    coords_col = module_cfg.get('coords_col', (1, 4))

    pool_op = np.max
    if module_cfg.get('pool_op', 'max') == 'mean':
        pool_op = np.mean

    # UResNet output
    predictions = seg_prediction
    label = segment_label[data_idx][:, -1]

    # Remove deltas from true points
    delta = 3
    #points_label_idx = points_label[data_idx][points_label[data_idx][:, -1] != delta]
    points_label_idx = points_label[data_idx]
    # type and idx in this order = -2, -1
    # print(np.unique(points_label_idx[:, -2]), np.unique(points_label_idx[:, -1]))

    ppn_voxels = points[data_idx][:, :3] + 0.5 + input_data[data_idx][:, coords_col[0]:coords_col[1]]
    ppn_score  = scipy.special.softmax(points[data_idx][:, 3:5], axis=1)[:, 1]
    ppn_type   = scipy.special.softmax(points[data_idx][:, 5:], axis=1)

    ppn_mask = (attention[data_idx][:, 0]==1) & (ppn_score > 0.5)

    mode = module_cfg.get('mode', 'select')
    seg_label_col = module_cfg.get('seg_label_col', -2)
    cluster_col = module_cfg.get('cluster_col', -1)

    if mode == 'simple':
        ppn_voxels = ppn_voxels[ppn_mask]
        ppn_score  = ppn_score[ppn_mask]
        ppn_type   = ppn_type[ppn_mask]

        # all_voxels, all_scores, all_types, all_ocupancy = [], [], [], []
        # clusts = dbscan_points(ppn_voxels, epsilon=1.99, minpts=1)
        # for c in clusts:
        #     all_voxels.append(np.mean(ppn_voxels[c], axis=0))
        #     all_scores.append(pool_op(ppn_score[c], axis=0))
        #     all_types.append(pool_op(ppn_type[c], axis=0))
        #     all_occupancy.append(len(c))
        # ppn_voxels = np.stack(all_voxels, axis=0)
        # ppn_score = np.stack(all_scores)
        # ppn_type = np.stack(all_types)
        # ppn_occupancy = np.stack(all_occupancy)
        ppn_occupancy = np.ones((ppn_score.shape[0],))
    else:
        if mode == 'no_type':
            ppn = uresnet_ppn_type_point_selector(input_data[data_idx], res, entry=data_idx, score_threshold=0.5, window_size=3, type_threshold=2, enforce_type=False)
        else:
            ppn = uresnet_ppn_type_point_selector(input_data[data_idx], res, entry=data_idx, score_threshold=0.5, window_size=3, type_threshold=2)

        if ppn.shape[0] == 0:
            return [([], []), ([], [])]

        # Remove delta from predicted points
        #ppn = ppn[ppn[:, -1] != delta]
        #ppn = ppn[ppn[:, -3] < 0.1]
        #print(ppn.shape, ppn[:5])
        ppn_voxels = ppn[:, 1:4]
        ppn_score = ppn[:, 5]
        ppn_occupancy = ppn[:, 6]
        ppn_type = ppn[:, 7:(7+num_classes)]#np.repeat(ppn[:, -1][:, None], num_classes, axis=1)
        #print('ppn_type shape', ppn_type.shape, ppn.shape)
    #print(ppn_voxels.shape, ppn_score.shape, ppn_type.shape)

    # Metrics now
    # Distance to closest true point (regardless of type)
    # Ignore points predicted as delta for this part
    no_delta = ppn_type[:, 3] < 0.5
    d = cdist(ppn_voxels, points_label_idx[:, :3])
    distance_to_closest_true_point = d.min(axis=1)
    distance_to_closest_true_point_nodelta = d[:, points_label_idx[:, seg_label_col] != 3].min(axis=1)
    num_voxels_closest_true_point = np.array([particles[data_idx][int(points_label_idx[j, cluster_col])].num_voxels() for j in d.argmin(axis=1)])
    if clusters is not None:
        num_voxels_cluster_closest_true_point = np.array([np.count_nonzero(clusters[data_idx][:, -1] == int(points_label_idx[j, cluster_col])) for j in d.argmin(axis=1)])
    else:
        num_voxels_cluster_closest_true_point = -1 * np.ones(ppn_voxels.shape[0],)

    distance_to_closest_true_point_type = []
    distance_to_closest_true_pix_type = []
    distance_to_closest_pred_pix_type = []
    closest_true_coords = []
    # num_voxels_closest_true_point = []
    # num_voxels_cluster_closest_true_point = []
    for c in range(num_classes):
        true_mask = points_label_idx[:, seg_label_col] == c
        d = cdist(ppn_voxels, points_label_idx[true_mask][:, coords_col[0]:coords_col[1]])
        #print(d.shape)
        if d.shape[1] > 0:
            distance_to_closest_true_point_type.append(d.min(axis=1))
            closest_true_coords.append(points_label_idx[true_mask][d.argmin(axis=1)][:, coords_col[0]:coords_col[1]])
            # print(particles[data_idx])
            # print(d.argmin(axis=1))
            # print(points_label_idx[d.argmin(axis=1), -1])
            # print(int(points_label_idx[d.argmin(axis=1), -1]))
            # num_voxels_closest_true_point.append(particles[data_idx][int(points_label_idx[d.argmin(axis=1), -1])].num_voxels())
            # if clusters is not None:
            #     num_voxels_cluster_closest_true_point.append(clusters[data_idx][int(points_label_idx[d.argmin(axis=1), -1])].shape[0])
            # else:
            #     num_voxels_cluster_closest_true_point.append(-1 * np.ones(ppn_voxels.shape[0],))
        else:
            distance_to_closest_true_point_type.append(-1 * np.ones(ppn_voxels.shape[0],))
            closest_true_coords.append(-1 * np.ones((ppn_voxels.shape[0], 3)))
            # num_voxels_closest_true_point.append(-1 * np.ones(ppn_voxels.shape[0],))
            # num_voxels_cluster_closest_true_point.append(-1 * np.ones(ppn_voxels.shape[0],))
        d = cdist(ppn_voxels, input_data[data_idx][segment_label[data_idx][:, -1] == c][:, coords_col[0]:coords_col[1]])
        if d.shape[1] > 0:
            distance_to_closest_true_pix_type.append(d.min(axis=1))
        else:
            distance_to_closest_true_pix_type.append(-1 * np.ones(ppn_voxels.shape[0],))
        d = cdist(ppn_voxels, input_data[data_idx][predictions[data_idx] == c][:, coords_col[0]:coords_col[1]])
        if d.shape[1] > 0:
            distance_to_closest_pred_pix_type.append(d.min(axis=1))
        else:
            distance_to_closest_pred_pix_type.append(-1 * np.ones(ppn_voxels.shape[0],))
    distance_to_closest_true_point_type = np.array(distance_to_closest_true_point_type)
    distance_to_closest_true_pix_type = np.array(distance_to_closest_true_pix_type)
    distance_to_closest_pred_pix_type = np.array(distance_to_closest_pred_pix_type)
    closest_true_coords = np.concatenate(closest_true_coords, axis=0)
    for i in range(ppn_voxels.shape[0]):
        rows_pred_names.append(('distance_to_closest_true_point', 'distance_to_closest_true_point_nodelta', 'num_voxels', 'num_voxels_cluster',
                        'score', 'x', 'y', 'z', 'type', 'occupancy', 'closest_x', 'closest_y', 'closest_z') + tuple(['distance_to_closest_true_point_type_%d' % c for c in range(num_classes)]) + tuple(['score_type_%d' % c for c in range(num_classes)]) + tuple(['distance_to_closest_true_pix_type_%d' % c for c in range(num_classes)]) + tuple(['distance_to_closest_pred_pix_type_%d' % c for c in range(num_classes)]))
        rows_pred_values.append((distance_to_closest_true_point[i], distance_to_closest_true_point_nodelta[i], num_voxels_closest_true_point[i], num_voxels_cluster_closest_true_point[i],
                        ppn_score[i], ppn_voxels[i, 0], ppn_voxels[i, 1], ppn_voxels[i, 2], np.argmax(ppn_type[i]), ppn_occupancy[i], closest_true_coords[i, 0], closest_true_coords[i, 1], closest_true_coords[i, 2]) + tuple(distance_to_closest_true_point_type[:, i]) + tuple(ppn_type[i]) + tuple(distance_to_closest_true_pix_type[:, i]) + tuple(distance_to_closest_pred_pix_type[:, i]))

    # Distance to closest pred point (regardless of type)
    d = cdist(ppn_voxels, points_label_idx[:, coords_col[0]:coords_col[1]])
    #print(d.shape)
    distance_to_closest_pred_point = d.min(axis=0)
    distance_to_closest_pred_point_nodelta = d[ppn_type[:, 3] < 0.5, :].min(axis=0)
    closest_pred_index = d.argmin(axis=0)
    score_of_closest_pred_point = ppn_score[closest_pred_index]
    types_of_closest_pred_point = ppn_type[closest_pred_index]
    closest_pred_coords = ppn_voxels[closest_pred_index, :3]
    #print(closest_pred_coords.shape, points_label_idx.shape, score_of_closest_pred_point.shape, types_of_closest_pred_point.shape)
    # closest pred point with type score >0.9

    distance_to_closest_true_pix_type = []
    distance_to_closest_pred_pix_type = []
    distance_to_closest_pred_point_type = []
    for c in range(num_classes):
        d2 = cdist(points_label_idx[:, coords_col[0]:coords_col[1]], input_data[data_idx][segment_label[data_idx][:, -1] == c][:, coords_col[0]:coords_col[1]])
        if d2.shape[1] > 0:
            distance_to_closest_true_pix_type.append(d2.min(axis=1))
        else:
            distance_to_closest_true_pix_type.append(-1 * np.ones(points_label_idx.shape[0],))
        d3 = cdist(points_label_idx[:, coords_col[0]:coords_col[1]], input_data[data_idx][predictions[data_idx] == c][:, coords_col[0]:coords_col[1]])
        if d3.shape[1] > 0:
            distance_to_closest_pred_pix_type.append(d3.min(axis=1))
        else:
            distance_to_closest_pred_pix_type.append(-1 * np.ones(points_label_idx.shape[0],))
        if np.count_nonzero(ppn_type[:, c] > 0.5) > 0:
            #print(c, d[ppn_type[:, c] > 0.5, :].min(axis=0).shape)
            distance_to_closest_pred_point_type.append(d[ppn_type[:, c] > 0.5, :].min(axis=0))
        else:
            distance_to_closest_pred_point_type.append(-1 * np.ones(points_label_idx.shape[0],))
    distance_to_closest_true_pix_type = np.array(distance_to_closest_true_pix_type)
    distance_to_closest_pred_pix_type = np.array(distance_to_closest_pred_pix_type)
    distance_to_closest_pred_point_type = np.array(distance_to_closest_pred_point_type)
    #print(distance_to_closest_pred_point_type.shape)

    for i in range(points_label_idx.shape[0]):
        #print(d.shape, ppn_voxels.shape,  ppn_type.shape, ppn_type[:, int(points_label_idx[i, -2])].shape)
        local_d = d[ppn_type[:, int(points_label_idx[i, seg_label_col])] > 0.5, i]
        closest_pred_point_same_type = 1000000
        if local_d.shape[0] > 0:
            closest_pred_point_same_type = local_d.min()

        num_voxels, energy_deposit, num_voxels_cluster = -1, -1, -1
        if particles is not None:
            num_voxels = particles[data_idx][int(points_label_idx[i, cluster_col])].num_voxels()
            energy_deposit = particles[data_idx][int(points_label_idx[i, cluster_col])].energy_deposit()
            if clusters is not None:
                num_voxels_cluster = np.count_nonzero(clusters[data_idx][:, -1] == int(points_label_idx[i, cluster_col]))
        # Whether this point is already missed in mask_ppn2 or not
        is_in_attention = cdist(input_data[data_idx][ppn_mask][:, coords_col[0]:coords_col[1]], [np.floor(points_label_idx[i, coords_col[0]:coords_col[1]])]).min(axis=0) < 1.
        rows_gt_names.append(('distance_to_closest_pred_point', 'distance_to_closest_pred_point_nodelta', 'type', 'score_of_closest_pred_point', 'x', 'y', 'z', 'closest_x', 'closest_y', 'closest_z',
                        'attention', 'particle_idx', 'num_voxels', 'num_voxels_cluster', 'energy_deposit', 'distance_to_closest_pred_point_same_type') + tuple(['distance_to_closest_pred_point_type_%d' % c for c in range(num_classes)]) + tuple(['type_of_closest_pred_point_%d' % c for c in range(num_classes)]) + tuple(['distance_to_closest_true_pix_type_%d' % c for c in range(num_classes)]) + tuple(['distance_to_closest_pred_pix_type_%d' % c for c in range(num_classes)]))
        rows_gt_values.append((distance_to_closest_pred_point[i], distance_to_closest_pred_point_nodelta[i], points_label_idx[i, seg_label_col], score_of_closest_pred_point[i],
                    points_label_idx[i, 0], points_label_idx[i, 1], points_label_idx[i, 2], closest_pred_coords[i, 0], closest_pred_coords[i, 1], closest_pred_coords[i, 2],
                        int(is_in_attention), points_label_idx[i, cluster_col], num_voxels, num_voxels_cluster, energy_deposit, closest_pred_point_same_type) + tuple(distance_to_closest_pred_point_type[:, i]) + tuple(types_of_closest_pred_point[i]) + tuple(distance_to_closest_true_pix_type[:, i]) + tuple(distance_to_closest_pred_pix_type[:, i]))

    return [(rows_gt_names, rows_gt_values), (rows_pred_names, rows_pred_values)]
