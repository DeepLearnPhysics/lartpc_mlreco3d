import torch
import numpy as np
from scipy.spatial.distance import cdist
import scipy
import os
from mlreco.utils import CSVData
from mlreco.utils.dbscan import dbscan_points
from mlreco.utils.ppn import mink_ppn_selector


def pairwise_distances(v1, v2):
    v1_2 = v1.unsqueeze(1).expand(v1.size(0), v2.size(0), v1.size(1)).double()
    v2_2 = v2.unsqueeze(0).expand(v1.size(0), v2.size(0), v1.size(1)).double()
    return torch.sqrt(torch.pow(v2_2 - v1_2, 2).sum(2))


def ppn_simple(cfg, processor_cfg, data_blob, result, logdir, iteration):
    # UResNet prediction
    print(processor_cfg)
    # Get the relevant data products
    index = data_blob['index']
    seg_label = data_blob['segment_label'][0]
    clust_data = data_blob['cluster_label'][0]
    particles = data_blob['particles_label'][0]

    ppn_layers = result['ppn_layers'][0]
    ppn_coords = result['ppn_coords'][0]
    points = result['points'][0]
    segmentation = result['segmentation'][0]
    mask_ppn = result['mask_ppn'][0]

    ppn_resolution = cfg['model']['modules']['ppn_loss']['ppn_resolution']
    batch_index = ppn_coords[-1][:, 0]

    if iteration:
        append = True
    else:
        append = False

    print('fout_gt')
    fout_gt=CSVData(os.path.join(logdir, 'ppn-metrics-gt.csv'), append=append)

    print('fout_pred')
    fout_pred=CSVData(os.path.join(logdir, 'ppn-metrics-pred.csv'), append=append)


    print('fout_all')
    fout_all=CSVData(os.path.join(logdir, 'ppn-metrics.csv'), append=append)

    # Loop over events
    for data_idx, tree_idx in enumerate(index):

        batch_mask = batch_index == data_idx
        slabels = seg_label[batch_mask][:, -1].int().numpy()
        clabels = clust_data[batch_mask].float().numpy()
        lowE_mask = slabels != 4
        ppn_score_layer = ppn_layers[-1][batch_index == data_idx]
        coords_layer = ppn_coords[-1][batch_index == data_idx]

        pixel_pred = points[batch_index == data_idx][:, 0:3] + coords_layer[:, 1:4]
        pixel_score = points[batch_index == data_idx][:, -1]
        pixel_logits = points[batch_index == data_idx][:, 3:8]
        

        points_label = particles[particles[:, 0] == data_idx]

        # Initialize log if one per event
        points_batch = points[batch_mask]
        segmentation_batch = segmentation[batch_mask]
        mask_ppn_batch = mask_ppn[-1][batch_mask]

        res = {
            'points': points_batch,
            'mask_ppn': mask_ppn_batch,
            'segmentation': segmentation_batch,
            'ppn_score': scipy.special.expit(pixel_score)
        }
       
        pred_seg = np.argmax(segmentation_batch, axis=1).astype(int)
        acc_seg = np.sum(pred_seg == slabels) / float(segmentation_batch.shape[0])

        ppn = mink_ppn_selector(clabels, res)
        if ppn.shape[0] == 0:
            continue

        ppn_voxels = ppn[:, :3]
        ppn_score = ppn[:, 4]
        ppn_occupancy = ppn[:, 5]
        ppn_type = ppn[:, 11]

        d = pairwise_distances(points_label[:, 1:4], torch.Tensor(ppn_voxels)).numpy()
        # print(d, d.shape)
        # print("pred_points = ", ppn_voxels.shape)
        # print("true_points = ", points_label.shape)

        d_pred_to_closest_true = d.min(axis=0)
        pred_to_closest_true_coords = points_label.numpy()[d.argmin(axis=0)]
        d_true_to_closest_pred = d.min(axis=1)

        # print(d_true_to_closest_pred, d_true_to_closest_pred.shape)
        # print(d_pred_to_closest_true, d_pred_to_closest_true.shape)
        # print(pred_to_closest_true_coords)
        true_seg_voxels = seg_label[batch_mask].numpy()
        true_mip_voxels = true_seg_voxels[slabels == 1]
        # Loop over true ppn points

        for i, true_point in enumerate(points_label):
            true_point_coord, true_point_type = true_point[1:4].cpu().numpy(), true_point[4]
            # print(int(true_point_type), d_true_to_closest_pred[i])
            fout_gt.record(('Index', 'Class', 'min_distance', 'x', 'y', 'z'), 
                (tree_idx, int(true_point_type), d_true_to_closest_pred[i], true_point_coord[0], true_point_coord[1], true_point_coord[2]))
            # print(true_point_coord[0])
            fout_gt.write()

        for i, pred_point in enumerate(ppn_voxels):
            pred_point_type, pred_point_score = ppn_type[i], ppn_score[i]
            x, y, z = pred_point
            closest_x, closest_y, closest_z = pred_to_closest_true_coords[i][1:4]
            segmentation_voxels = clabels[:, 1:4][pred_seg == pred_point_type]
            if segmentation_voxels.shape[0] > 0:
                d_same_type = pairwise_distances(torch.Tensor(pred_point).view(1, -1), torch.Tensor(segmentation_voxels)).numpy()
                d_same_type_closest = d_same_type.min(axis=1)[0]
            else:
                d_same_type_closest = -1
            # points_label_track = points_label[points_label[:, 4] == 1]
            if true_mip_voxels.shape[0] > 0:
                d_mip = pairwise_distances(torch.Tensor(pred_point).view(1, -1), torch.Tensor(true_mip_voxels[:, 1:4])).numpy()
                d_closest_mip = d_mip.min(axis=1)[0]

            else:
                d_closest_mip = -1
            fout_pred.record(('Index', 'Class', 'Score', 'min_distance', 'd_pred_to_same_type', 'd_true_mip', 'x', 'y', 'z', 'closest_x', 'closest_y', 'closest_z'), \
                (tree_idx, int(pred_point_type), float(pred_point_score), \
                 float(d_pred_to_closest_true[i]), float(d_same_type_closest), float(d_closest_mip),
                 ppn_voxels[i][0], ppn_voxels[i][1], ppn_voxels[i][2], closest_x, closest_y, closest_z))
            # print(ppn_voxels[i][0])
            fout_pred.write()

    fout_gt.close()
    fout_pred.close()
    fout_all.close()