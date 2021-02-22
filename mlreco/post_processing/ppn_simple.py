import numpy as np
from scipy.spatial.distance import cdist
import scipy
import os
from mlreco.utils import CSVData
from mlreco.utils.dbscan import dbscan_points
from mlreco.utils.ppn import uresnet_ppn_point_selector, uresnet_ppn_type_point_selector
import torch


def pairwise_distances(v1, v2):
    v1_2 = v1.unsqueeze(1).expand(v1.size(0), v2.size(0), v1.size(1)).double()
    v2_2 = v2.unsqueeze(0).expand(v1.size(0), v2.size(0), v1.size(1)).double()
    return torch.sqrt(torch.pow(v2_2 - v1_2, 2).sum(2))


def ppn_simple(cfg, data_blob, res, logdir, iteration):
    # UResNet prediction

    # Get the relevant data products
    index = data_blob['index']
    seg_label = data_blob['segment_label'][0]
    clust_data = data_blob['cluster_label'][0]
    particles = data_blob['particles_label'][0]

    ppn_layers = res['ppn_layers'][0]
    ppn_coords = res['ppn_coords'][0]
    points = res['points'][0]
    segmentation = res['segmentation'][0]

    ppn_resolution = cfg['model']['modules']['ppn_loss']['ppn_resolution']
    batch_index = ppn_coords[-1][:, 0]

    if iteration:
        append = True
    else:
        append = False

    fout_gt=CSVData(os.path.join(logdir, 'ppn-metrics-gt.csv'), append=append)
    fout_pred=CSVData(os.path.join(logdir, 'ppn-metrics-pred.csv'), append=append)
    fout_all=CSVData(os.path.join(logdir, 'ppn-metrics.csv'), append=append)

    # Loop over events
    for data_idx, tree_idx in enumerate(index):

        batch_mask = batch_index == data_idx
        slabels = seg_label[batch_mask][:, -1].int().numpy()
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
       
        pred_seg = np.argmax(segmentation_batch, axis=1).astype(int)
        acc_seg = np.sum(pred_seg == slabels) / float(segmentation_batch.shape[0])


        for c in np.unique(slabels):
            
            if int(c) > 2:
                continue


            class_mask = slabels == c
            num_true = np.sum(pred_seg[class_mask] == slabels[class_mask])
            num_total = slabels[class_mask].shape[0]

            d = pairwise_distances(points_label[:, 1:4].float(), torch.Tensor(coords_layer[:, 1:4][class_mask])).numpy()
            positives = (d < ppn_resolution).any(axis=0)
            if (np.sum(positives) < 1):
                continue
            predicted_positives = pixel_score[class_mask] > 0
            point_score_acc = np.sum(positives == predicted_positives) / float(pixel_score[class_mask].shape[0])


            positive_masked = pixel_pred[class_mask][pixel_score[class_mask] > 0]

            d = pairwise_distances(points_label[:, 1:4].float(), torch.Tensor(positive_masked)).numpy()

            if d.shape[0] == 0 or d.shape[1] == 0:
                continue

            d_pred_to_true = d.min(axis=0)
            d_true_to_pred = d.min(axis=1)
            num_t2p = np.sum(d_true_to_pred < ppn_resolution)
            num_p2t = np.sum(d_pred_to_true < ppn_resolution)

            pixel_seg = pixel_logits[class_mask][pixel_score[class_mask] > 0]
            pixel_seg_masked = pixel_seg[d_pred_to_true < ppn_resolution]

            pixel_pred_class = np.argmax(pixel_seg_masked, axis=1)
            pixel_true_class = slabels[class_mask][pixel_score[class_mask] > 0][d_pred_to_true < ppn_resolution]
    
            purity = np.sum(pixel_pred_class == c) / float(pixel_pred_class.shape[0])
            efficiency = np.sum(pixel_true_class == c) / float(pixel_true_class.shape[0])

            d_pred = pairwise_distances(torch.Tensor(coords_layer[:, 1:4][pred_seg == c]), torch.Tensor(positive_masked)).numpy()

            row = (tree_idx, int(c), acc_seg, point_score_acc, d_pred_to_true.mean(), \
                d_true_to_pred.mean(), num_true, num_total, purity, efficiency)
            #output.append(row)
            fout_all.record(('Index', 'Class', 'seg_acc', 'score_acc', \
                'd_p2t', 'd_t2p', 'num_true_segmentation', 'total_num_voxels', 'purity', 'efficiency'), row)
            fout_all.write()

            if d_pred.shape[0] > 0:
                d_same_type = d_pred.min(axis=0)
                for i, r in enumerate(d_pred_to_true):
                    fout_pred.record(('Index', 'Class', 'min_distance', 'd_pred_to_same_type'), (tree_idx, int(c), r, d_same_type[i]))
                    fout_pred.write()

            for r in d_true_to_pred:
                fout_gt.record(('Index', 'Class', 'min_distance'), (tree_idx, int(c), r))
                fout_gt.write()

    # fout_gt.close()
    # fout_pred.close()
    # fout_all.close()