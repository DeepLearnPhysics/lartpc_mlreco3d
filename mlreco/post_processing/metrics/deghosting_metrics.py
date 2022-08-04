import numpy as np
import scipy
from scipy.spatial.distance import cdist
from mlreco.post_processing import post_processing


@post_processing('deghosting_metrics',
                ['input_data', 'seg_label'],
                ['segmentation'])
def deghosting_metrics(cfg, module_cfg, data_blob, res, logdir, iteration,
                        data_idx=None, input_data=None, seg_label=None,
                        segmentation=None, seg_prediction=None, **kwargs):
    """
    Some useful metrics to measure deghosting performance

    Parameters
    ----------
    data_blob: dict
        Input dictionary returned by iotools
    res: dict
        Results from the network, dictionary using `analysis_keys`
    cfg: dict
        Configuration
    idx: int
        Iteration number

    Input
    -----
    Requires the following analysis keys:
    - `segmentation`
    - `ghost` only if 5+2 types architecture for GhostNet
    Requires the following input keys:
    - `input_data`
    - `segment_label`
    Assumes no minibatching

    Output
    ------
    Writes to a CSV file `deghosting_metrics-*`
    """
    import torch
    row_names, row_values = [], []

    deghosting_type = module_cfg.get('method', '5+2')
    assert(deghosting_type in ['5+2','6','2'])

    pcluster = None
    if 'pcluster' in data_blob:
        pcluster = data_blob['pcluster'][data_idx][:, -1]

    data = input_data[data_idx]
    label = seg_label[data_idx][:,-1]
    predictions = seg_prediction[data_idx]
    softmax_predictions = scipy.special.softmax(segmentation[data_idx], axis=1)

    num_classes = segmentation[data_idx].shape[1]
    num_ghost_points = np.count_nonzero(label == 5)
    num_nonghost_points = np.count_nonzero(label < 5)

    row_names += ['num_ghost_points', 'num_nonghost_points']
    row_values += [num_ghost_points, num_nonghost_points]

    if deghosting_type == '5+2':
        # Accuracy for ghost prediction for 5+2
        ghost_predictions = np.argmax(res['ghost'][data_idx], axis=1)
        ghost_softmax = scipy.special.softmax(res['ghost'][data_idx], axis=1)
        mask = ghost_predictions == 0

        if isinstance(label, torch.Tensor):
            label = label.numpy()

        # 0 = non ghost, 1 = ghost
        # Fraction of true points predicted correctly
        ghost_acc = ((ghost_predictions == 1) == (label == 5)).sum() / float(label.shape[0])
        # Fraction of ghost points predicted as ghost points
        ghost2ghost = (ghost_predictions[label == 5] == 1).sum() / float(num_ghost_points)
        # Fraction of true non-ghost points predicted as true non-ghost points
        nonghost2nonghost = (ghost_predictions[label < 5] == 0).sum() / float(num_nonghost_points)
        row_names += ["ghost2ghost", "nonghost2nonghost"]
        row_values += [ghost2ghost, nonghost2nonghost]

        # # Looking at mistakes: true ghost predicted as nonghost
        # # distance from a true ghost point predicted as nonghost, to closest true nonghost point
        # d = cdist(data[(ghost_predictions == 0) & (label == 5), :3], data[label < 5, :3])
        # closest_true_nonghost = d.argmin(axis=1)
        # for d_idx in range(d.shape[0]):
        #     csv_logger2.record(("distance_to_closest_true_nonghost", "semantic_of_closest_true_nonghost", "predicted_semantic",
        #                         "nonghost_softmax"),
        #                        (d[d_idx, closest_true_nonghost[d_idx]], label[label<5][closest_true_nonghost[d_idx]], predictions[(ghost_predictions == 0) & (label == 5)][d_idx],
        #                        ghost_softmax[(ghost_predictions == 0) & (label == 5)][d_idx][0]))
        #     for c in range(num_classes):
        #         csv_logger2.record(("softmax_class%d" %c,),
        #                             (softmax_predictions[(ghost_predictions == 0) & (label == 5)][d_idx][c],))
        #     csv_logger2.write()
        #
        # # Looking at mistakes: true nonghost predicted as ghost
        # d = cdist(data[(ghost_predictions == 1) & (label < 5), :3], data[label == 5, :3])
        # closest_true_ghost = d.argmin(axis=1)
        # for d_idx in range(d.shape[0]):
        #     csv_logger3.record(("distance_to_closest_true_ghost", "semantic",
        #                         "ghost_softmax", "predicted_semantic"),
        #                         (d[d_idx, closest_true_ghost[d_idx]], label[(ghost_predictions == 1) & (label < 5)][d_idx],
        #                         ghost_softmax[(ghost_predictions == 1) & (label < 5)][d_idx][1],
        #                         predictions[(ghost_predictions == 1) & (label < 5)][d_idx]))
        #     for c in range(num_classes):
        #         csv_logger3.record(("softmax_class%d" % c,),
        #                             (softmax_predictions[(ghost_predictions == 1) & (label < 5)][d_idx][c],))
        #     csv_logger3.write()

        # Accuracy for 5 types, global
        uresnet_acc = (label[label < 5] == predictions[label < 5]).sum() / float(np.count_nonzero(label < 5))
        row_names += ['ghost_acc', 'uresnet_acc']
        row_values += [ghost_acc, uresnet_acc]
        # Class-wise nonzero accuracy for 5 types, based on true mask
        acc, num_true_pix, num_pred_pix = [], [], []
        num_pred_pix_true, num_true_pix_pred = [], []
        num_true_deghost_pix, num_original_pix = [], []
        ghost_false_positives, ghost_true_positives = [], []
        ghost2nonghost = []
        for c in range(num_classes):
            class_mask = label == c
            class_predictions = predictions[class_mask]
            # Fraction of pixels in this class predicted correctly
            acc.append((class_predictions == c).sum() / float(class_predictions.shape[0]))
            # Pixel counts
            # Pixels in sparse3d_semantics_reco
            num_true_pix.append(np.count_nonzero(class_mask))
            # Pixels in sparse3d_semantics_reco predicted as nonghost
            num_true_deghost_pix.append(np.count_nonzero(class_mask & mask))
            # Pixels in original pcluster
            if pcluster is not None:
                num_original_pix.append(np.count_nonzero(pcluster == c))
            # Pixels in predictions + nonghost
            num_pred_pix.append(np.count_nonzero(predictions[mask] == c))
            # Pixels in predictions + nonghost that are correctly classified
            num_pred_pix_true.append(np.count_nonzero(class_predictions == c))
            num_true_pix_pred.append(np.count_nonzero(predictions[mask & class_mask] == c))
            # Fraction of pixels in this class (wrongly) predicted as ghost
            ghost_false_positives.append(np.count_nonzero(ghost_predictions[class_mask] == 1))
            # Fraction of pixels in this class (correctly) predicted as nonghost
            ghost_true_positives.append(np.count_nonzero(ghost_predictions[class_mask] == 0))
            # Fraction of true ghost points predicted to belong to this class
            ghost2nonghost.append(np.count_nonzero((label == 5) & (ghost_predictions == 0) & (predictions == c)))
            # confusion matrix
            # pixels predicted as nonghost + should be in class c, but predicted as c2
            for c2 in range(num_classes):
                row_names += ['confusion_%d_%d' % (c, c2)]
                row_values += [((class_predictions == c2) & (ghost_predictions[class_mask] == 0)).sum()]
        row_names += ['acc_class%d' % c for c in range(num_classes)]
        row_values += acc
        row_names += ['num_true_pix_class%d' % c for c in range(num_classes)]
        row_values += num_true_pix
        row_names += ['num_true_deghost_pix_class%d' % c for c in range(num_classes)]
        row_values += num_true_deghost_pix
        if pcluster is not None:
            row_names += ['num_original_pix_class%d' % c for c in range(num_classes)]
            row_values += num_original_pix
        row_names += ['num_pred_pix_class%d' % c for c in range(num_classes)]
        row_values += num_pred_pix
        row_names += ['num_pred_pix_true_class%d' % c for c in range(num_classes)]
        row_values += num_pred_pix_true
        row_names += ['num_true_pix_pred_class%d' % c for c in range(num_classes)]
        row_values += num_true_pix_pred
        row_names += ['ghost_false_positives_class%d' % c for c in range(num_classes)]
        row_values += ghost_false_positives
        row_names += ['ghost_true_positives_class%d' % c for c in range(num_classes)]
        row_values += ghost_true_positives
        row_names += ['ghost2nonghost_class%d' % c for c in range(num_classes)]
        row_values += ghost2nonghost

    elif deghosting_type == '6':
        ghost2ghost = (predictions[label == 5] == 5).sum() / float(num_ghost_points)
        nonghost2nonghost = (predictions[label < 5] < 5).sum() / float(num_nonghost_points)
        row_names += ["ghost2ghost", "nonghost2nonghost"]
        row_values += [ghost2ghost, nonghost2nonghost]
        # 6 types confusion matrix
        for c in range(num_classes):
            for c2 in range(num_classes):
                # Fraction of points of class c, predicted as c2
                x = (predictions[label == c] == c2).sum() / float(np.count_nonzero(label == c))
                row_names += ['confusion_%d_%d' % (c, c2)]
                row_values += [x]
    elif deghosting_type == '2':
        ghost2ghost = (predictions[label == 5] == 1).sum() / float(num_ghost_points)
        nonghost2nonghost = (predictions[label < 5] == 0).sum() / float(num_nonghost_points)
        row_names += ["ghost2ghost", "nonghost2nonghost"]
        row_values += [ghost2ghost, nonghost2nonghost]
    else:
        print('Invalid "deghosting_type" config parameter value:',deghosting_type)
        raise ValueError

    return tuple(row_names), tuple(row_values)
