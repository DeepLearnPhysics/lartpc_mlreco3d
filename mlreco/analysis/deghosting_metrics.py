import numpy as np
from mlreco.utils import utils


def deghosting_metrics(data_blob, res, cfg, idx):
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
    csv_logger = utils.CSVData("%s/deghosting_metrics-%.07d.csv" % (cfg['training']['log_dir'], idx))

    model_cfg = cfg['model']['modules']['uresnet_lonely']

    ghost = model_cfg['ghost']

    segmentation_all = res['segmentation'][0]  # (N, 5)
    predictions_all = np.argmax(segmentation_all, axis=1)
    data_all = data_blob['input_data'][0][0]
    label_all = data_blob['segment_label'][0][0][:, -1]
    idx_all = data_blob['index'][0][0]

    if ghost:
        ghost_all = res['ghost'][0]  # (N, 2)
        # First mask ghost points in predictions
        # 0 = non ghost, 1 = ghost
        ghost_predictions = np.argmax(ghost_all, axis=1)
        mask = ghost_predictions == 0

    batch_ids = np.unique(data_all[:, 3])
    num_classes = segmentation_all.shape[1]
    for b in batch_ids:
        batch_index = data_all[:, 3] == b
        event_index = idx_all[int(b)][0]  # Assuming no minibatching here
        label = label_all[batch_index]
        num_ghost_points = np.count_nonzero(label == 5)
        num_nonghost_points = np.count_nonzero(label < 5)
        csv_logger.record(('num_ghost_points', 'num_nonghost_points', 'idx'),
                          (num_ghost_points, num_nonghost_points, event_index))
        if ghost:  # 5+2 type
            # Accuracy for ghost prediction
            # 0 = non ghost, 1 = ghost
            # Fraction of true points predicted correctly
            ghost_acc = ((ghost_predictions[batch_index] == 1) == (label == 5)).sum() / float(label.shape[0])
            # Fraction of ghost points predicted as ghost points
            ghost2ghost = (ghost_predictions[batch_index][label == 5] == 1).sum() / float(num_ghost_points)
            # Fraction of true non-ghost points predicted as true non-ghost points
            nonghost2nonghost = (ghost_predictions[batch_index][label < 5] == 0).sum() / float(num_nonghost_points)
            csv_logger.record(("ghost2ghost", "nonghost2nonghost"),
                              (ghost2ghost, nonghost2nonghost))

            # Accuracy for 5 types, global
            uresnet_acc = (label[label < 5] == predictions_all[batch_index][label < 5]).sum() / float(np.count_nonzero(label < 5))
            csv_logger.record(('ghost_acc', 'uresnet_acc'),
                              (ghost_acc, uresnet_acc))
            # Class-wise nonzero accuracy for 5 types, based on true mask
            acc, num_true_pix, num_pred_pix = [], [], []
            num_pred_pix_true = []
            ghost_false_positives, ghost_true_positives = [], []
            for c in range(num_classes):
                class_mask = label == c
                class_predictions = predictions_all[batch_index][class_mask]
                # Fraction of pixels in this class predicted correctly
                acc.append((class_predictions == c).sum() / float(class_predictions.shape[0]))
                # Pixel counts
                num_true_pix.append(np.count_nonzero(class_mask))
                num_pred_pix.append(np.count_nonzero(predictions_all[batch_index & mask] == c))
                num_pred_pix_true.append(np.count_nonzero(class_predictions == c))
                # Fraction of pixels in this class (wrongly) predicted as ghost
                ghost_false_positives.append(np.count_nonzero(ghost_predictions[batch_index][class_mask] == 1))
                # Fraction of pixels in this class (correctly) predicted as nonghost
                ghost_true_positives.append(np.count_nonzero(ghost_predictions[batch_index][class_mask] == 0))
                # confusion matrix
                for c2 in range(num_classes):
                    csv_logger.record(('confusion_%d_%d' % (c, c2),),
                                      (((class_predictions == c2) & (ghost_predictions[batch_index][class_mask] == 0)).sum(),))
            csv_logger.record(['acc_class%d' % c for c in range(num_classes)],
                              acc)
            csv_logger.record(['num_true_pix_class%d' % c for c in range(num_classes)],
                              num_true_pix)
            csv_logger.record(['num_pred_pix_class%d' % c for c in range(num_classes)],
                              num_pred_pix)
            csv_logger.record(['num_pred_pix_true_class%d' % c for c in range(num_classes)],
                              num_pred_pix_true)
            csv_logger.record(['ghost_false_positives_class%d' % c for c in range(num_classes)],
                              ghost_false_positives)
            csv_logger.record(['ghost_true_positives_class%d' % c for c in range(num_classes)],
                              ghost_true_positives)

        else:
            if 'ghost_label' not in model_cfg or model_cfg['ghost_label'] == -1:  # 6 types
                ghost2ghost = (predictions_all[batch_index][label == 5] == 5).sum() / float(num_ghost_points)
                nonghost2nonghost = (predictions_all[batch_index][label < 5] < 5).sum() / float(num_nonghost_points)
                csv_logger.record(("ghost2ghost", "nonghost2nonghost"),
                                  (ghost2ghost, nonghost2nonghost))
                # 6 types confusion matrix
                for c in range(num_classes):
                    for c2 in range(num_classes):
                        # Fraction of points of class c, predicted as c2
                        x = (predictions_all[batch_index][label == c] == c2).sum() / float(np.count_nonzero(label == c))
                        csv_logger.record(('confusion_%d_%d' % (c, c2),), (x,))
            else:  # ghost only segmentation
                ghost2ghost = (predictions_all[batch_index][label == 5] == 1).sum() / float(num_ghost_points)
                nonghost2nonghost = (predictions_all[batch_index][label < 5] == 0).sum() / float(num_nonghost_points)
                csv_logger.record(("ghost2ghost", "nonghost2nonghost"),
                                  (ghost2ghost, nonghost2nonghost))
        csv_logger.write()
    csv_logger.close()
