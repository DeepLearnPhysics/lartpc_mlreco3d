import numpy as np
from mlreco.utils import utils


def deghosting_metrics(data_blob, res, cfg, idx):
    """
    Some useful metrics to measure deghosting performance
    """
    csv_logger = utils.CSVData("%s/deghosting_metrics-%.07d.csv" % (cfg['training']['log_dir'], idx))

    model_cfg = cfg['model']

    segmentation_all = res['segmentation'][0]  # (N, 5)
    predictions_all = np.argmax(segmentation_all, axis=1)
    ghost_all = res['ghost'][0]  # (N, 2)
    data_all = data_blob['input_data'][0][0]
    label_all = data_blob['segment_label'][0][0][:, -1]

    # First mask ghost points in predictions
    ghost_predictions = np.argmax(ghost_all, axis=1)
    mask = ghost_predictions == 0

    batch_ids = np.unique(data_all[:, 3])
    num_classes = segmentation_all.shape[1]
    for b in batch_ids:
        batch_index = data_all[:, 3] == b
        label = label_all[batch_index]

        # Accuracy for ghost prediction
        ghost_acc = ((ghost_predictions[batch_index] == 1) == (label == 5)).sum() / float(label.shape[0])

        # Accuracy for 5 types, global
        uresnet_acc = (label[label < 5] == predictions_all[batch_index][label < 5]).sum() / float(np.count_nonzero(label < 5))

        # Class-wise nonzero accuracy for 5 types, based on true mask
        acc, num_true_pix, num_pred_pix = [], [], []
        num_pred_pix_true = []
        ghost_false_positives, ghost_true_positives = [], []
        for c in range(num_classes):
            class_mask = label == c
            class_predictions = predictions_all[batch_index][mask[batch_index] & class_mask]
            acc.append((class_predictions == c).sum() / float(class_predictions.shape[0]))
            num_true_pix.append(np.count_nonzero(class_mask))
            num_pred_pix.append(np.count_nonzero(predictions_all[batch_index & mask] == c))
            num_pred_pix_true.append(np.count_nonzero(class_predictions == c))
            ghost_false_positives.append(np.count_nonzero(ghost_predictions[batch_index][class_mask] == 1))
            ghost_true_positives.append(np.count_nonzero(ghost_predictions[batch_index][class_mask] == 0))
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
        csv_logger.record(('ghost_acc', 'uresnet_acc'),
                          (ghost_acc, uresnet_acc))
        csv_logger.write()
    csv_logger.close()
