import numpy as np
import scipy
from scipy.spatial.distance import cdist
from mlreco.post_processing import post_processing


@post_processing('doublet_metrics',
                ['input_data', 'nhits', 'seg_label_full'],
                ['segmentation'])
def doublet_metrics(cfg, module_cfg, data_blob, res, logdir, iteration,
                        data_idx=None, input_data=None,
                        segmentation=None, nhits=None, seg_label_full=None, **kwargs):
    import torch
    row_names, row_values = [], []
    data = input_data[data_idx]
    label = seg_label_full[data_idx][:,-1]
    nhits = nhits[data_idx][:, -1]

    num_classes_ghost = segmentation[data_idx].shape[1]
    num_classes_semantic = module_cfg.get('num_classes_semantic', 5)
    num_ghost_points = np.count_nonzero(label == num_classes_semantic)
    num_nonghost_points = np.count_nonzero(label < num_classes_semantic)

    shower_label = module_cfg.get('shower_label', 0)
    edep_col     = module_cfg.get('edep_col', -2)
    assert shower_label >= 0 and shower_label < num_classes_semantic

    row_names += ['num_ghost_points', 'num_nonghost_points']
    row_values += [num_ghost_points, num_nonghost_points]

    ghost_predictions = np.argmax(res['segmentation'][data_idx], axis=1)
    mask = ghost_predictions == 0

    # Fraction of ghost points predicted as ghost points
    ghost2ghost = (ghost_predictions[label == num_classes_semantic] == 1).sum() / float(num_ghost_points)
    # Fraction of true non-ghost points predicted as true non-ghost points
    nonghost2nonghost = (ghost_predictions[label < num_classes_semantic] == 0).sum() / float(num_nonghost_points)
    row_names += ["ghost2ghost", "nonghost2nonghost"]
    row_values += [ghost2ghost, nonghost2nonghost]

    for c in range(num_classes_semantic):
        row_names += ['num_true_pix_class_%d' % c]
        row_values += [np.count_nonzero(label == c)]
        #print(c, np.count_nonzero(label == c), np.count_nonzero((label == c) & (ghost_predictions == 1)))
        row_names += ['num_pred_pix_class_%d_%d' % (c, x) for x in range(num_classes_ghost)]
        row_values += [np.count_nonzero((label == c) & (ghost_predictions == x)) for x in range(num_classes_ghost)]

        row_names += ['num_pred_pix_doublets_class_%d_%d' % (c, x) for x in range(num_classes_ghost)]
        row_values += [np.count_nonzero((label == c) & (ghost_predictions == x) & (nhits == 2)) for x in range(num_classes_ghost)]

        row_names += ['num_pred_pix_triplets_class_%d_%d' % (c, x) for x in range(num_classes_ghost)]
        row_values += [np.count_nonzero((label == c) & (ghost_predictions == x) & (nhits == 3)) for x in range(num_classes_ghost)]

        row_names += ['num_doublets_class_%d' % c, 'num_triplets_class_%d' % c]
        row_values += [np.count_nonzero((label == c) & (nhits == 2)), np.count_nonzero((label == c) & (nhits == 3))]

    row_names += ['num_doublets_ghost', 'num_triplets_ghost']
    row_values += [np.count_nonzero((label == num_classes_semantic) & (nhits == 2)), np.count_nonzero((label == num_classes_semantic) & (nhits == 3))]

    row_names += ['num_doublets_ghost_%d' % x for x in range(num_classes_ghost)]
    row_values += [np.count_nonzero((label == num_classes_semantic) & (nhits == 2) & (ghost_predictions == x)) for x in range(num_classes_ghost)]

    row_names += ['num_triplets_ghost_%d' % x for x in range(num_classes_ghost)]
    row_values += [np.count_nonzero((label == num_classes_semantic) & (nhits == 3) & (ghost_predictions == x)) for x in range(num_classes_ghost)]

    # Record shower voxels sum in true mask and in (true & pred) mask
    # to see if we lose a significant amount of energy
    # (might be offset by true ghost predicted as nonghost)
    row_names += ['shower_true_voxel_sum', 'shower_true_pred_voxel_sum']
    row_values += [data[label == shower_label, edep_col].sum(), data[(label == shower_label) & mask, edep_col].sum()]

    row_names += ['shower_true_voxel_sum_doublets', 'shower_true_pred_voxel_sum_doublets']
    row_values += [data[(label == shower_label) & (nhits == 2), edep_col].sum(), data[(label == shower_label) & mask & (nhits == 2), edep_col].sum()]

    row_names += ['shower_true_voxel_sum_triplets', 'shower_true_pred_voxel_sum_triplets']
    row_values += [data[(label == shower_label) & (nhits == 3), edep_col].sum(), data[(label == shower_label) & mask & (nhits == 3), edep_col].sum()]

    return tuple(row_names), tuple(row_values)
