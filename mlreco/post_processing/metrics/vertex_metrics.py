import numpy as np

from mlreco.post_processing import post_processing
from mlreco.utils.metrics import *
from mlreco.utils.gnn.network import get_fragment_edges
from mlreco.utils.gnn.cluster import get_cluster_label_np, get_momenta_label_np


def extent(voxels):
    centroid = voxels[:, :3].mean(axis=0)
    return np.linalg.norm(voxels[:, :3] - centroid, axis=1)


@post_processing('vertex-metrics',
                ['seg_label', 'clust_data', 'particles', 'kinematics'],
                ['node_pred_vtx', 'clusts'])
def vertex_metrics(cfg, module_cfg, data_blob, res, logdir, iteration,
                    seg_label=None, clust_data=None, particles=None, kinematics=None,
                    node_pred_vtx=None, clusts=None, data_idx=None, **kwargs):
    spatial_size = module_cfg.get('spatial_size', 768)
    vtx_col = module_cfg.get('vtx_col', 9)
    vtx_positives_col = module_cfg.get('vtx_positives_col', 12)

    node_pred_vtx = node_pred_vtx[data_idx]
    clusts = clusts[data_idx]

    node_x_vtx = get_cluster_label_np(kinematics[data_idx], clusts, column=vtx_col)
    node_y_vtx = get_cluster_label_np(kinematics[data_idx], clusts, column=vtx_col+1)
    node_z_vtx = get_cluster_label_np(kinematics[data_idx], clusts, column=vtx_col+2)

    node_assn_vtx = np.stack([node_x_vtx, node_y_vtx, node_z_vtx], axis=1)
    node_assn_vtx = node_assn_vtx/spatial_size

    good_index = np.all(np.abs(node_assn_vtx) <= 1., axis=1)

    positives = []
    for c in clusts:
        positives.append(kinematics[data_idx][c, vtx_positives_col].max().item())
    positives = np.array(positives)

    n_clusts_vtx = (good_index).sum()
    n_clusts_vtx_positives = (good_index & positives.astype(bool)).sum()

    node_pred_vtx = node_pred_vtx[good_index]
    node_assn_vtx = node_assn_vtx[good_index]
    positives = positives[good_index]

    pred_positives = np.argmax(node_pred_vtx[:, 3:], axis=1)
    accuracy_positives = (pred_positives == positives).sum() / len(positives)
    # SMAPE metric
    accuracy_position = np.sum(1. - np.abs(node_pred_vtx[positives.astype(bool), :3]-node_assn_vtx[positives.astype(bool)])/(np.abs(node_assn_vtx[positives.astype(bool)]) + np.abs(node_pred_vtx[positives.astype(bool), :3])))/3.

    row_names = ('accuracy_score', 'num_pred_positives', 'num_true_positives',
                'accuracy_true_positives', 'accuracy_pred_positives',
                'accuracy_position', 'n_clusts_vtx', 'n_clusts_vtx_positives')
    row_values = (accuracy_positives, np.count_nonzero(pred_positives), np.count_nonzero(positives),
                (pred_positives == positives)[positives > 0].sum(), (pred_positives == positives)[pred_positives > 0].sum(),
                accuracy_position, n_clusts_vtx, n_clusts_vtx_positives)

    return row_names, row_values
