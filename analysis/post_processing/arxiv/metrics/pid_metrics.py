import numpy as np

from mlreco.post_processing import post_processing
from mlreco.utils.gnn.evaluation import edge_assignment, node_assignment, node_assignment_bipartite, clustering_metrics, edge_assignment_from_graph
from mlreco.utils.cluster.dense_cluster import gaussian_kernel, ellipsoidal_kernel, fit_predict_np, find_cluster_means
from mlreco.utils.metrics import *
from mlreco.utils.gnn.network import get_fragment_edges
from mlreco.utils.gnn.cluster import get_cluster_label, get_momenta_label
from mlreco.post_processing.common import extent


@post_processing('pid-metrics',
                ['kinematics'],
                ['inter_particles', 'node_pred_type'])
def pid_metrics(cfg, module_cfg, data_blob, res, logdir, iteration,
                        data_idx=None, kinematics=None,
                        inter_particles=None,
                        node_pred_type=None,
                        counter=None, **kwargs):
    """
    Compute metrics for PID.

    Parameters
    ----------
    data_blob: dict
        The input data dictionary from iotools.
    res: dict
        The output of the network, formatted using `analysis_keys`.
    cfg: dict
        Configuration.
    logdir: string
        Path to folder where CSV logs can be stored.
    iteration: int
        Current iteration number.

    Notes
    -----
    N/A.
    """
    spatial_size = module_cfg.get('spatial_size', 768)
    coords_col = module_cfg.get('coords_col', (1, 4))

    pred_particles = inter_particles[data_idx]
    node_pred_type = node_pred_type[data_idx] if node_pred_type is not None else None

    node_true_type = get_cluster_label(kinematics[data_idx], pred_particles, column=7) # pdg label
    node_true_cluster_id = get_cluster_label(kinematics[data_idx], pred_particles, column=6) # cluster id
    node_true_nu_id = get_cluster_label(kinematics[data_idx], pred_particles, column=8)
    print("true nu id", node_true_nu_id)
    print(kinematics[data_idx][:5])

    #clust_ids = get_cluster_label(clust_data[data_idx], pred_particles, 5) # or 6 ?
    #print(kinematics[data_idx].shape, pred_particles)
    # Loop over particles
    row_names, row_values = [], []
    for i in range(len(node_true_type)):
        true_cluster_id = node_true_cluster_id[i]
        # p = particles[data_idx][true_cluster_id]
        pred_voxels = kinematics[data_idx][pred_particles[i]]
        true_voxels = kinematics[data_idx][kinematics[data_idx][:, 6] == true_cluster_id]

        true_d = extent(true_voxels)
        pred_d = extent(pred_voxels)
        boundaries = np.min(np.concatenate([true_voxels[:, coords_col[0]:coords_col[1]], spatial_size - true_voxels[:, coords_col[0]:coords_col[1]]], axis=1))

        tuple_names = ('true_cluster_id', 'true_type', 'pred_type', 'distance_to_boundary',
                        'pred_num_voxels', 'pred_sum_voxels',
                        'true_num_voxels', 'true_sum_voxels',
                        'true_spatial_extent', 'true_spatial_std', 'pred_spatial_extent', 'pred_spatial_std',
                        'true_nu_id',)
        tuple_values = (true_cluster_id, node_true_type[i], node_pred_type[i].argmax(), boundaries,
                        len(pred_voxels), pred_voxels[:, 4].sum(),
                        len(true_voxels), true_voxels[:, 4].sum(),
                        true_d.max(), true_d.std(), pred_d.max(), pred_d.std(),
                        node_true_nu_id[i],)


        row_names.append(tuple_names)
        row_values.append(tuple_values)

    return row_names, row_values
