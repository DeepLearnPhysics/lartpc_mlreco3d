import numpy as np

from mlreco.post_processing import post_processing
from mlreco.utils.gnn.evaluation import edge_assignment, node_assignment, node_assignment_bipartite, clustering_metrics, edge_assignment_from_graph
from mlreco.utils.cluster.dense_cluster import gaussian_kernel, ellipsoidal_kernel, fit_predict_np, find_cluster_means
from mlreco.utils.metrics import *
from mlreco.utils.gnn.network import get_fragment_edges
from mlreco.utils.gnn.cluster import get_cluster_label, get_momenta_label
from mlreco.post_processing.common import extent


@post_processing('kinematics-metrics',
                ['clust_data', 'particles', 'kinematics', 'particle_graph'],
                ['kinematics_particles', 'inter_particles', 'node_pred_type', 'node_pred_p', 'flow_edge_pred', 'kinematics_edge_index'])
def kinematics_metrics(cfg, module_cfg, data_blob, res, logdir, iteration,
                        data_idx=None, clust_data=None, particles=None, kinematics=None,
                        particle_graph=None, kinematics_particles=None, inter_particles=None, node_pred_type=None,
                        node_pred_p=None, flow_edge_pred=None, kinematics_edge_index=None, counter=None, **kwargs):
    """
    Compute metrics for particle hierarchy + kinematics stage (GNN).

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
    if kinematics_particles is None and inter_particles is None:
        raise Exception("kinematics_metrics needs either interaction or kinematics GNN to be enabled.")

    spatial_size = module_cfg.get('spatial_size', 768)
    coords_col = module_cfg.get('coords_col', (1, 4))

    # Loop over events
    particle_graph_idx = counter
    if particle_graph_idx >= len(particle_graph) or np.unique(particle_graph[particle_graph_idx][:, 0])[0] != data_idx:
        print("No particle graph")
        return (), ()

    pred_particles = kinematics_particles[data_idx] if kinematics_particles is not None else inter_particles[data_idx]
    if len(pred_particles) < 1:
        return (), ()
    node_pred_type = node_pred_type[data_idx] if node_pred_type is not None else None
    node_pred_p = node_pred_p[data_idx] if node_pred_p is not None else None
    edge_pred = flow_edge_pred[data_idx].argmax(axis=1) if flow_edge_pred is not None else None # shape (E,)
    edge_index = kinematics_edge_index[data_idx] if kinematics_edge_index is not None else None# shape (E, 2)

    node_true_type = get_cluster_label(kinematics[data_idx], pred_particles, column=7) # pdg label
    node_true_p = get_momenta_label(kinematics[data_idx], pred_particles, column=8).reshape((-1,)) # momentum label
    node_true_cluster_id = get_cluster_label(kinematics[data_idx], pred_particles, column=6) # cluster id

    clust_ids = get_cluster_label(clust_data[data_idx], pred_particles, 5) # or 6 ?
    subgraph = particle_graph[particle_graph_idx][:, 1:3]
    true_edge_index = get_fragment_edges(subgraph, clust_ids)
    edge_assn = edge_assignment_from_graph(edge_index, true_edge_index) if edge_index is not None else None # shape (E,), values 0 or 1

    if edge_assn is not None:
        edge_accuracy = (edge_pred == edge_assn).sum()
        existing_edge_accuracy = (edge_pred == edge_assn)[edge_assn == 1].sum()
        nonexisting_edge_accuracy = (edge_pred == edge_assn)[edge_assn == 0].sum()
        num_existing_edges = np.count_nonzero(edge_assn == 1)
        num_nonexisting_edges = np.count_nonzero(edge_assn == 0)

    # Loop over particles
    row_names, row_values = [], []
    for i in range(len(node_true_type)):
        true_cluster_id = node_true_cluster_id[i]
        p = particles[data_idx][true_cluster_id]
        pred_voxels = kinematics[data_idx][pred_particles[i]]
        true_voxels = kinematics[data_idx][kinematics[data_idx][:, 6] == true_cluster_id]

        true_d = extent(true_voxels)
        pred_d = extent(pred_voxels)
        boundaries = np.min(np.concatenate([true_voxels[:, coords_col[0]:coords_col[1]], spatial_size - true_voxels[:, coords_col[0]:coords_col[1]]], axis=1))

        tuple_names = ('true_cluster_id', 'true_type', 'pred_type', 'distance_to_boundary',
                        'pred_num_voxels', 'pred_sum_voxels',
                        'true_num_voxels', 'true_sum_voxels', 'pdg', 'energy_deposit', 'energy_init',
                        'px', 'py', 'pz', 'true_spatial_extent', 'true_spatial_std', 'pred_spatial_extent', 'pred_spatial_std',)
        tuple_values = (true_cluster_id, node_true_type[i], node_pred_type[i].argmax(), boundaries,
                        len(pred_voxels), pred_voxels[:, 4].sum(),
                        len(true_voxels), true_voxels[:, 4].sum(), p.pdg_code(), p.energy_deposit(), p.energy_init(),
                        p.px(), p.py(), p.pz(), true_d.max(), true_d.std(), pred_d.max(), pred_d.std(),)

        if kinematics_particles is not None:
            # Children
            children_index = edge_index[:, 0] == i
            true_num_children = np.count_nonzero(edge_assn[children_index])
            pred_num_children = np.count_nonzero(edge_pred[children_index])
            overlap_num_children = np.count_nonzero(edge_assn[children_index] & (edge_assn[children_index] == edge_pred[children_index]))

            parent_index = edge_index[:, 1] == i
            true_num_parents = np.count_nonzero(edge_assn[parent_index])
            pred_num_parents = np.count_nonzero(edge_pred[parent_index])
            overlap_num_parents = np.count_nonzero(edge_assn[parent_index] & (edge_assn[parent_index] == edge_pred[parent_index]))

            tuple_values += (
                        node_true_p[i], node_pred_p[i][0],
                         true_num_children, pred_num_children, overlap_num_children,
                        true_num_parents, pred_num_parents, overlap_num_parents,
                        edge_accuracy, existing_edge_accuracy, nonexisting_edge_accuracy, num_existing_edges, num_nonexisting_edges)

            tuple_names += (
                        'true_p', 'pred_p',
                         'true_num_children', 'pred_num_children', 'overlap_num_children',
                        'true_num_parents', 'pred_num_parents', 'overlap_num_parents',
                        'edge_accuracy', 'existing_edge_accuracy', 'nonexisting_edge_accuracy', 'num_existing_edges', 'num_nonexisting_edges')

        row_names.append(tuple_names)
        row_values.append(tuple_values)

    return row_names, row_values
