import numpy as np

from mlreco.post_processing import post_processing
from mlreco.utils.gnn.evaluation import edge_assignment, node_assignment, node_assignment_bipartite, clustering_metrics, edge_assignment_from_graph
from mlreco.utils.dense_cluster import gaussian_kernel, ellipsoidal_kernel, fit_predict_np, find_cluster_means
from mlreco.utils.metrics import *
from mlreco.utils.gnn.network import get_fragment_edges
from mlreco.utils.gnn.cluster import get_cluster_label_np, get_momenta_label_np


def extent(voxels):
    centroid = voxels[:, :3].mean(axis=0)
    return np.linalg.norm(voxels[:, :3] - centroid, axis=1)


@post_processing('kinematics-metrics',
                ['seg_label', 'clust_data', 'particles', 'kinematics', 'particle_graph'],
                ['kinematics_particles', 'node_pred_type', 'node_pred_p', 'flow_edge_pred', 'kinematics_edge_index'])
def kinematics_metrics(cfg, module_cfg, data_blob, res, logdir, iteration,
                        seg_label=None, clust_data=None, particles=None, kinematics=None,
                        particle_graph=None, kinematics_particles=None, node_pred_type=None,
                        node_pred_p=None, flow_edge_prd=None, kinematics_edge_index=None, counter=None, **kwargs):
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
    spatial_size = module_cfg.get('spatial_size', 768)

    # Loop over events
    particle_graph_idx = counter
    if particle_graph_idx >= len(particle_graph) or np.unique(particle_graph[particle_graph_idx][:, -1])[0] != data_idx:
        print(iteration, tree_idx, "No particle graph")
        return (), ()

    pred_particles = kinematics_particles[data_idx]
    node_pred_type = node_pred_type[data_idx]
    node_pred_p = node_pred_p[data_idx]
    edge_pred = flow_edge_pred[data_idx].argmax(axis=1) # shape (E,)
    edge_index = kinematics_edge_index[data_idx] # shape (E, 2)

    node_true_type = get_cluster_label_np(kinematics[data_idx], pred_particles, column=7) # pdg label
    node_true_p = get_momenta_label_np(kinematics[data_idx], pred_particles, column=8).reshape((-1,)) # momentum label
    node_true_cluster_id = get_cluster_label_np(kinematics[data_idx], pred_particles, column=6) # cluster id

    clust_ids = get_cluster_label_np(clust_data[data_idx], pred_particles, 5) # or 6 ?
    subgraph = particle_graph[particle_graph_idx][:, :2]
    true_edge_index = get_fragment_edges(subgraph, clust_ids)
    edge_assn = edge_assignment_from_graph(edge_index, true_edge_index) # shape (E,), values 0 or 1

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
        boundaries = np.min(np.concatenate([true_voxels[:, :3], spatial_size - true_voxels[:, :3]], axis=1))

        # Children
        children_index = edge_index[:, 0] == i
        true_num_children = np.count_nonzero(edge_assn[children_index])
        pred_num_children = np.count_nonzero(edge_pred[children_index])
        overlap_num_children = np.count_nonzero(edge_assn[children_index] & (edge_assn[children_index] == edge_pred[children_index]))

        parent_index = edge_index[:, 1] == i
        true_num_parents = np.count_nonzero(edge_assn[parent_index])
        pred_num_parents = np.count_nonzero(edge_pred[parent_index])
        overlap_num_parents = np.count_nonzero(edge_assn[parent_index] & (edge_assn[parent_index] == edge_pred[parent_index]))

        row_values.append((iteration, tree_idx, true_cluster_id, node_true_type[i], node_pred_type[i].argmax(),
                    node_true_p[i], node_pred_p[i][0], len(pred_voxels), pred_voxels[:, 4].sum(),
                    len(true_voxels), true_voxels[:, 4].sum(), p.pdg_code(), p.energy_deposit(), p.energy_init(),
                    p.px(), p.py(), p.pz(), true_d.max(), true_d.std(), pred_d.max(), pred_d.std(),
                    boundaries, true_num_children, pred_num_children, overlap_num_children,
                    true_num_parents, pred_num_parents, overlap_num_parents,
                    edge_accuracy, existing_edge_accuracy, nonexisting_edge_accuracy, num_existing_edges, num_nonexisting_edges))

        row_names.append(('iter', 'idx', 'true_cluster_id', 'true_type', 'pred_type',
                    'true_p', 'pred_p', 'pred_num_voxels', 'pred_sum_voxels',
                    'true_num_voxels', 'true_sum_voxels', 'pdg', 'energy_deposit', 'energy_init',
                    'px', 'py', 'pz', 'true_spatial_extent', 'true_spatial_std', 'pred_spatial_extent', 'pred_spatial_std',
                    'distance_to_boundary', 'true_num_children', 'pred_num_children', 'overlap_num_children',
                    'true_num_parents', 'pred_num_parents', 'overlap_num_parents',
                    'edge_accuracy', 'existing_edge_accuracy', 'nonexisting_edge_accuracy', 'num_existing_edges', 'num_nonexisting_edges'))

    return row_names, row_values
