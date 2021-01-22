import os
import numpy as np
from mlreco.utils import CSVData
from mlreco.utils.gnn.evaluation import edge_assignment, node_assignment, node_assignment_bipartite, clustering_metrics, edge_assignment_from_graph
from mlreco.utils.deghosting import adapt_labels_numpy as adapt_labels
from mlreco.utils.dense_cluster import gaussian_kernel, ellipsoidal_kernel, fit_predict_np, find_cluster_means
from mlreco.utils.metrics import *
from mlreco.utils.gnn.network import get_fragment_edges
from mlreco.utils.gnn.cluster import get_cluster_label_np


def extent(voxels):
    centroid = voxels[:, :3].mean(axis=0)
    return np.linalg.norm(voxels[:, :3] - centroid, axis=1)


def kinematics_metrics(cfg, data_blob, res, logdir, iteration):
    deghosting = cfg['post_processing']['kinematics_metrics'].get('ghost', False)

    store_method = cfg['post_processing']['kinematics_metrics']['store_method']
    store_per_event = store_method == 'per-event'
    spatial_size = cfg['post_processing']['kinematics_metrics'].get('spatial_size', False)

    if store_method == 'per-iteration':
        fout = CSVData(os.path.join(logdir, 'kinematics-metrics-iter-%07d.csv' % iteration))
    if store_method == 'single-file':
        append = True if iteration else False
        fout = CSVData(os.path.join(logdir, 'kinematics-metrics.csv'), append=append)

    # Get the relevant data products
    index = data_blob['index']
    seg_label = data_blob['segment_label']
    clust_data = data_blob['cluster_label']
    particles = data_blob['particles']
    kinematics = data_blob['kinematics_label']
    particle_graph = data_blob['particle_graph']

    if deghosting:
        clust_data = adapt_labels(res, seg_label, data_blob['cluster_label'])
        kinematics = adapt_labels(res, seg_label, kinematics)
        seg_label = [seg_label[i][res['ghost'][i].argmax(axis=1) == 0] for i in range(len(seg_label))]

    batch_ids = []
    for data_idx, _ in enumerate(index):
        batch_ids.append(np.ones((seg_label[data_idx].shape[0],)) * data_idx)
    batch_ids = np.hstack(batch_ids)

    # Loop over events
    particle_graph_idx = 0
    for data_idx, tree_idx in enumerate(index):
        # Initialize log if one per event
        if store_per_event:
            fout = CSVData(os.path.join(logdir, 'kinematics-metrics-event-%07d.csv' % tree_idx))

        if particle_graph_idx >= len(particle_graph) or np.unique(particle_graph[particle_graph_idx][:, -1])[0] != data_idx:
            print(iteration, tree_idx, "No particle graph")
            continue

        pred_particles = res['kinematics_particles'][data_idx]
        node_pred_type = res['node_pred_type'][data_idx]
        node_pred_p = res['node_pred_p'][data_idx]
        edge_pred = res['flow_edge_pred'][data_idx].argmax(axis=1) # shape (E,)
        edge_index = res['kinematics_edge_index'][data_idx] # shape (E, 2)

        node_true_type = get_cluster_label_np(kinematics[data_idx], pred_particles, column=7) # pdg label
        node_true_p = get_cluster_label_np(kinematics[data_idx], pred_particles, column=8) # momentum label
        node_true_cluster_id = get_cluster_label_np(kinematics[data_idx], pred_particles, column=6) # cluster id

        clust_ids = get_cluster_label_np(clust_data[data_idx], pred_particles, 5) # or 6 ?
        subgraph = particle_graph[particle_graph_idx][:, :2]
        true_edge_index = get_fragment_edges(subgraph, clust_ids)
        edge_assn = edge_assignment_from_graph(edge_index, true_edge_index) # shape (E,), values 0 or 1

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

            row = (iteration, tree_idx, true_cluster_id, node_true_type[i], node_pred_type[i].argmax(),
            node_true_p[i], node_pred_p[i][0], len(pred_voxels), pred_voxels[:, 4].sum(),
            len(true_voxels), true_voxels[:, 4].sum(), p.pdg_code(), p.energy_deposit(), p.energy_init(),
            p.px(), p.py(), p.pz(), true_d.max(), true_d.std(), pred_d.max(), pred_d.std(),
            boundaries, true_num_children, pred_num_children, overlap_num_children,
            true_num_parents, pred_num_parents, overlap_num_parents)

            fout.record(('iter', 'idx', 'true_cluster_id', 'true_type', 'pred_type',
                        'true_p', 'pred_p', 'pred_num_voxels', 'pred_sum_voxels',
                        'true_num_voxels', 'true_sum_voxels', 'pdg', 'energy_deposit', 'energy_init',
                        'px', 'py', 'pz', 'true_spatial_extent', 'true_spatial_std', 'pred_spatial_extent', 'pred_spatial_std',
                        'distance_to_boundary', 'true_num_children', 'pred_num_children', 'overlap_num_children',
                        'true_num_parents', 'pred_num_parents', 'overlap_num_parents'),
                        row)
            fout.write()

        if store_per_event:
            fout.close()

        particle_graph_idx += 1

    if not store_per_event:
        fout.close()
