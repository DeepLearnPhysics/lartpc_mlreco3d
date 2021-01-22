# GNN clustering prediction
import os
import numpy as np
from mlreco.utils import CSVData
from mlreco.utils.gnn.evaluation import edge_assignment, node_assignment, node_assignment_bipartite, clustering_metrics
from mlreco.utils.deghosting import adapt_labels_numpy as adapt_labels
from mlreco.utils.gnn.cluster import form_clusters


def extent(voxels):
    centroid = voxels[:, :3].mean(axis=0)
    return np.linalg.norm(voxels[:, :3] - centroid, axis=1)

def cluster_gnn_metrics(cfg, data_blob, res, logdir, iteration):
    deghosting = cfg['post_processing']['cluster_gnn_metrics'].get('ghost', False)

    # If there is no prediction, proceed
    cfg_edge_pred = cfg['post_processing']['cluster_gnn_metrics'].get('edge_pred', 'edge_pred')

    # Get the post processor parameters
    cfg_column = cfg['post_processing']['cluster_gnn_metrics'].get('target_col', 6)
    cfg_column_source = cfg['post_processing']['cluster_gnn_metrics'].get('source_col', 5)
    cfg_chain = cfg['post_processing']['cluster_gnn_metrics'].get('chain', 'chain')

    cfg_store_method = cfg['post_processing']['cluster_gnn_metrics']['store_method']
    cfg_enable_physics_metrics = cfg['post_processing']['cluster_gnn_metrics'].get('enable_physics_metrics', False)
    spatial_size = cfg['post_processing']['cluster_gnn_metrics'].get('spatial_size', 768)

    cfg_filename = cfg['post_processing']['cluster_gnn_metrics'].get('filename', 'cluster-gnn-metrics')
    cfg_edge_index = cfg['post_processing']['cluster_gnn_metrics'].get('edge_index', 'edge_index')
    cfg_clusts = cfg['post_processing']['cluster_gnn_metrics'].get('clusts', 'clusts')
    if isinstance(cfg_column, list):
        assert isinstance(cfg_chain, list)
        assert isinstance(cfg_store_method, list)
        assert isinstance(cfg_filename, list)
        assert isinstance(cfg_edge_pred, list)
        assert isinstance(cfg_edge_index, list)
        assert isinstance(cfg_clusts, list)
        for s in cfg_store_method:
            assert s in ['single-file', 'per-iteration', 'per-event']
    else:
        cfg_column = [cfg_column]
        cfg_column_source = [cfg_column_source]
        cfg_chain = [cfg_chain]
        cfg_store_method = [cfg_store_method]
        cfg_filename = [cfg_filename]
        cfg_edge_pred = [cfg_edge_pred]
        cfg_edge_index = [cfg_edge_index]
        cfg_clusts = [cfg_clusts]

    # Get the relevant data products
    index = data_blob['index']

    seg_label = data_blob['segment_label']
    clust_data = data_blob['cluster_label']
    particles = data_blob['particles']

    if deghosting:
        clust_data = adapt_labels(res, seg_label, data_blob['cluster_label'])
        # seg_label = [seg_label[i][res['ghost'][i].argmax(axis=1) == 0] for i in range(len(seg_label))]

    for column, column_source, chain, store_method, filename, edge_pred_label, edge_index_label, clusts_label in zip(cfg_column, cfg_column_source, cfg_chain, cfg_store_method, cfg_filename, cfg_edge_pred, cfg_edge_index, cfg_clusts):
        if not edge_pred_label in res: continue
        bipartite = cfg['model']['modules'][chain].get('network', 'complete') == 'bipartite'
        store_per_event = store_method == 'per-event'

        if store_method == 'per-iteration':
            fout = CSVData(os.path.join(logdir, '%s-iter-%07d.csv' % (filename, iteration)))
        if store_method == 'single-file':
            append = True if iteration else False
            fout = CSVData(os.path.join(logdir, '%s.csv' % filename), append=append)

        edge_pred = res[edge_pred_label]
        clusts = res[clusts_label]
        edge_index = res[edge_index_label]
        original_clust_data = data_blob['cluster_label']

        # Loop over events
        for data_idx, tree_idx in enumerate(index):
            if not len(clusts) or not len(clust_data):
                continue
            # Initialize log if one per event
            if store_per_event:
                fout = CSVData(os.path.join(logdir, '%s-event-%07d.csv' % (filename, tree_idx)))

            # If there is no node, append default
            if not len(clusts[data_idx]) or not len(clust_data[data_idx]):
                # fout.record(['ite', 'idx', 'ari', 'ami', 'sbd', 'pur', 'eff', 'num_clusts', 'num_pix'],
                #             [iteration, tree_idx, -1, -1, -1, -1, -1, -1, -1])
                continue

            # Use group id to make node labels
            group_ids = []
            for c in clusts[data_idx]:
                v, cts = np.unique(clust_data[data_idx][c,column], return_counts=True)
                group_ids.append(int(v[cts.argmax()]))

            # Assign predicted group ids
            n = len(clusts[data_idx])
            num_pix = np.sum([len(c) for c in clusts[data_idx]])
            if not bipartite:
                # Determine the predicted group IDs by using union find
                edge_assn = np.argmax(edge_pred[data_idx], axis=1)
                node_pred = node_assignment(edge_index[data_idx], edge_assn, n)
            else:
                # Determine the predicted group by chosing the most likely primary for each secondary
                primary_ids = np.unique(edge_index[data_idx][:,0])
                node_pred = node_assignment_bipartite(edge_index[data_idx], edge_pred[data_idx][:,1], primary_ids, n)

            if cfg_enable_physics_metrics:
                # Loop over true clusters
                for true_id in np.unique(group_ids):
                    true_cluster = clusts[data_idx][group_ids == true_id]
                    pred_id = np.bincount(node_pred[group_ids == true_id]).argmax()
                    pred_cluster = clusts[data_idx][node_pred == pred_id]
                    overlap_cluster = clusts[data_idx][(group_ids == true_id) & (node_pred == pred_id)]

                    original_indices = np.where(original_clust_data[data_idx][:, column] == true_id)[0]
                    original_cluster = [np.where(original_clust_data[data_idx][original_indices][:, column_source] == x)[0] for x in np.unique(original_clust_data[data_idx][original_indices][:, column_source])]
                    #original_cluster = form_clusters(original_clust_data[data_idx][original_indices], column=column_source)
                    original_cluster = [original_indices[c] for c in original_cluster]

                    # Purity + efficiency
                    true_voxel_count = np.sum([len(c) for c in true_cluster])
                    pred_voxel_count = np.sum([len(c) for c in pred_cluster])
                    original_voxel_count = np.sum([len(c) for c in original_cluster])
                    overlap_voxel_count = np.sum([len(c) for c in overlap_cluster])
                    efficiency = overlap_voxel_count / true_voxel_count
                    purity = overlap_voxel_count / pred_voxel_count

                    # True particle information
                    true_particles_idx = np.unique(clust_data[data_idx][np.hstack(true_cluster), 6])
                    # Remove -1
                    true_particles_idx = true_particles_idx[true_particles_idx >= 0]
                    energy_deposit = 0.
                    energy_init = 0.
                    pdg, px, py, pz = [], [], [], []
                    for j in true_particles_idx:
                        p = particles[data_idx][int(j)]
                        energy_deposit += p.energy_deposit()
                        energy_init += p.energy_init()
                        pdg.append(p.pdg_code())
                        px.append(p.px())
                        py.append(p.py())
                        pz.append(p.pz())

                    if len(pdg) == 0:
                        pdg = [-1]

                    # True interaction information
                    true_interaction_idx = np.unique(clust_data[data_idx][clust_data[data_idx][:, column] == true_id, 7])
                    # Remove -1
                    true_interaction_idx = true_interaction_idx[true_interaction_idx >= 0]
                    nu_id = []
                    for j in true_interaction_idx:
                        nu_idx = np.unique(clust_data[data_idx][(clust_data[data_idx][:, 7] == j) & (clust_data[data_idx][:, column] == true_id), 8])
                        nu_id.append(nu_idx[0])
                    if len(nu_id) == 0:
                        nu_id = [-2]

                    # Voxels information
                    true_voxels = clust_data[data_idx][np.hstack(true_cluster), :5]
                    pred_voxels = clust_data[data_idx][np.hstack(pred_cluster), :5]
                    original_voxels = original_clust_data[data_idx][np.hstack(original_cluster), :5]
                    true_d = extent(true_voxels)
                    pred_d = extent(pred_voxels)
                    original_d = extent(original_voxels)

                    boundaries = np.min(np.concatenate([true_voxels[:, :3], spatial_size - true_voxels[:, :3]], axis=1))
                    true_fragments_count = len(true_cluster)
                    pred_fragments_count = len(pred_cluster)
                    overlap_fragments_count = len(overlap_cluster)
                    original_fragments_count = len(original_cluster)

                    fout.record(('Iteration', 'Index', 'true_id', 'pred_id',
                                'true_voxel_count', 'pred_voxel_count', 'overlap_voxel_count', 'original_voxel_count',
                                'purity', 'efficiency', 'true_voxels_sum', 'pred_voxels_sum', 'original_voxels_sum', 
                                'true_fragments_count', 'pred_fragments_count', 'overlap_fragments_count', 'original_fragments_count',
                                'true_spatial_extent', 'true_spatial_std', 'distance_to_boundary',
                                'pred_spatial_extent', 'pred_spatial_std', 'particle_count',
                                'original_spatial_extent', 'original_spatial_std',
                                'true_energy_deposit', 'true_energy_init', 'true_pdg',
                                'true_px', 'true_py', 'true_pz', 'nu_idx'),
                                (iteration, tree_idx, true_id, pred_id,
                                true_voxel_count, pred_voxel_count, overlap_voxel_count, original_voxel_count,
                                purity, efficiency, true_voxels[:, -1].sum(), pred_voxels[:, -1].sum(), original_voxels[:, -1].sum(),
                                true_fragments_count, pred_fragments_count, overlap_fragments_count, original_fragments_count,
                                true_d.max(), true_d.std(), boundaries,
                                pred_d.max(), pred_d.std(), len(true_particles_idx),
                                original_d.max(), original_d.std(),
                                energy_deposit, energy_init, pdg[0],
                                np.sum(px), np.sum(py), np.sum(pz), nu_id[0]))

            else:
                # Evaluate clustering metrics
                ari, ami, sbd, pur, eff = clustering_metrics(clusts[data_idx], group_ids, node_pred)

                # Store
                fout.record(['ite', 'idx', 'ari', 'ami', 'sbd', 'pur', 'eff',
                            'num_fragments', 'num_pix', 'num_true_clusts', 'num_pred_clusts'],
                            [iteration, tree_idx, ari, ami, sbd, pur, eff,
                            n, num_pix, len(np.unique(group_ids)), len(np.unique(node_pred))])

            fout.write()
            if store_per_event:
                fout.close()

        if not store_per_event:
            fout.close()
