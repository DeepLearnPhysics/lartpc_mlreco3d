# GNN clustering prediction
import numpy as np

from mlreco.post_processing import post_processing
from mlreco.utils.gnn.evaluation import (node_purity_mask,
                                         edge_purity_mask,
                                         edge_assignment_score,
                                         edge_assignment,
                                         node_assignment,
                                         node_assignment_score,
                                         node_assignment_bipartite,
                                         clustering_metrics,
                                         primary_assignment)
from mlreco.utils.gnn.cluster import form_clusters
from mlreco.post_processing.common import extent


@post_processing('cluster-gnn-metrics',
                 ['clust_data', 'particles'],
                 ['edge_pred', 'clusts', 'node_pred', 'edge_index'])
def cluster_gnn_metrics(cfg, module_cfg, data_blob, res, logdir, iteration,
                        edge_pred=None, clusts=None, node_pred=None, edge_index=None,
                        clust_data=None, particles=None, data_idx=None, clust_data_noghost=None,
                        **kwargs):
    """
    Compute metrics for GRAPPA stage (GNN clustering).

    `enable_physics_metrics` = compute detailed cluster-wise metrics
    `integrated_metrics` = compute voxel-wise ARI/Purity/Efficiency metrics
    (only if `enable_physics_metrics: False`)

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
    # If there is no prediction, proceed
    edge_pred_label = module_cfg.get('edge_pred', 'edge_pred')

    # Get the post processor parameters
    coords_col = module_cfg.get('coords_col', (1, 4))
    column = module_cfg.get('target_col', 6)
    column_source = module_cfg.get('source_col', 5)
    column_primary = module_cfg.get('primary_col', 10)
    chain = module_cfg.get('chain', 'chain')

    enable_physics_metrics = module_cfg.get('enable_physics_metrics', False)
    spatial_size = module_cfg.get('spatial_size', 768)

    #if not edge_pred_label in res: continue
    bipartite = cfg['model']['modules'][chain]['base'].get('network', 'complete') == 'bipartite'
    group_pred_alg = cfg['model']['modules'][chain + '_loss'].get('node_loss', {}).get('group_pred_alg', 'score')
    high_purity = cfg['model']['modules'][chain + '_loss'].get('edge_loss', {}).get('high_purity', False)

    node_predictions = node_pred
    original_clust_data = clust_data_noghost if clust_data_noghost is not None else clust_data

    if not len(clusts) or not len(clust_data):
        return (), ()

    # If there is no node, append default
    if not len(clusts[data_idx]) or not len(clust_data[data_idx]):
        # fout.record(['ite', 'idx', 'ari', 'ami', 'sbd', 'pur', 'eff', 'num_clusts', 'num_pix'],
        #             [iteration, tree_idx, -1, -1, -1, -1, -1, -1, -1])
        return (), ()

    # Use group id to make node labels
    group_ids, cluster_ids, primary_ids = [], [], []
    for c in clusts[data_idx]:
        v, cts = np.unique(clust_data[data_idx][c,column], return_counts=True)
        group_ids.append(int(v[cts.argmax()]))
        v, cts = np.unique(clust_data[data_idx][c,column_source], return_counts=True)
        cluster_ids.append(int(v[cts.argmax()]))
        v, cts = np.unique(clust_data[data_idx][c,column_primary], return_counts=True)
        primary_ids.append(int(v[cts.argmax()]))
    group_ids = np.array(group_ids, dtype=np.int64)
    cluster_ids = np.array(cluster_ids, dtype=np.int64)
    primary_ids = np.array(primary_ids, dtype=np.int64)

    #print('clusts', [len(c) for c in clusts[data_idx]], len(group_ids), edge_index[data_idx].shape)
    #edge_assn = edge_assignment(edge_index[data_idx], group_ids)
    #purity_mask = edge_purity_mask(edge_index[data_idx], cluster_ids, group_ids)
    #edge_predictions, _, _ = edge_assignment_score(edge_index[data_idx], edge_pred[data_idx], n)
    #print('metrics' , edge_pred[data_idx].shape, edge_assn.shape, edge_pred[data_idx][:10], np.unique(edge_assn, return_counts=True))
    #print(np.sum(edge_assn == np.argmax(edge_pred[data_idx], axis=1))/len(edge_assn))
    #print(np.sum(edge_assn[purity_mask] == np.argmax(edge_pred[data_idx][purity_mask], axis=1))/len(edge_assn[purity_mask]))

    # Assign predicted group ids
    n = len(clusts[data_idx])
    num_pix = np.sum([len(c) for c in clusts[data_idx]])
    if not bipartite:
        # Determine the predicted group IDs by using union find
        edge_assn = np.argmax(edge_pred[data_idx], axis=1)
        if group_pred_alg == 'threshold':
            node_pred = node_assignment(edge_index[data_idx], edge_assn, n)
        elif group_pred_alg == 'score':
            node_pred = node_assignment_score(edge_index[data_idx], edge_pred[data_idx], n)
        else:
            raise ValueError('Group prediction algorithm not recognized: ' + group_pred_alg)
    else:
        # Determine the predicted group by chosing the most likely primary for each secondary
        primary_ids = np.unique(edge_index[data_idx][:,0])
        node_pred = node_assignment_bipartite(edge_index[data_idx], edge_pred[data_idx][:,1], primary_ids, n)
    node_pred = np.array(node_pred, dtype=np.int64)

    # primary prediction
    node_pred_primary = None
    if node_predictions is not None:
        node_pred_primary = primary_assignment(node_predictions[data_idx], group_ids=node_pred)
        node_true_primary = np.equal(cluster_ids, group_ids)

    if enable_physics_metrics:
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

            # Primary identification
            pred_primaries_accuracy = -1
            if node_pred_primary is not None:
                pred_primaries = node_true_primary[node_pred == pred_id] & node_pred_primary[node_pred == pred_id]
                pred_primaries_accuracy = pred_primaries.sum()

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

            boundaries = np.min(np.concatenate([true_voxels[:, coords_col[0]:coords_col[1]], spatial_size - true_voxels[:, coords_col[0]:coords_col[1]]], axis=1))
            true_fragments_count = len(true_cluster)
            pred_fragments_count = len(pred_cluster)
            overlap_fragments_count = len(overlap_cluster)
            original_fragments_count = len(original_cluster)

            row_names = ('true_id', 'pred_id',
                        'true_voxel_count', 'pred_voxel_count', 'overlap_voxel_count', 'original_voxel_count',
                        'purity', 'efficiency', 'true_voxels_sum', 'pred_voxels_sum', 'original_voxels_sum',
                        'true_fragments_count', 'pred_fragments_count', 'overlap_fragments_count', 'original_fragments_count',
                        'true_spatial_extent', 'true_spatial_std', 'distance_to_boundary',
                        'pred_spatial_extent', 'pred_spatial_std', 'particle_count',
                        'original_spatial_extent', 'original_spatial_std',
                        'true_energy_deposit', 'true_energy_init', 'true_pdg',
                        'true_px', 'true_py', 'true_pz', 'nu_idx', 'pred_primaries_accuracy')
            row_values = (true_id, pred_id,
                        true_voxel_count, pred_voxel_count, overlap_voxel_count, original_voxel_count,
                        purity, efficiency, true_voxels[:, -1].sum(), pred_voxels[:, -1].sum(), original_voxels[:, -1].sum(),
                        true_fragments_count, pred_fragments_count, overlap_fragments_count, original_fragments_count,
                        true_d.max(), true_d.std(), boundaries,
                        pred_d.max(), pred_d.std(), len(true_particles_idx),
                        original_d.max(), original_d.std(),
                        energy_deposit, energy_init, pdg[0],
                        np.sum(px), np.sum(py), np.sum(pz), nu_id[0], pred_primaries_accuracy)

    else:
        integrated_metrics = module_cfg.get('integrated_metrics', False)
        # Evaluate clustering metrics pixel-wise
        if integrated_metrics:
            pixel_group_ids = np.hstack([[g] * len(clusts[data_idx][c_idx]) for c_idx, g in enumerate(group_ids)])
            pixel_cluster_ids = np.hstack([[g] * len(clusts[data_idx][c_idx]) for c_idx, g in enumerate(cluster_ids)])
            pixel_clusts = np.hstack(clusts[data_idx])[:, None]
            pixel_node_pred = np.hstack([[g] * len(clusts[data_idx][c_idx]) for c_idx, g in enumerate(node_pred)])

            if high_purity:
                purity_mask = node_purity_mask(cluster_ids, group_ids, primary_ids)
                if not purity_mask.any():
                    return (), ()
                pixel_group_ids = np.hstack([[g] * len(clusts[data_idx][purity_mask][c_idx]) for c_idx, g in enumerate(group_ids[purity_mask])])
                pixel_clusts = np.hstack(clusts[data_idx][purity_mask])[:, None]
                pixel_node_pred = np.hstack([[g] * len(clusts[data_idx][purity_mask][c_idx]) for c_idx, g in enumerate(node_pred[purity_mask])])

            ari, ami, sbd, pur, eff = clustering_metrics(pixel_clusts,
                                                        pixel_group_ids,
                                                        pixel_node_pred)
        else:
            if not high_purity:
                ari, ami, sbd, pur, eff = clustering_metrics(clusts[data_idx], group_ids, node_pred)
            else:
                purity_mask = node_purity_mask(cluster_ids, group_ids, primary_ids)
                if not purity_mask.any():
                    return (), ()
                ari, ami, sbd, pur, eff = clustering_metrics(clusts[data_idx][purity_mask], group_ids[purity_mask], node_pred[purity_mask])
            #print(ari, pur, eff)
        primary_accuracy = -1.
        high_purity_value = -1
        if high_purity and node_pred_primary is not None:
            primary_accuracy = np.count_nonzero(node_pred_primary == node_true_primary) / len(node_pred_primary)
            high_purity_value = purity_mask.any()
            #print(data_idx, "primary accuracy", primary_accuracy, high_purity)
        # Store
        row_names = ('ari', 'ami', 'sbd', 'pur', 'eff',
                    'num_fragments', 'num_pix', 'num_true_clusts', 'num_pred_clusts', 'primary_accuracy', 'high_purity')
        row_values = (ari, ami, sbd, pur, eff,
                    n, num_pix, len(np.unique(group_ids)), len(np.unique(node_pred)), primary_accuracy, high_purity_value)

    return row_names, row_values
