import os
import numpy as np
from mlreco.utils import CSVData
from mlreco.utils.gnn.evaluation import edge_assignment, node_assignment, node_assignment_bipartite, clustering_metrics
from mlreco.utils.deghosting import adapt_labels_numpy as adapt_labels
from mlreco.utils.cluster.dense_cluster import gaussian_kernel, ellipsoidal_kernel, fit_predict_np, find_cluster_means
from mlreco.utils.metrics import *


def extent(voxels):
    centroid = voxels[:, :3].mean(axis=0)
    return np.linalg.norm(voxels[:, :3] - centroid, axis=1)


def cluster_cnn_metrics(cfg, data_blob, res, logdir, iteration):
    deghosting = cfg['post_processing']['cluster_cnn_metrics'].get('ghost', False)

    store_method = cfg['post_processing']['cluster_cnn_metrics']['store_method']
    store_per_event = store_method == 'per-event'
    s_thresholds = cfg['post_processing']['cluster_cnn_metrics'].get('s_threshold', [0, 0, 0, 0])
    p_thresholds = cfg['post_processing']['cluster_cnn_metrics'].get('p_thresholds', [0.5, 0.5, 0.5, 0.5])
    spatial_size = cfg['post_processing']['cluster_cnn_metrics'].get('spatial_size', 768)
    enable_physics_metrics = cfg['post_processing']['cluster_cnn_metrics'].get('enable_physics_metrics', False)

    spice_min_voxels = cfg['model']['modules']['spice']['fragment_clustering'].get('min_voxels', 2)

    if store_method == 'per-iteration':
        fout = CSVData(os.path.join(logdir, 'cluster-cnn-metrics-iter-%07d.csv' % iteration))
    if store_method == 'single-file':
        append = True if iteration else False
        fout = CSVData(os.path.join(logdir, 'cluster-cnn-metrics.csv'), append=append)

    # Get the relevant data products
    index = data_blob['index']
    seg_label = data_blob['segment_label']
    clust_data = data_blob['cluster_label']
    particles = data_blob['particles']

    seg_prediction = [res['segmentation'][i].argmax(axis=1) for i in range(len(res['segmentation']))]

    if deghosting:
        ghost_mask = [res['ghost'][i].argmax(axis=1) == 0 for i in range(len(res['ghost']))]
        true_ghost_mask = [ seg_label[i][:, -1] < 5 for i in range(len(seg_label))]
        clust_data = adapt_labels(res, seg_label, data_blob['cluster_label'])
        seg_label = [seg_label[i][ghost_mask[i]] for i in range(len(seg_label))]
        seg_prediction = [seg_prediction[i][ghost_mask[i]] for i in range(len(seg_prediction))]

    batch_ids = []
    for data_idx, _ in enumerate(index):
        batch_ids.append(np.ones((seg_label[data_idx].shape[0],)) * data_idx)
    batch_ids = np.hstack(batch_ids)

    # Loop over events
    for data_idx, tree_idx in enumerate(index):
        # Initialize log if one per event
        if store_per_event:
            fout = CSVData(os.path.join(logdir, 'cluster-cnn-metrics-event-%07d.csv' % tree_idx))

        embeddings = np.array(res['embeddings'])[batch_ids == data_idx]
        margins = np.array(res['margins'])[batch_ids == data_idx]
        seediness = np.array(res['seediness'])[batch_ids == data_idx]
        coords = seg_label[data_idx][:, :3]

        # Compute total momentum and energy per interaction
        total_momentum = {}
        for interaction_id in np.unique(clust_data[data_idx][:, 7]):
            interaction_mask = clust_data[data_idx][:, 7] == interaction_id
            total_px, total_py, total_pz, total_energy_init, total_energy_deposit = 0, 0, 0, 0, 0
            for c, cluster_id in enumerate(np.unique(clust_data[data_idx][interaction_mask, 6])):
                total_px += particles[data_idx][int(c)].px()
                total_py += particles[data_idx][int(c)].py()
                total_pz += particles[data_idx][int(c)].pz()
                total_energy_init += particles[data_idx][int(c)].energy_init()
                total_energy_deposit += particles[data_idx][int(c)].energy_deposit()
            total_momentum[interaction_id] = (total_px, total_py, total_pz, total_energy_init, total_energy_deposit)

        if enable_physics_metrics:
            # Loop over semantic classes
            for c in np.unique(seg_label[data_idx][:, -1]):
                if int(c) >= 4:
                    continue
                original_clust_data = data_blob['cluster_label'][data_idx][data_blob['cluster_label'][data_idx][:, -1] == c]
                semantic_mask = seg_prediction[data_idx] == c

                embedding_class = embeddings[semantic_mask]
                seed_class = seediness[semantic_mask]
                margins_class = margins[semantic_mask]

                if len(embedding_class) < spice_min_voxels:
                    continue
                pred = fit_predict_np(embedding_class, seed_class, margins_class, gaussian_kernel,
                                    s_threshold=s_thresholds[int(c)], p_threshold=p_thresholds[int(c)])

                #original_semantic_mask = data_blob['segment_label'][data_idx][:, -1] == c
                coords_class = coords[semantic_mask]
                clabels = clust_data[data_idx][semantic_mask][:, 6]
                original_coords_class = original_clust_data[:, :3]
                original_clabels = original_clust_data[:, 6]

                #_, true_centroids = find_cluster_means(coords_class, clabels)
                #_, original_centroids = find_cluster_means(original_coords_class, original_clabels)
                # Loop over predicted clusters
                for j, true_id in enumerate(np.unique(clabels)):
                    cluster_mask = clabels == true_id
                    pred_id = np.bincount(pred[cluster_mask]).argmax()
                    pred_mask = pred == pred_id
                    original_mask = original_clabels == true_id

                    # "Purity" + efficiency
                    overlap_pixel_count = np.count_nonzero(pred[cluster_mask] == pred_id)
                    true_pixel_count = np.count_nonzero(cluster_mask)
                    pred_pixel_count = np.count_nonzero(pred_mask)
                    original_pixel_count = np.count_nonzero(original_mask)
                    efficiency = overlap_pixel_count / true_pixel_count
                    purity = overlap_pixel_count / pred_pixel_count

                    # True particle information
                    p = particles[data_idx][int(true_id)]
                    # print(c, true_id, p.pdg_code(), p.shape(), true_pixel_count, np.unique(clust_data[data_idx][semantic_mask][cluster_mask, -1]))

                    # Voxel information
                    true_voxels = clust_data[data_idx][semantic_mask][cluster_mask, :5]
                    pred_voxels = clust_data[data_idx][semantic_mask][pred_mask, :5]
                    original_voxels = original_clust_data[original_mask, :5]

                    d = extent(coords_class[cluster_mask])
                    pred_d = extent(coords_class[pred_mask])
                    boundaries = np.min(np.concatenate([coords_class[cluster_mask], spatial_size - coords_class[cluster_mask]], axis=1))

                    if original_pixel_count:
                        original_overlap_pixel_count = len(np.intersect1d(np.where(ghost_mask[data_idx])[0][semantic_mask][pred_mask],
                                                                        np.where(true_ghost_mask[data_idx])[0][data_blob['segment_label'][data_idx][true_ghost_mask[data_idx]][:, -1] == c][original_mask]))
                        original_d = extent(original_coords_class[original_mask])
                        original_boundaries = np.min(np.concatenate([original_coords_class[original_mask], spatial_size - original_coords_class[original_mask]], axis=1))
                    else:
                        original_overlap_pixel_count = -1
                        original_d = np.array([-1])
                        original_boundaries = -1

                    fout.record(('Iteration', 'Index', 'Class', 'true_id', 'pred_id',
                                'true_pixel_count', 'pred_pixel_count', 'overlap_pixel_count',
                                'purity', 'efficiency', 'spatial_extent', 'spatial_std',
                                'pred_spatial_extent', 'pred_spatial_std',
                                'true_voxels_sum', 'pred_voxels_sum', 'distance_to_boundary',
                                'pdg', 'px', 'py', 'pz', 'energy_init', 'energy_deposit',
                                'original_pixel_count', 'original_voxels_sum',
                                'original_spatial_extent', 'original_spatial_std', 'original_distance_to_boundary'),
                                (iteration, tree_idx, c, true_id, pred_id,
                                true_pixel_count, pred_pixel_count, overlap_pixel_count,
                                purity, efficiency, d.max(), d.std(),
                                pred_d.max(), pred_d.std(),
                                true_voxels[:, -1].sum(), pred_voxels[:, -1].sum(), boundaries,
                                p.pdg_code(), p.px(), p.py(), p.pz(), p.energy_init(), p.energy_deposit(),
                                original_pixel_count, original_voxels[:, -1].sum(),
                                original_d.max(), original_d.std(), original_boundaries))
                    fout.write()
        else:
            # Loop over semantic classes
            for c in np.unique(seg_label[data_idx][:, -1]):
                if int(c) >= 4:
                    continue
                semantic_mask = seg_label[data_idx][:, -1] == c

                embedding_class = embeddings[semantic_mask]
                seed_class = seediness[semantic_mask]
                margins_class = margins[semantic_mask]
                coords_class = coords[semantic_mask]
                clabels = clust_data[data_idx][semantic_mask][:, 6]

                if len(embedding_class) < spice_min_voxels:
                    continue
                pred = fit_predict_np(embedding_class, seed_class, margins_class, gaussian_kernel,
                                    s_threshold=s_thresholds[int(c)], p_threshold=p_thresholds[int(c)])

                purity, efficiency = purity_efficiency(pred, clabels)
                # purity = purity.mean()
                # efficiency = efficiency.mean()
                fscore = 2 * (purity * efficiency) / (purity + efficiency)
                ari = ARI(pred, clabels)
                sbd = SBD(pred, clabels)
                nclusters = len(np.unique(clabels))
                #num_particles = len(particles[data_idx])
                event_num_particles = len(np.unique(clust_data[data_idx][:, 6]))
                class_num_particles = len(np.unique(clust_data[data_idx][semantic_mask][:, 6]))
                event_num_pix = seg_label[data_idx].shape[0]
                class_num_pix = seg_label[data_idx][semantic_mask].shape[0]
                event_num_interactions = len(np.unique(clust_data[data_idx][:, 7]))
                _, true_centroids = find_cluster_means(coords_class, clabels)
                true_num_clusters = len(np.unique(clabels))
                for j, cluster_id in enumerate(np.unique(clabels)):
                    margin = np.mean(margins_class[clabels == cluster_id])
                    true_size = np.std(np.linalg.norm(coords_class[clabels == cluster_id] - true_centroids[j], axis=1))
                    cluster_num_pix = (clabels == cluster_id).sum()
                    #interaction_id = np.unique(clust_data[data_idx][clust_data[data_idx][:, 5] == cluster_id, 7])[0]

                    row = (iteration, tree_idx, c, ari, sbd, purity, efficiency, fscore, \
                        nclusters, true_num_clusters, margin, true_size, event_num_particles, \
                        cluster_num_pix, event_num_pix, class_num_particles, class_num_pix, \
                        event_num_interactions, particles[data_idx][j].energy_deposit(),
                        particles[data_idx][j].energy_init(),
                        particles[data_idx][j].pdg_code(),
                        particles[data_idx][j].nu_interaction_type(),
                        particles[data_idx][j].px(),
                        particles[data_idx][j].py(),
                        particles[data_idx][j].pz(),
                        # total_momentum[interaction_id][0],
                        # total_momentum[interaction_id][1],
                        # total_momentum[interaction_id][2],
                        # total_momentum[interaction_id][3],
                        # total_momentum[interaction_id][4],
                        cluster_id,
                        # interaction_id
                        )
                    #output.append(row)
                    fout.record(('Iteration', 'Index', 'Class', 'ARI', 'SBD',
                                'Purity', 'Efficiency', 'FScore', 'num_clusters', 'true_num_clusters',
                                'margin', 'true_size', 'event_num_particles', 'cluster_num_pix',
                                'event_num_pix', 'class_num_particles', 'class_num_pix',
                                'event_num_interactions', 'particle_energy_deposit',
                                'particle_energy_init', 'particle_pdg_code', 'particle_interaction_type',
                                'particle_px', 'particle_py', 'particle_pz', #'interaction_px',
                                #'interaction_py', 'interaction_pz', 'interaction_energy_id',
                                #'interaction_energy_deposit',
                                'cluster_id'), row)
                    fout.write()

        #fout.record(('ite', 'idx', 'ari', 'pur', 'eff'), (iteration, tree_idx, -1, -1, -1))

        if store_per_event:
            fout.close()

    if not store_per_event:
        fout.close()
