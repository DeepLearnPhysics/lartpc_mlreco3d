import os
import numpy as np
from mlreco.utils import CSVData
from mlreco.utils.gnn.evaluation import edge_assignment, node_assignment, node_assignment_bipartite, clustering_metrics
from mlreco.utils.deghosting import adapt_labels_numpy as adapt_labels
from mlreco.utils.dense_cluster import gaussian_kernel, ellipsoidal_kernel, fit_predict_np, find_cluster_means
from mlreco.utils.metrics import *


def cluster_cnn_metrics(cfg, data_blob, res, logdir, iteration):
    deghosting = cfg['post_processing']['cluster_cnn_metrics'].get('ghost', False)

    store_method = cfg['post_processing']['cluster_cnn_metrics']['store_method']
    store_per_event = store_method == 'per-event'
    s_thresholds = cfg['post_processing']['cluster_cnn_metrics'].get('s_threshold', [0, 0, 0, 0])
    p_thresholds = cfg['post_processing']['cluster_cnn_metrics'].get('p_thresholds', [0.5, 0.5, 0.5, 0.5])

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

    if deghosting:
        clust_data = adapt_labels(res, seg_label, data_blob['cluster_label'])
        seg_label = [seg_label[i][res['ghost'][i].argmax(axis=1) == 0] for i in range(len(seg_label))]

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
            for c, cluster_id in enumerate(np.unique(clust_data[data_idx][interaction_mask, 5])):
                total_px += particles[data_idx][int(c)].px()
                total_py += particles[data_idx][int(c)].py()
                total_pz += particles[data_idx][int(c)].pz()
                total_energy_init += particles[data_idx][int(c)].energy_init()
                total_energy_deposit += particles[data_idx][int(c)].energy_deposit()
            total_momentum[interaction_id] = (total_px, total_py, total_pz, total_energy_init, total_energy_deposit)

        # Loop over semantic classes
        for c in np.unique(seg_label[data_idx][:, -1]):
            if int(c) >= 4:
                continue
            semantic_mask = seg_label[data_idx][:, -1] == c
            embedding_class = embeddings[semantic_mask]
            seed_class = seediness[semantic_mask]
            margins_class = margins[semantic_mask]
            coords_class = coords[semantic_mask]
            clabels = clust_data[data_idx][semantic_mask][:, 5]

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
                interaction_id = np.unique(clust_data[data_idx][clust_data[data_idx][:, 5] == cluster_id, 7])[0]

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
                    total_momentum[interaction_id][0],
                    total_momentum[interaction_id][1],
                    total_momentum[interaction_id][2],
                    total_momentum[interaction_id][3],
                    total_momentum[interaction_id][4],
                    cluster_id,
                    interaction_id
                    )
                #output.append(row)
                fout.record(('Iteration', 'Index', 'Class', 'ARI', 'SBD',
                            'Purity', 'Efficiency', 'FScore', 'num_clusters', 'true_num_clusters',
                            'margin', 'true_size', 'event_num_particles', 'cluster_num_pix',
                            'event_num_pix', 'class_num_particles', 'class_num_pix',
                            'event_num_interactions', 'particle_energy_deposit',
                            'particle_energy_init', 'particle_pdg_code', 'particle_interaction_type',
                            'particle_px', 'particle_py', 'particle_pz', 'interaction_px',
                            'interaction_py', 'interaction_pz', 'interaction_energy_id',
                            'interaction_energy_deposit', 'cluster_id', 'interaction_id'), row)
                fout.write()

        #fout.record(('ite', 'idx', 'ari', 'pur', 'eff'), (iteration, tree_idx, -1, -1, -1))

        if store_per_event:
            fout.close()

    if not store_per_event:
        fout.close()
