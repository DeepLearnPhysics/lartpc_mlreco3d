# Nu vs cosmic discrimination prediction
import os
import numpy as np
from mlreco.utils import CSVData
from mlreco.utils.gnn.evaluation import edge_assignment, node_assignment, node_assignment_bipartite, clustering_metrics
from mlreco.utils.deghosting import adapt_labels_numpy as adapt_labels
from mlreco.utils.gnn.cluster import get_cluster_label
from scipy.special import softmax


def extent(voxels):
    centroid = voxels[:, :3].mean(axis=0)
    return np.linalg.norm(voxels[:, :3] - centroid, axis=1)


def cosmic_discriminator_metrics(cfg, data_blob, res, logdir, iteration):
    deghosting = cfg['post_processing']['cosmic_discriminator_metrics'].get('ghost', False)
    N = cfg['post_processing']['cosmic_discriminator_metrics'].get('spatial_size', 768)

    store_method = cfg['post_processing']['cosmic_discriminator_metrics']['store_method']
    store_per_event = store_method == 'per-event'
    enable_physics_metrics = cfg['post_processing']['cosmic_discriminator_metrics'].get('enable_physics_metrics', False)
    spatial_size = cfg['post_processing']['cosmic_discriminator_metrics'].get('spatial_size', 768)

    if store_method == 'per-iteration':
        fout = CSVData(os.path.join(logdir, 'cosmic-discriminator-metrics-iter-%07d.csv' % iteration))
    if store_method == 'single-file':
        append = True if iteration else False
        fout = CSVData(os.path.join(logdir, 'cosmic-discriminator-metrics.csv'), append=append)

    # Get the relevant data products
    index = data_blob['index']
    seg_label = data_blob['segment_label']
    clust_data = data_blob['cluster_label']
    particles = data_blob['particles']

    if deghosting:
        clust_data = adapt_labels(res, seg_label, data_blob['cluster_label'])

    # Loop over events
    for data_idx, tree_idx in enumerate(index):
        # Initialize log if one per event
        if store_per_event:
            fout = CSVData(os.path.join(logdir, 'cosmic-discriminator-metrics-event-%07d.csv' % tree_idx))
        nu_label = get_cluster_label(clust_data[data_idx], res['interactions'][data_idx], column=8)
        nu_label = (nu_label > -1).astype(int)

        n_interactions = nu_label.shape[0]
        n_nu = (nu_label == 1).sum()
        n_cosmic = (nu_label == 0).sum()

        nu_score = softmax(res['inter_cosmic_pred'][data_idx], axis=1)
        nu_pred = np.argmax(nu_score, axis=1)

        n_pred_nu = (nu_pred == 1).sum()
        n_pred_cosmic = (nu_pred == 0).sum()

        if enable_physics_metrics:
            true_interaction_idx = get_cluster_label(clust_data[data_idx], res['interactions'][data_idx], column=7)
            for i, j in enumerate(true_interaction_idx):
                true_interaction = clust_data[data_idx][clust_data[data_idx][:, 7] == j]
                pred_interaction = clust_data[data_idx][res['interactions'][data_idx][i]]
                true_particles_idx = np.unique(true_interaction[:, 6])
                true_particles_idx = true_particles_idx[true_particles_idx>-1]
                energy_init, energy_deposit = 0., 0.
                px, py, pz = [], [], []
                for k in true_particles_idx:
                    p = particles[data_idx][int(k)]
                    energy_init += p.energy_init()
                    energy_deposit += p.energy_deposit()
                    px.append(p.px())
                    py.append(p.py())
                    pz.append(p.pz())

                true_d = extent(true_interaction)
                pred_d = extent(pred_interaction)
                boundaries = np.min(np.concatenate([true_interaction[:, :3], spatial_size - true_interaction[:, :3]], axis=1))

                fout.record(('iter', 'idx', 'interaction_id', 'label', 'prediction', 'softmax_score',
                            'true_voxel_count', 'pred_voxel_count', 'energy_init', 'energy_deposit',
                            'px', 'py', 'pz', 'true_voxel_sum', 'pred_voxel_sum',
                            'true_particle_count', 'distance_to_boundary',
                            'true_spatial_extent', 'true_spatial_std', 'pred_spatial_extent', 'pred_spatial_std'),
                            (iteration, tree_idx, j, nu_label[i], nu_pred[i], nu_score[i, 1],
                            len(true_interaction), len(pred_interaction), energy_init, energy_deposit,
                            np.sum(px), np.sum(py), np.sum(pz), true_interaction[:, 4].sum(), pred_interaction[:, 4].sum(),
                            len(true_particles_idx), boundaries,
                            true_d.max(), true_d.std(), pred_d.max(), pred_d.std()))
        else:
            acc = (nu_pred == nu_label).sum() / n_interactions
            acc_nu = (nu_pred == nu_label)[nu_label == 1].sum() / n_nu
            acc_cosmic =  (nu_pred == nu_label)[nu_label == 0].sum() / n_cosmic

            # Distance to boundaries
            c = np.hstack(res['interactions'][data_idx])
            x = clust_data[data_idx][c]
            distances = np.stack([x[:, 0], N - x[:, 0], x[:, 1], N-x[:,1], x[:, 2], N - x[:, 2]], axis=1)
            d = np.amin(distances, axis=0)

            fout.record(('iter', 'idx', 'n_interactions', 'n_nu', 'n_cosmic', 'n_pred_nu', 'n_pred_cosmic',
                        'acc', 'acc_nu', 'acc_cosmic',
                        'd_x_low', 'd_x_high', 'd_y_low', 'd_y_high', 'd_z_low', 'd_z_high'),
                        (iteration, tree_idx, n_interactions, n_nu, n_cosmic, n_pred_nu, n_pred_cosmic,
                        acc, acc_nu, acc_cosmic,
                        d[0], d[1], d[2], d[3], d[4], d[5]))
        fout.write()

        if store_per_event:
            fout.close()

    if not store_per_event:
        fout.close()
