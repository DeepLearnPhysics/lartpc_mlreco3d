# Nu vs cosmic discrimination prediction
import numpy as np
from scipy.special import softmax

from mlreco.post_processing import post_processing
from mlreco.utils.gnn.evaluation import edge_assignment, node_assignment, node_assignment_bipartite, clustering_metrics
from mlreco.utils.gnn.cluster import get_cluster_label
from mlreco.post_processing.common import extent


@post_processing('cosmic-discriminator-metrics', ['clust_data', 'particles'], ['interactions', 'inter_cosmic_pred'])
def cosmic_discriminator_metrics(cfg, module_cfg, data_blob, res, logdir, iteration,
                                data_idx=None, clust_data=None, particles=None,
                                interactions=None, inter_cosmic_pred=None, **kwargs):
    N = module_cfg.get('spatial_size', 768)
    enable_physics_metrics = module_cfg.get('enable_physics_metrics', False)
    spatial_size = module_cfg.get('spatial_size', 768)
    coords_col = module_cfg.get('coords_col', (1, 4))

    nu_label = get_cluster_label(clust_data[data_idx], interactions[data_idx], column=8)
    nu_label = (nu_label > -1).astype(int)

    n_interactions = nu_label.shape[0]
    n_nu = (nu_label == 1).sum()
    n_cosmic = (nu_label == 0).sum()

    nu_score = softmax(inter_cosmic_pred[data_idx], axis=1)
    nu_pred = np.argmax(nu_score, axis=1)

    n_pred_nu = (nu_pred == 1).sum()
    n_pred_cosmic = (nu_pred == 0).sum()

    if enable_physics_metrics:
        true_interaction_idx = get_cluster_label(clust_data[data_idx], interactions[data_idx], column=7)
        for i, j in enumerate(true_interaction_idx):
            true_interaction = clust_data[data_idx][clust_data[data_idx][:, 7] == j]
            pred_interaction = clust_data[data_idx][interactions[data_idx][i]]
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
            boundaries = np.min(np.concatenate([true_interaction[:, coords_col[0]:coords_col[1]], spatial_size - true_interaction[:, coords_col[0]:coords_col[1]]], axis=1))

            row_names = ('interaction_id', 'label', 'prediction', 'softmax_score',
                        'true_voxel_count', 'pred_voxel_count', 'energy_init', 'energy_deposit',
                        'px', 'py', 'pz', 'true_voxel_sum', 'pred_voxel_sum',
                        'true_particle_count', 'distance_to_boundary',
                        'true_spatial_extent', 'true_spatial_std', 'pred_spatial_extent', 'pred_spatial_std')
            row_values = (j, nu_label[i], nu_pred[i], nu_score[i, 1],
                        len(true_interaction), len(pred_interaction), energy_init, energy_deposit,
                        np.sum(px), np.sum(py), np.sum(pz), true_interaction[:, 4].sum(), pred_interaction[:, 4].sum(),
                        len(true_particles_idx), boundaries,
                        true_d.max(), true_d.std(), pred_d.max(), pred_d.std())
    else:
        acc = (nu_pred == nu_label).sum() / n_interactions
        acc_nu = (nu_pred == nu_label)[nu_label == 1].sum() / n_nu
        acc_cosmic =  (nu_pred == nu_label)[nu_label == 0].sum() / n_cosmic

        # Distance to boundaries
        c = np.hstack(interactions[data_idx])
        x = clust_data[data_idx][c]
        distances = np.stack([x[:, 0], N - x[:, 0], x[:, 1], N-x[:,1], x[:, 2], N - x[:, 2]], axis=1)
        d = np.amin(distances, axis=0)

        row_names = ('n_interactions', 'n_nu', 'n_cosmic', 'n_pred_nu', 'n_pred_cosmic',
                    'acc', 'acc_nu', 'acc_cosmic',
                    'd_x_low', 'd_x_high', 'd_y_low', 'd_y_high', 'd_z_low', 'd_z_high')
        row_values = (n_interactions, n_nu, n_cosmic, n_pred_nu, n_pred_cosmic,
                    acc, acc_nu, acc_cosmic,
                    d[0], d[1], d[2], d[3], d[4], d[5])

    return row_names, row_values
