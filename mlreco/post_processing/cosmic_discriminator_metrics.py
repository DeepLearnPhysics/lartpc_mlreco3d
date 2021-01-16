# Nu vs cosmic discrimination prediction
import os
import numpy as np
from mlreco.utils import CSVData
from mlreco.utils.gnn.evaluation import edge_assignment, node_assignment, node_assignment_bipartite, clustering_metrics
from mlreco.utils.deghosting import adapt_labels_numpy as adapt_labels
from mlreco.utils.gnn.cluster import get_cluster_label_np
from scipy.special import softmax


def cosmic_discriminator_metrics(cfg, data_blob, res, logdir, iteration):
    deghosting = cfg['post_processing']['cosmic_discriminator_metrics'].get('ghost', False)
    N = cfg['post_processing']['cosmic_discriminator_metrics'].get('spatial_size', 768)

    store_method = cfg['post_processing']['cosmic_discriminator_metrics']['store_method']
    store_per_event = store_method == 'per-event'

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
        nu_label = get_cluster_label_np(clust_data[data_idx], res['interactions'][data_idx], column=8)
        nu_label = (nu_label > -1).astype(int)

        n_interactions = nu_label.shape[0]
        n_nu = (nu_label == 1).sum()
        n_cosmic = (nu_label == 0).sum()

        nu_score = softmax(res['inter_cosmic_pred'][data_idx], axis=1)
        nu_pred = np.argmax(nu_score, axis=1)

        n_pred_nu = (nu_pred == 1).sum()
        n_pred_cosmic = (nu_pred == 0).sum()

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
