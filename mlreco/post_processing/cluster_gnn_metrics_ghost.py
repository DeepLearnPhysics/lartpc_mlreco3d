# GNN clustering prediction
import os
import numpy as np
from mlreco.utils import CSVData
from mlreco.utils.gnn.evaluation import edge_assignment, node_assignment, node_assignment_bipartite, clustering_metrics
from mlreco.utils.deghosting import adapt_labels_numpy as adapt_labels


def cluster_gnn_metrics_ghost(cfg, data_blob, res, logdir, iteration):
    # If there is no prediction, proceed
    edge_pred = cfg['post_processing']['cluster_gnn_metrics_ghost'].get('edge_pred', 'edge_pred')
    if not edge_pred in res: return

    # Get the post processor parameters
    column = cfg['post_processing']['cluster_gnn_metrics_ghost'].get('column', 6)
    chain = cfg['post_processing']['cluster_gnn_metrics_ghost'].get('chain', 'chain')
    bipartite = cfg['model']['modules'][chain].get('network', 'complete') == 'bipartite'
    store_method = cfg['post_processing']['cluster_gnn_metrics_ghost']['store_method']
    assert store_method in ['single-file', 'per-iteration', 'per-event']
    store_per_event = store_method == 'per-event'
    if store_method == 'per-iteration':
        fout = CSVData(os.path.join(logdir, 'cluster-gnn-metrics-iter-%07d.csv' % iteration))
    if store_method == 'single-file':
        append = True if iteration else False
        fout = CSVData(os.path.join(logdir, 'cluster-gnn-metric.csv'), append=append)

    # Get the relevant data products
    index = data_blob['index']

    seg_label = data_blob['segment_label']
    #clust_data = data_blob['clust_label']
    clust_data = adapt_labels(res, seg_label, data_blob['clust_label'])
    seg_label = [seg_label[i][res['ghost'][i].argmax(axis=1) == 0] for i in range(len(seg_label))]
    #clust_data = [clust_data[i][res['segmentation'][i][(res['ghost'][i].argmax(axis=1) == 0)].argmax(axis=1) == 0] for i in range(len(clust_data))]

    edge_pred = res[edge_pred]
    edge_index = res[cfg['post_processing']['cluster_gnn_metrics_ghost'].get('edge_index', 'edge_index')]
    clusts = res[cfg['post_processing']['cluster_gnn_metrics_ghost'].get('clusts', 'clusts')]

    # Loop over events
    for data_idx, tree_idx in enumerate(index):
        if not len(clusts) or not len(clust_data):
            continue
        # Initialize log if one per event
        if store_per_event:
            fout = CSVData(os.path.join(logdir, 'cluster-gnn-metrics-event-%07d.csv' % tree_idx))

        # If there is no node, append default
        if not len(clusts[data_idx]) or not len(clust_data[data_idx]):
            fout.record(['ite', 'idx', 'ari', 'ami', 'sbd', 'pur', 'eff'], [iteration, tree_idx, -1, -1, -1, -1, -1])
            continue

        # Use group id to make node labels
        group_ids = []
        for c in clusts[data_idx]:
            v, cts = np.unique(clust_data[data_idx][c,column], return_counts=True)
            group_ids.append(int(v[cts.argmax()]))

        # Assign predicted group ids
        n = len(clusts[data_idx])
        if not bipartite:
            # Determine the predicted group IDs by using union find
            edge_assn = np.argmax(edge_pred[data_idx], axis=1)
            node_pred = node_assignment(edge_index[data_idx], edge_assn, n)
        else:
            # Determine the predicted group by chosing the most likely primary for each secondary
            primary_ids = np.unique(edge_index[data_idx][:,0])
            node_pred = node_assignment_bipartite(edge_index[data_idx], edge_pred[data_idx][:,1], primary_ids, n)

        # Evaluate clustering metrics
        ari, ami, sbd, pur, eff = clustering_metrics(clusts[data_idx], group_ids, node_pred)

        # Store
        fout.record(['ite', 'idx', 'ari', 'ami', 'sbd', 'pur', 'eff'], [iteration, tree_idx, ari, ami, sbd, pur, eff])
        fout.write()
        if store_per_event:
            fout.close()

    if not store_per_event:
        fout.close()
