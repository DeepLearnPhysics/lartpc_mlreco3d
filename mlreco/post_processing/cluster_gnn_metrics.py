# GNN clustering prediction
import os
import numpy as np
from mlreco.utils import CSVData
from mlreco.utils.gnn.evaluation import edge_assignment, node_assignment, node_assignment_bipartite, clustering_metrics
from mlreco.utils.deghosting import adapt_labels_numpy as adapt_labels


def cluster_gnn_metrics(cfg, data_blob, res, logdir, iteration):
    deghosting = cfg['post_processing']['cluster_gnn_metrics'].get('ghost', False)

    # If there is no prediction, proceed
    cfg_edge_pred = cfg['post_processing']['cluster_gnn_metrics'].get('edge_pred', 'edge_pred')

    # Get the post processor parameters
    cfg_column = cfg['post_processing']['cluster_gnn_metrics'].get('column', 6)
    cfg_chain = cfg['post_processing']['cluster_gnn_metrics'].get('chain', 'chain')

    cfg_store_method = cfg['post_processing']['cluster_gnn_metrics']['store_method']

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
        seg_label = [seg_label[i][res['ghost'][i].argmax(axis=1) == 0] for i in range(len(seg_label))]

    for column, chain, store_method, filename, edge_pred_label, edge_index_label, clusts_label in zip(cfg_column, cfg_chain, cfg_store_method, cfg_filename, cfg_edge_pred, cfg_edge_index, cfg_clusts):
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

        # Loop over events
        for data_idx, tree_idx in enumerate(index):
            if not len(clusts) or not len(clust_data):
                continue
            # Initialize log if one per event
            if store_per_event:
                fout = CSVData(os.path.join(logdir, '%s-event-%07d.csv' % (filename, tree_idx)))

            # If there is no node, append default
            if not len(clusts[data_idx]) or not len(clust_data[data_idx]):
                fout.record(['ite', 'idx', 'ari', 'ami', 'sbd', 'pur', 'eff', 'num_clusts', 'num_pix'],
                            [iteration, tree_idx, -1, -1, -1, -1, -1, -1, -1])
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
