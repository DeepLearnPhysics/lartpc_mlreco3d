import os
import numpy as np
from mlreco.utils import CSVData
from mlreco.utils.deghosting import adapt_labels_numpy as adapt_labels
from mlreco.utils.metrics import *
from mlreco.utils.occuseg import *
import networkx as nx


def graphspice_nognn(cfg, data_blob, res, logdir, iteration):
    deghosting = cfg['post_processing']['graphspice_nognn'].get('ghost', False)

    store_method = cfg['post_processing']['graphspice_nognn']['store_method']
    store_per_event = store_method == 'per-event'

    batch_col = cfg['post_processing']['graphspice_nognn']['batch_col']
    cluster_col = cfg['post_processing']['graphspice_nognn']['cluster_col']

    cut_threshold = cfg['post_processing']['graphspice_nognn']['cut_threshold']
    min_cluster = cfg['post_processing']['graphspice_nognn']['min_cluster']

    predictor_cfg = cfg['post_processing']['graphspice_nognn']['predictor_cfg']
    constructor_cfg = cfg['post_processing']['graphspice_nognn']['constructor_cfg']

    if store_method == 'per-iteration':
        fout = CSVData(os.path.join(logdir, 'graphspice_nognn-iter-%07d.csv' % iteration))
    if store_method == 'single-file':
        append = True if iteration else False
        fout = CSVData(os.path.join(logdir, 'graphspice_nognn.csv'), append=append)

    # Results
    segmentation = res['segmentation'][0]
    covariance = res['covariance'][0]
    occupancy = res['occupancy'][0]
    sp_embeddings = res['spatial_embeddings'][0]
    ft_embeddings = res['feature_embeddings'][0]

    # Labels
    input_data = data_blob['input_data'][0]
    labels = data_blob['cluster_label'][0]
    batch_index = input_data[:, batch_col].astype(int)
    nbatches = len(np.unique(batch_index))

    graph = res['graph'][0]
    edge_index = graph.edge_index
    graph_index = np.asarray(graph.index)
    graph_list = graph.to_data_list()

    edge_pred = graph.edge_attr.cpu().numpy()
    # forward_time_per_image = float(end - start) / float(nbatches)

    # Loop over non-degenerate graphs:
    for i, entry in enumerate(graph.index):
        bidx, c = entry
        event_id = data_blob['index'][bidx]
        # Initialize log if one per event
        if store_per_event:
            fout = CSVData(os.path.join(logdir, 'graphspice_nognn-event-%07d.csv' % tree_idx))

        seg_batch = segmentation[batch_index == bidx]
        cov_batch = covariance[batch_index == bidx]
        occ_batch = occupancy[batch_index == bidx]
        sp_batch = sp_embeddings[batch_index == bidx]
        ft_batch = ft_embeddings[batch_index == bidx]
        labels_batch = labels[batch_index == bidx]

        pred_seg = np.argmax(seg_batch, axis=1).astype(int)
        # class_mask = labels_batch[:, -1] == klass
        class_mask = pred_seg == c
        seg_class = seg_batch[class_mask]
        cov_class = cov_batch[class_mask]
        occ_class = occ_batch[class_mask]
        sp_class = sp_batch[class_mask]
        ft_class = ft_batch[class_mask]
        coords_class = labels_batch[:, :batch_col][class_mask]
        frag_labels, _ = unique_label(labels_batch[:, cluster_col][class_mask])

        graph_id = int( np.where( (graph_index == (bidx, c) ).all(axis=1))[0] )
        node_indices = np.arange(graph.batch.shape[0])[graph.batch == graph_id]

        subgraph = graph_list[graph_id]
        edge_probs = edge_pred[np.isin(graph.edge_index.T[:, 0].cpu().numpy(), node_indices)]
        edge_pred_class = edge_probs.squeeze() > cut_threshold
        edge_truth_class = get_edge_truth(subgraph.edge_index.cpu().numpy(), frag_labels)
        edge_class = subgraph.edge_index.T

        hypergraph_features = np.concatenate(
            [coords_class, ft_class, sp_class, cov_class, occ_class], axis=1)

        pred_labels, _ = fit_graph(coords_class,
                                   edge_class,
                                   edge_pred_class,
                                   hypergraph_features,
                                   min_cluster=min_cluster)

        ari = ARI(pred_labels, frag_labels)
        pur, eff = purity_efficiency(pred_labels, frag_labels)

        true_num_clusters = len(np.unique(frag_labels))
        pred_num_clusters = len(np.unique(pred_labels))

        sbd = SBD(pred_labels, frag_labels)

        tp = np.sum(edge_truth_class & edge_pred_class).astype(float)
        tn = np.sum(~edge_truth_class & ~edge_pred_class).astype(float)
        fp = np.sum(~edge_truth_class & edge_pred_class).astype(float)
        fn = np.sum(edge_truth_class & ~edge_pred_class).astype(float)

        precision = (tp + 1e-6) / (tp + fp + 1e-6)
        recall = (tp + 1e-6) / (tp + fn + 1e-6)
        tnr = (tn + 1e-6) / (tn + fp + 1e-6)
        npv = (tn + 1e-6) / (tn + fn + 1e-6)

        f1_pos = 2.0 * precision * recall / (precision + recall + 1e-6)
        f1_neg = 2.0 * tnr * npv / (tnr + npv + 1e-6)

        row = (
            iteration, event_id, c, ari, sbd, pur, eff,
            true_num_clusters,
            pred_num_clusters,
            precision, recall, tnr, npv, f1_pos, f1_neg)

        if (c == 0) or (c == 1):
            print('Class = {}, ARI = {:.4f}'.format(c, ari))

        fout.record((
            'Iteration', 'ID', 'Class', 'ARI', 'SBD', 'Purity',
            'Efficiency',
            'true_num_clusters',
            'pred_num_clusters',
            'Precision', 'Recall', 'THR', 'NPV', 'F1_POS', 'F1_NEG'), row)

        fout.write()

    if store_per_event:
        fout.close()

    if not store_per_event:
        fout.close()