import os
import numpy as np
from mlreco.utils import CSVData

from mlreco.utils.metrics import *
from mlreco.utils.cluster.graph_batch import GraphBatch

from pprint import pprint

from mlreco.utils.deghosting import adapt_labels_numpy as adapt_labels
from mlreco.utils.cluster.cluster_graph_constructor import (
    ClusterGraphConstructor, get_edge_weight)
from mlreco.utils.metrics import ARI, SBD, purity, efficiency
from mlreco.models.layers.cluster_cnn.losses.spatial_embeddings import *

def num_true_clusters(pred, truth):
    return len(np.unique(truth))

def num_pred_clusters(pred, truth):
    return len(np.unique(pred))

def num_small_clusters(pred, truth, threshold=5):
    val, cnts = np.unique(pred, return_counts=True)
    return np.count_nonzero(cnts < threshold)

def modified_ARI(pred, truth, threshold = 5):
    val, cnts = np.unique(pred, return_counts=True)
    mask = np.isin(pred, val[cnts >= threshold])
    val, cnts = np.unique(truth, return_counts=True)
    mask2 = np.isin(truth, val[cnts >= threshold])
    return ARI(pred[mask & mask2], truth[mask & mask2])

def modified_purity(pred, truth, threshold = 5):
    val, cnts = np.unique(pred, return_counts=True)
    mask = np.isin(pred, val[cnts >= threshold])
    val, cnts = np.unique(truth, return_counts=True)
    mask2 = np.isin(truth, val[cnts >= threshold])
    return purity(pred[mask & mask2], truth[mask & mask2])

def modified_efficiency(pred, truth, threshold = 5):
    val, cnts = np.unique(pred, return_counts=True)
    mask = np.isin(pred, val[cnts >= threshold])
    val, cnts = np.unique(truth, return_counts=True)
    mask2 = np.isin(truth, val[cnts >= threshold])
    return efficiency(pred[mask & mask2], truth[mask & mask2])

def graph_spice_metrics(cfg, processor_cfg, data_blob, res, logdir, iteration):

    append = True if iteration else False
    ghost = cfg['post_processing']['graph_spice_metrics'].get('ghost', False)

    labels = data_blob['cluster_label'][0]
    data_index = data_blob['index']

    skip_classes = cfg['model']['modules']['graph_spice']['skip_classes']
    min_points = cfg['model']['modules']['graph_spice'].get('min_points', 1)
    invert = cfg['model']['modules']['graph_spice_loss'].get('invert', True)
    use_labels = cfg['post_processing']['graph_spice_metrics'].get('use_labels', True)

    segmentation = np.concatenate(res['segmentation'], axis=0)
    if ghost:
        labels = adapt_labels(res, data_blob['segment_label'], data_blob['cluster_label'])
        labels = np.concatenate(labels, axis=0)#labels[0]
        ghost_mask = np.concatenate(res['ghost'], axis=0)
        ghost_mask = (ghost_mask.argmax(axis=1) == 0)
        segmentation = segmentation[ghost_mask]

    if not use_labels:
        semantic_pred = torch.tensor(np.argmax(segmentation, axis=1))
        # Only compute loss on voxels where true/predicted semantics agree
        labels[:, 5] = np.where(semantic_pred.cpu().numpy() == labels[:, -1].astype(int), labels[:, 5], -1)
        labels[:, -1] = semantic_pred

    mask = ~np.isin(labels[:, -1], skip_classes)

    labels = labels[mask]
    if labels.shape[0] == 0:
        return
    batch_ids = np.unique(labels[:, 0])
    #name = cfg['post_processing']['graph_spice_metrics']['output_filename']
    graph = res['graph'][0]

    # graph_batch_ids = graph.batch.unique().cpu().numpy()
    # batch_mask = np.isin(graph_batch_ids, batch_ids)
    # graph_list = graph.to_data_list()
    # corrected_batch_list = np.arange(len(graph_list))[batch_mask]
    # graph = GraphBatch.from_data_list([graph_list[idx] for idx in corrected_batch_list])

    graph_info = res['graph_info'][0]

    # Reassign index numbers
    index_mapping = { key : val for key, val in zip(
       range(0, len(graph_info.Index.unique())), data_index)}

    graph_info['Index'] = graph_info['Index'].map(index_mapping)
    # graph_info = graph_info[graph_info['Index'].isin(corrected_batch_list)]

    constructor_cfg = cfg['model']['modules']['graph_spice']['constructor_cfg']

    gs_manager = ClusterGraphConstructor(constructor_cfg,
                                         graph_batch=graph,
                                         graph_info=graph_info,
                                         batch_col=0,
                                         training=False)
    gs_manager.fit_predict(invert=invert, min_points=min_points)
    funcs = [ARI, purity, efficiency]
            # num_true_clusters, num_pred_clusters,
            # num_small_clusters, modified_ARI, modified_purity, modified_efficiency]
    df = gs_manager.evaluate_nodes(labels, funcs)
    #import pandas as pd
    #pd.set_option('display.max_columns', None)

    fout = CSVData(os.path.join(logdir, 'graph-spice-metrics.csv'), append=append)

    for row in df.iterrows():
        columns = tuple(row[1].keys().values)
        values = tuple(row[1].values)
        fout.record(columns, values)
        fout.write()

    fout.close()


def graph_spice_metrics_loop_threshold(cfg, processor_cfg, data_blob, res, logdir, iteration):

    append = True if iteration else False
    ghost = cfg['post_processing']['graph_spice_metrics_loop_threshold'].get('ghost', False)

    labels = data_blob['cluster_label'][0]
    data_index = data_blob['index']
    invert = cfg['model']['modules']['graph_spice_loss'].get('invert', True)
    skip_classes = cfg['model']['modules']['graph_spice']['skip_classes']
    min_points = cfg['model']['modules']['graph_spice'].get('min_points', 1)
    use_labels = cfg['post_processing']['graph_spice_metrics_loop_threshold'].get('use_labels', True)

    if not use_labels:
        segmentation = np.concatenate(res['segmentation'], axis=0)
        if ghost:
            labels = adapt_labels(res, data_blob['segment_label'], data_blob['cluster_label'])
            labels = np.concatenate(labels, axis=0)#labels[0]
            ghost_mask = np.concatenate(res['ghost'], axis=0)
            ghost_mask = (ghost_mask.argmax(axis=1) == 0)
            segmentation = segmentation[ghost_mask]

    if use_labels:
        mask = ~np.isin(labels[:, -1], skip_classes)
    else:
        mask = ~np.isin(np.argmax(segmentation, axis=1), skip_classes)
        labels[:, -1] = torch.tensor(np.argmax(segmentation, axis=1))

    labels = labels[mask]

    #name = cfg['post_processing']['graph_spice_metrics_loop_threshold']['output_filename']

    graph = res['graph'][0]

    graph_info = res['graph_info'][0]

    # Reassign index numbers
    index_mapping = { key : val for key, val in zip(
        range(0, len(graph_info.Index.unique())), data_index)}

    graph_info['Index'] = graph_info['Index'].map(index_mapping)
    # print(graph_info)

    constructor_cfg = cfg['model']['modules']['graph_spice']['constructor_cfg']

    min_ths = cfg['post_processing']['graph_spice_metrics_loop_threshold'].get('min_edge_threshold', 0.)
    max_ths = cfg['post_processing']['graph_spice_metrics_loop_threshold'].get('max_edge_threshold', 1.)
    step_ths = cfg['post_processing']['graph_spice_metrics_loop_threshold'].get('step_edge_threshold', 0.1)

    edge_ths_range = np.arange(min_ths, max_ths, step_ths)

    for edge_ths in edge_ths_range:

        edge_threshold = lambda x,y: edge_ths

        constructor_cfg['edge_cut_threshold'] = edge_ths

        gs_manager = ClusterGraphConstructor(constructor_cfg,
                                            graph_batch=graph,
                                            graph_info=graph_info)
        gs_manager.fit_predict(gen_numpy_graph=True, invert=invert, min_points=min_points)
        funcs = [ARI, SBD, purity, efficiency, num_true_clusters,
                 num_pred_clusters, edge_threshold]
        column_names = ['ARI', 'SBD', 'Purity', 'Efficiency', 'num_true_clusters',
                        'num_pred_clusters', 'edge_threshold']
        df = gs_manager.evaluate_nodes(labels, funcs, column_names=column_names)

        fout = CSVData(os.path.join(logdir, 'graph-spice-metrics-loop.csv'), append=append)

        for row in df.iterrows():
            columns = tuple(row[1].keys().values)
            values = tuple(row[1].values)
            fout.record(columns, values)
            fout.write()

    fout.close()
