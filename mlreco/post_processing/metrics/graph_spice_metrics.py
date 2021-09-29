import os
import numpy as np
from mlreco.utils import CSVData

from mlreco.utils.metrics import *

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


def graph_spice_metrics(cfg, processor_cfg, data_blob, res, logdir, iteration):

    append = True if iteration else False
    ghost = cfg['post_processing']['graph_spice_metrics'].get('ghost', False)

    labels = data_blob['cluster_label'][0]
    data_index = data_blob['index']
    skip_classes = cfg['model']['modules']['graph_spice_loss']['skip_classes']
    invert = cfg['model']['modules']['graph_spice_loss']['invert']
    segmentation = res['segmentation'][0]
    if ghost:
        labels = adapt_labels(res, data_blob['segment_label'], data_blob['cluster_label'])
        labels = labels[0]
        ghost_mask = (res['ghost'][0].argmax(axis=1) == 0)
        segmentation = segmentation[ghost_mask]
        # print(labels.shape, segmentation.shape)

    #mask = ~np.isin(labels[:, -1], skip_classes)
    mask = ~np.isin(np.argmax(segmentation, axis=1), skip_classes)
    labels[:, -1] = torch.tensor(np.argmax(segmentation, axis=1))

    labels = labels[mask]
    #name = cfg['post_processing']['graph_spice_metrics']['output_filename']
    graph = res['graph'][0]

    graph_info = res['graph_info'][0]

    # Reassign index numbers
    index_mapping = { key : val for key, val in zip(
       range(0, len(graph_info.Index.unique())), data_index)}

    graph_info['Index'] = graph_info['Index'].map(index_mapping)
    # print(graph_info)

    constructor_cfg = cfg['model']['modules']['graph_spice']['constructor_cfg']

    gs_manager = ClusterGraphConstructor(constructor_cfg,
                                         graph_batch=graph,
                                         graph_info=graph_info,
                                         batch_col=0,
                                         training=False)
    gs_manager.fit_predict(gen_numpy_graph=True, invert=invert)
    funcs = [ARI, SBD, purity, efficiency, num_true_clusters, num_pred_clusters]
    df = gs_manager.evaluate_nodes(labels, funcs)

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
    invert = cfg['model']['modules']['graph_spice_loss']['invert']
    skip_classes = cfg['model']['modules']['graph_spice_loss']['skip_classes']
    segmentation = res['segmentation'][0]
    if ghost:
        labels = adapt_labels(res, data_blob['segment_label'], data_blob['cluster_label'])
        labels = labels[0]
        ghost_mask = (res['ghost'][0].argmax(axis=1) == 0)
        segmentation = segmentation[ghost_mask]

    use_labels = cfg['post_processing']['graph_spice_metrics_loop_threshold'].get('use_labels', True)

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
        gs_manager.fit_predict(gen_numpy_graph=True, invert=invert)
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
