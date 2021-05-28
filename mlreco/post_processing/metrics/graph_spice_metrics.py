import os
import numpy as np
from mlreco.utils import CSVData

from mlreco.utils.metrics import *

from pprint import pprint

from mlreco.utils.cluster.graph_spice import (
    ClusterGraphConstructor, get_edge_weight)
from mlreco.utils.metrics import ARI, SBD, purity, efficiency
from mlreco.models.cluster_cnn.losses.spatial_embeddings import *


def extent(voxels):
    centroid = voxels[:, :3].mean(axis=0)
    return np.linalg.norm(voxels[:, :3] - centroid, axis=1)

def num_true_clusters(pred, truth):
    return len(np.unique(truth))

def num_pred_clusters(pred, truth):
    return len(np.unique(pred))


def graph_spice_metrics(cfg, processor_cfg, data_blob, res, logdir, iteration):

    append = True if iteration else False

    labels = data_blob['cluster_label'][0]
    data_index = data_blob['index']
    skip_classes = cfg['model']['modules']['spice_loss']['skip_classes']
    mask = ~np.isin(labels[:, -1], skip_classes)
    labels = labels[mask]

    name = cfg['post_processing']['graph_spice_metrics']['output_filename']

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
                                         graph_info=graph_info)
    gs_manager.fit_predict(gen_numpy_graph=True, invert=True)
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

    labels = data_blob['cluster_label'][0]
    data_index = data_blob['index']
    skip_classes = cfg['model']['modules']['spice_loss']['skip_classes']
    mask = ~np.isin(labels[:, -1], skip_classes)
    labels = labels[mask]

    name = cfg['post_processing']['graph_spice_metrics_loop_threshold']['output_filename']

    graph = res['graph'][0]

    graph_info = res['graph_info'][0]

    # Reassign index numbers
    index_mapping = { key : val for key, val in zip(
        range(0, len(graph_info.Index.unique())), data_index)}
    
    graph_info['Index'] = graph_info['Index'].map(index_mapping)
    # print(graph_info)

    constructor_cfg = cfg['model']['modules']['graph_spice']['constructor_cfg']

    edge_ths_range = np.linspace(0.01, 0.1, 20)

    for edge_ths in edge_ths_range:

        edge_threshold = lambda x,y: edge_ths

        constructor_cfg['edge_cut_threshold'] = edge_ths

        gs_manager = ClusterGraphConstructor(constructor_cfg, 
                                            graph_batch=graph, 
                                            graph_info=graph_info)
        gs_manager.fit_predict(gen_numpy_graph=True, invert=True)
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