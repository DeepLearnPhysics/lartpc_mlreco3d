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


def graph_spice_metrics(cfg, data_blob, res, logdir, iteration):

    append = True if iteration else False

    labels = data_blob['cluster_label'][0]
    skip_classes = cfg['model']['modules']['spice_loss']['skip_classes']
    mask = ~np.isin(labels[:, -1], skip_classes)
    labels = labels[mask]

    name = cfg['post_processing']['graph_spice_metrics']['output_filename']

    graph = res['graph'][0]

    graph_info = res['graph_info'][0]
    constructor_cfg = cfg['model']['modules']['graph_spice']['constructor_cfg']

    pprint(constructor_cfg)

    gs_manager = ClusterGraphConstructor(constructor_cfg, 
                                         graph_batch=graph, 
                                         graph_info=graph_info)
    gs_manager.fit_predict(gen_numpy_graph=True)
    funcs = [ARI, SBD, purity, efficiency]
    df = gs_manager.evaluate_nodes(labels, funcs)

    fout = CSVData(os.path.join(logdir, 'graph-spice-metrics-{}.csv'.format(name)), append=append)

    for row in df.iterrows():
        columns = tuple(row[1].keys().values)
        values = tuple(row[1].values)
        fout.record(columns, values)
        fout.write()
    
    fout.close()