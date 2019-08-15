from mlreco.utils.gnn.features.utils import *
import numpy as np
from mlreco.utils.gnn.features.cone import cone_features
from mlreco.utils.gnn.features.spectral import spectral_features
from mlreco.utils.gnn.features.dbscan import dbscan_features

# node features: x, y, z
# edge features: distance
def basic_features(data, em_filter, edges):
    positions = data['segment_label'][em_filter][:, :3]
    nf = positions
    ef = np.linalg.norm(positions[edges[:, 0]] - positions[edges[:, 1]], axis=1)
    ef = np.reshape(ef, (-1, 1))
    return nf, ef

# returns positions, edges, node features, edge features (assumes batch size == 1)
def generate_graph(data, feature_types=['basic', 'cone', 'spectral', 'dbscan']):
    em_filter = np.where(data['segment_label'] == 2)[0]
    positions = data['segment_label'][em_filter][:, :3]
    edges = create_edge_indices(positions)
    
    all_nf = []
    all_ef = []
    for ft in feature_types:
        if ft == 'basic':
            nf, ef = basic_features(data, em_filter, edges)
        elif ft == 'cone':
            nf, ef = cone_features(data, em_filter, edges)
        elif ft == 'spectral':
            nf, ef = spectral_features(data, em_filter, edges)
        elif ft == 'dbscan':
            nf, ef = dbscan_features(data, em_filter, edges)
        all_nf.append(nf)
        all_ef.append(ef)
    
    all_nf = np.concatenate(tuple(all_nf), axis=1)
    all_ef = np.concatenate(tuple(all_ef), axis=1)
    
    return positions, edges, all_nf, all_ef

# returns positions, edges, node features, edge features (assumes batch size == 1)
def generate_truth(data, positions=None, edges=None):
    em_filter = np.where(data['segment_label'] == 2)[0]
    if positions is None:
        positions = data['segment_label'][em_filter][:, :3]
    if edges is None:
        edges = create_edge_indices(positions)
    node_labels = data['group_label'][em_filter][:, -1]
    edge_labels = node_labels_to_edge_labels(edges, node_labels)
    return positions, edges, edge_labels