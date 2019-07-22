import itertools
import numpy as np
from scipy.spatial import Delaunay

def node_labels_to_edge_labels(edges, node_labels):
    label_starts = node_labels[edges[:, 0]]
    label_ends = node_labels[edges[:, 1]]
    edge_labels = np.ones(len(edges))
    edge_labels[np.where(label_starts != label_ends)] = 0
    return edge_labels

def find_parent(parent, i):
    if i != parent[i]:
        parent[i] = find_parent(parent, parent[i])
    return parent[i]

# union find
def edge_labels_to_node_labels(positions, edges, edge_labels, threshold=0.5, node_len=None):
    print('node_len', node_len)
    print('max edge label', np.amax(edge_labels))
    on_edges = edges[np.where(edge_labels > threshold)[0]]
    if node_len is not None:
        node_labels = np.arange(node_len)
    else:
        node_labels = np.arange(len(positions))
    for a, b in on_edges:
        p1 = find_parent(node_labels, a)
        p2 = find_parent(node_labels, b)
        if p1 != p2:
            node_labels[p1] = p2
    return node_labels

def node_labels_to_cluster_sizes(node_labels):
    unique, counts = np.unique(node_labels, return_counts=True)
    sizes = np.zeros(len(node_labels))
    for i in range(len(unique)):
        sizes[np.where(node_labels == unique[i])] = counts[i]
    return sizes
    
def create_edge_indices(positions):
    n = len(positions)
    nodes = np.arange(n)
    
    simplices = Delaunay(positions).simplices
    simplices.sort()
    edges = set()
    for s in simplices:
        edges |= set(itertools.combinations(s, 2))
    edges = np.array(list(edges))
    return edges