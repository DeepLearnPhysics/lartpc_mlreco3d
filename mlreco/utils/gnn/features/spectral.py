from mlreco.utils.gnn.features.utils import *
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
import itertools
from scipy.sparse import diags, coo_matrix
from scipy.spatial import distance_matrix
from scipy.linalg import qr
from sklearn.neighbors import NearestNeighbors

# merge spectral clusters with directly neighboring voxels
def merge_neighbors(coords, cluster_assignments):
    neigh = NearestNeighbors(n_neighbors=6, radius=1.0)
    neigh.fit(coords)
    in_radius = neigh.radius_neighbors(coords)[1]

    labels_to_merge = []
    candidate_mergers = []
    for point in in_radius:
        sp_labels = np.unique(cluster_assignments[point])
        if len(sp_labels) > 1:
            merge = sp_labels.tolist()
            candidate_mergers.extend(list(itertools.combinations(merge, 2)))
    candidate_mergers = np.array(candidate_mergers)
    pairs, counts = np.unique(candidate_mergers, axis=0, return_counts=True)
    pairs = pairs[np.where(counts > 30)]
    for merge in pairs:
        merge_index = -1
        for i in range(len(labels_to_merge)):
            for m in merge:
                if m in labels_to_merge[i]:
                    merge_index = i
                    break
            if merge_index > -1:
                break
        if merge_index == -1:
            labels_to_merge.append(merge)
        else:
            labels_to_merge[merge_index] = list(set().union(labels_to_merge[merge_index], merge))
    for merge in labels_to_merge:
        for i in range(1, len(merge)):
            cluster_assignments[np.where(cluster_assignments == merge[i])] = merge[0]
    return cluster_assignments

def dist_metric(v1, v2, characteristic_length=45):
    norms = np.linalg.norm(v2-v1, axis=1)
    weights = np.exp(-norms/characteristic_length)
    return weights

def adjacency(positions, edges, eps_regularization=0.0001):
    n = len(positions)
    dists = dist_metric(positions[edges[:, 0]], positions[edges[:, 1]])
    weights = dists
    print('distance weights', weights)
#     directions = direction_metric(edges, positions, dists)
#     weights = directions*dists
#     print('directions', directions)
    
    # this is only upper triangular
    A_upper_half = coo_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(n, n), dtype=weights.dtype).tocsc()
    A_lower_half = coo_matrix((weights, (edges[:, 1], edges[:, 0])), shape=(n, n), dtype=weights.dtype).tocsc()
    A = A_upper_half + A_lower_half
    
    A = A + eps_regularization*(np.ones((n, n)) - np.identity(n))
    
    D_vec = np.abs(np.array(A.sum(axis=1)).flatten())
    D_norm_vec = 1/np.sqrt(D_vec)
    D = diags(D_vec)
    D_norm = diags(D_norm_vec)
    
    A_norm = D_norm * A * D_norm
    print('A normalized')
    
    return A_norm
    
def to_spectral_space(positions, edges, n_vecs=8):
    A = adjacency(positions, edges)
    _, vecs = eigsh(A, k=n_vecs, which='LA')
    vecs = np.flip(vecs, axis=0)
    return vecs

def spectral_edge_features(edges, vecs, cluster_assignments):
    edge_distances = np.linalg.norm(vecs[edges[:, 0]] - vecs[edges[:, 1]], axis=1)
    edge_labels = node_labels_to_edge_labels(edges, cluster_assignments)
    return np.concatenate((np.reshape(edge_distances, (-1, 1)), np.reshape(edge_labels, (-1, 1))), axis=1)

def spectral_node_features(edges, vecs, cluster_assignments):
    cluster_sizes = node_labels_to_cluster_sizes(cluster_assignments)
    return np.concatenate((np.reshape(cluster_sizes, (-1, 1)), vecs), axis=1)

# node features: [# voxels in cluster, embedded coord 1, ..., embedded coord n_vecs]
# edge features: [voxel pair distance in embedded space, edge labels]*len(n_clusts)
def spectral_features(data, em_filter, edges):
    positions = data['segment_label'][em_filter][:, :3]
    vecs = to_spectral_space(positions, edges)
    n_clusts = [4, 8]
    nf = []
    ef = []
#     for m in range(2): # merge
    for c in n_clusts:
        Q, R, P = qr(vecs[:, :c].T, pivoting=True)
        cluster_assignments = np.zeros(len(P))
        for i in range(len(P)):
            p = P[i]
            col = R[:, i]
            col_ind = np.argmax(np.abs(col))
            cluster_assignments[p] = np.sign(col[col_ind])*(col_ind + 1) # add 1 since -0 == 0
#         if m == 0:
#             cluster_assignments = merge_neighbors(positions, cluster_assignments)
        nf.append(spectral_node_features(edges, vecs, cluster_assignments))
        ef.append(spectral_edge_features(edges, vecs, cluster_assignments))
    nf = np.concatenate(tuple(nf), axis=1)
    ef = np.concatenate(tuple(ef), axis=1)
    return nf, ef