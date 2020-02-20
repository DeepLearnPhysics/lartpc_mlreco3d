# Defines network incidence matrices
import numpy as np
import torch

def complete_graph(batches, dist=None, max_dist=-1):
    """
    Function that returns an incidence matrix of a complete graph
    that connects every node with ever other node.

    Args:
        batches (np.ndarray): (C) List of batch ids
        dist (np.ndarray)   : (C,C) Tensor of pair-wise cluster distances
        max_dist (double)   : Maximal edge length
    Returns:
        np.ndarray: (2,E) Tensor of edges
    """
    # Create the incidence matrix
    ids = np.arange(len(batches))
    edges = [[i, j] for i in ids for j in ids if (batches[i] == batches[j] and j != i)]
    if not len(edges):
        return np.empty((2,0))
    ret = np.vstack(edges).T

    # If requested, remove the edges above a certain length threshold
    if max_dist > -1:
        dists = np.array([dist[i, j] for i in ids for j in ids if (batches[i] == batches[j] and j != i)])
        ret = ret[:,dists < max_dist]

    return ret


def delaunay_graph(data, clusts, dist=None, max_dist=-1):
    """
    Function that returns an incidence matrix that connects nodes
    that share an edge in their corresponding Euclidean Delaunay graph.

    Args:
        data (np.ndarray)    : (N,4) [x, y, z, batchid]
        clusts ([np.ndarray]): (C) List of arrays of voxel IDs in each cluster
        dist (np.ndarray)    : (C,C) Tensor of pair-wise cluster distances
        max_dist (double)    : Maximal edge length
    Returns:
        np.ndarray: (2,E) Tensor of edges
    """
    # Only keep the voxels that are in the clusters
    mask = np.concatenate(clusts)
    labels = np.concatenate([np.full(len(c), i) for i, c in enumerate(clusts)])
    voxels = data[mask,:3]
    batches = data[mask,3]

    # For each batch, find the list of edges, append it
    from scipy.spatial import Delaunay
    ret = np.empty((0, 2), dtype=int)
    for i in np.unique(batches):
        where = np.where(batches == i)[0]
        tri = Delaunay(voxels[where], qhull_options='QJ') # Joggled input guarantees simplical faces
        edges = np.array([[labels[where[i]], labels[where[j]]] for s in tri.simplices for i in s for j in s if labels[where[i]] < labels[where[j]]])
        if len(edges):
            ret = np.vstack((ret, np.unique(edges, axis=0)))

    # If requested, remove the edges above a certain length threshold
    if max_dist > -1:
        dists = np.array([dist[e[0], e[1]] for e in ret])
        ret = ret[dists < max_dist]

    return ret.T


def mst_graph(batches, dist, max_dist=-1):
    """
    Function that returns an incidence matrix that connects nodes
    that share an edge in their corresponding Euclidean Minimum Spanning Tree (MST).

    Args:
        batches (np.ndarray) : (N) List of batch ids
        dist (np.ndarray)    : (C,C) Tensor of pair-wise cluster distances
        max_dist (double)    : Maximal edge length
    Returns:
        np.ndarray: (2,E) Tensor of edges
    """
    from scipy.sparse.csgraph import minimum_spanning_tree
    mst_mat = minimum_spanning_tree(dist).toarray().astype(float)
    inds = np.where(mst_mat.flatten() > 0.)[0]
    ret = np.array(np.unravel_index(inds, mst_mat.shape))
    ret = np.sort(ret, axis=0)

    # If requested, remove the edges above a certain length threshold
    if max_dist > -1:
        dists = mst_mat.flatten()[inds]
        ret = ret[:,dists < max_dist]

    return ret


def bipartite_graph(batches, primaries, dist=None, max_dist=-1):
    """
    Function that returns an incidence matrix of the bipartite graph
    between primary nodes and non-primary nodes.

    Args:
        batches (np.ndarray)  : (C) List of batch ids
        primaries (np.ndarray): (P) List of primary ids
        dist (np.ndarray)     : (C,C) Tensor of pair-wise cluster distances
        max_dist (double)     : Maximal edge length
    Returns:
        np.ndarray: (2,E) Tensor of edges
    """
    # Create the incidence matrix
    others = [i for i in range(len(batches)) if i not in primaries]
    edges = [[i.item(), j] for i in primaries for j in others if batches[i] == batches[j]]
    if not len(edges):
        return np.empty((2,0))
    ret = np.vstack(edges).T

    # If requested, remove the edges above a certain length threshold
    if max_dist > -1:
        dists = np.array([dist[i, j] for i in primaries for j in others if batches[i] == batches[j]])
        ret = ret[:,np.where(dists < max_dist)[0]]

    return ret


def inter_cluster_distance(voxels, clusts, mode='set'):
    """
    Function that returns the matrix of pair-wise cluster distances.

    Args:
        voxels (np.ndarray)  : (N,3) Tensor of voxel coordinates
        clusts ([np.ndarray]): (C) List of arrays of voxel IDs in each cluster
        mode (str)           : Maximal edge length (distance mode: set or centroid)
    Returns:
        np.ndarray: (C,C) Tensor of pair-wise cluster distances
    """
    from scipy.spatial.distance import cdist
    if mode == 'set':
        dist_mat = np.array([np.min(cdist(voxels[ci], voxels[cj])) for ci in clusts for cj in clusts]).reshape((len(clusts), len(clusts)))
    elif mode == 'centroid':
        centroids = np.vstack([np.mean(voxels[c], axis=0) for c in clusts])
        dist_mat = cdist(centroids, centroids)
    else:
        raise(ValueError('Distance mode not recognized: '+mode))

    return dist_mat


def inter_cluster_distance(voxels, clusts, mode='set'):
    """
    Function that returns the matrix of pair-wise cluster distances.

    Args:
        voxels (torch.tensor): (N,3) Tensor of voxel coordinates
        clusts ([np.ndarray]): (C) List of arrays of voxel IDs in each cluster
        mode (str)           : Maximal edge length (distance mode: set or centroid)
    Returns:
        torch.tensor: (C,C) Tensor of pair-wise cluster distances
    """
    if mode == 'set':
        from mlreco.utils import local_cdist
        dist_mat = np.zeros(shape=(len(clusts),len(clusts)),dtype=np.float32)
        for i, ci in enumerate(clusts):
            for j, cj in enumerate(clusts):
                if i < j:
                    dist_mat[i,j] = local_cdist(voxels[ci], voxels[cj]).min()
                else:
                    dist_mat[i,j] = dist_mat[j,i]

    elif mode == 'centroid':
        centroids = torch.stack([torch.mean(voxels[c], dim=0) for c in clusts])
        dist_mat = local_cdist(centroids, centroids)
    else:
        raise(ValueError('Distance mode not recognized: '+mode))

    return dist_mat


def get_fragment_edges(graph, clust_ids):
    """
    Function that converts a set of edges between cluster ids
    to a set of edges between fragment ids (ordering in list)

    Args:
        graph (torch.tensor)    : (E,3) Tensor of [clust_id_1, clust_id_2, batch_id]
        clust_ids (np.ndarray): (C) List of fragment cluster ids
        batch_ids (np.ndarray): (C) List of fragment batch ids
    Returns:
        np.ndarray: (E,2) Tensor of true edges [frag_id_1, frag_id2]
    """
    # Loop over the graph edges, find the fragment ids, append
    true_edges = np.empty((0,2), dtype=int)
    for e in graph:
        n1 = np.where(clust_ids == e[0].item())[0]
        n2 = np.where(clust_ids == e[1].item())[0]
        if len(n1) and len(n2):
            true_edges = np.vstack((true_edges, [n1[0], n2[0]]))

    return true_edges

