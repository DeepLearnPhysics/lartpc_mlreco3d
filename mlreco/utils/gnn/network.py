# Defines network incidence matrices
import numpy as np
import torch

def loop_graph(n):
    """
    Function that returns an incidence matrix of a graph that only
    connects nodes with themselves.

    Args:
        n (int): Number of nodes C
    Returns:
        np.ndarray: (2,C) Tensor of edges
    """
    # Create the incidence matrix
    if not n:
        return np.empty((2,0))
    return np.vstack([[i,i] for i in range(n)]).T


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
    edges = [[i, j] for i in ids for j in ids if (batches[i] == batches[j] and j > i)]
    if not len(edges):
        return np.empty((2,0))
    ret = np.vstack(edges)

    # If requested, remove the edges above a certain length threshold
    if max_dist > -1:
        dists = np.array([dist[i, j] for i in ids for j in ids if (batches[i] == batches[j] and j > i)])
        ret = ret[dists < max_dist]

    # Add the reciprocal edges as to create an undirected graph
    ret = np.vstack((ret, np.flip(ret, axis=1)))

    return ret.T


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
    for b in np.unique(batches):
        where = np.where(batches == b)[0]
        tri = Delaunay(voxels[where], qhull_options='QJ') # Joggled input guarantees simplical faces
        edges = np.array([[labels[where[i]], labels[where[j]]] for s in tri.simplices for i in s for j in s if labels[where[i]] < labels[where[j]]])
        if len(edges):
            ret = np.vstack((ret, np.unique(edges, axis=0)))

    # If requested, remove the edges above a certain length threshold
    if max_dist > -1:
        dists = np.array([dist[e[0], e[1]] for e in ret])
        ret = ret[dists < max_dist]

    # Add the reciprocal edges as to create an undirected graph
    ret = np.vstack((ret, np.flip(ret, axis=1)))

    return ret.T


def mst_graph(batches, dist, max_dist=-1):
    """
    Function that returns an incidence matrix that connects nodes
    that share an edge in their corresponding Euclidean Minimum Spanning Tree (MST).

    Args:
        batches (np.ndarray) : (C) List of batch ids
        dist (np.ndarray)    : (C,C) Tensor of pair-wise cluster distances
        max_dist (double)    : Maximal edge length
    Returns:
        np.ndarray: (2,E) Tensor of edges
    """
    # For each batch, find the list of edges, append it
    from scipy.sparse.csgraph import minimum_spanning_tree
    ret = np.empty((0, 2), dtype=int)
    for b in np.unique(batches):
        where = np.where(batches == b)[0]
        if len(where) > 1:
            mst_mat = minimum_spanning_tree(dist[np.ix_(where,where)]).toarray().astype(float)
            inds = np.where(mst_mat.flatten() > 0.)[0]
            ind_pairs = np.array(np.unravel_index(inds, mst_mat.shape)).T
            edges = np.array([[where[i], where[j]] for i, j in ind_pairs])
            edges.sort(axis = 1)
            ret = np.vstack((ret, edges))

    # If requested, remove the edges above a certain length threshold
    if max_dist > -1:
        dists = np.array([dist[e[0], e[1]] for e in ret])
        ret = ret[dists < max_dist]

    # Add the reciprocal edges as to create an undirected graph
    ret = np.vstack((ret, np.flip(ret, axis=1)))

    return ret.T


def knn_graph(batches, dist, k=5, undirected=False):
    """
    Function that returns an incidence matrix that connects nodes
    that are k nearest neighbors. Sorts the distance matrix.

    Args:
        batches (np.ndarray) : (C) List of batch ids
        dist (np.ndarray)    : (C,C) Tensor of pair-wise cluster distances
        k (int)            : Number of connected neighbors for each node
    Returns:
        np.ndarray: (2,E) Tensor of edges
    """
    # Use the available distance matrix to build a kNN graph
    ret = np.empty((0, 2), dtype=int)
    for b in np.unique(batches):
        where = np.where(batches == b)[0]
        if len(where) > 1:
            k = min(k+1, len(where))
            idxs = np.argsort(dist[np.ix_(where,where)], axis=1)[:,1:k]
            for i, v in enumerate(idxs):
                edges = np.array([[where[j], where[i]] for j in v])
                if len(edges):
                    ret = np.vstack((ret, edges))

    # Add the reciprocal edges as to create an undirected graph
    if undirected:
        ret = np.vstack((ret, np.flip(ret, axis=1)))

    return ret.T


def kdtree_graph(data, k=5, undireced=False):
    """
    Function that returns an incidence matrix that connects nodes
    that are k nearest neighbors. Uses a KDTree on the input points.

    Args:
        data (np.ndarray): (N,4) [x, y, z, batchid]
        k (int)            : Number of connected neighbors for each node
    Returns:
        np.ndarray: (2,E) Tensor of edges
    """
    # Build a KDTree on each batch, find the nearest neighbors
    from scipy.spatial import cKDTree
    voxels  = data[:,:3]
    batches = data[:,3]
    ret = np.empty((0, 2), dtype=int)
    for b in np.unique(batches):
        where = np.where(batches == b)[0]
        kdtree = cKDTree(voxels[where])
        for i, v in enumerate(voxels[where]):
            _, idxs = kdtree.query(v, k=k)
            edges = np.array([[where[j], where[i]] for j in idxs])
            if len(edges):
                ret = np.vstack((ret, edges))

    # Add the reciprocal edges as to create an undirected graph
    if undirected:
        ret = np.vstack((ret, np.flip(ret, axis=1)))

    return ret.T


def bipartite_graph(batches, primaries, dist=None, max_dist=-1, directed=True, directed_to='secondary'):
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
    edges = [[i, j] for i in primaries for j in others if batches[i] == batches[j]]
    if not len(edges):
        return np.empty((2,0))
    ret = np.vstack(edges)

    # If requested, remove the edges above a certain length threshold
    if max_dist > -1:
        dists = np.array([dist[i, j] for i in primaries for j in others if batches[i] == batches[j]])
        ret = ret[dists < max_dist]

    # Handle directedness, by default graph is directed towards secondaries
    if directed:
        if directed_to == 'primary':
            ret = np.flip(ret, axis=1)
        elif directed_to != 'secondary':
            raise ValueError('Graph orientation not recognized:', directed_to)
    else:
        ret = np.vstack((ret, np.flip(ret, axis=1)))

    return ret.T


def inter_cluster_distance(voxels, clusts, batch_ids, mode='set', use_numpy=True):
    """
    Function that returns the matrix of pair-wise cluster distances.

    Args:
        voxels (torch.tensor) : (N,3) Tensor of voxel coordinates
        clusts ([np.ndarray]) : (C) List of arrays of voxel IDs in each cluster
        batch_ids (np.ndarray): (C) List of cluster batch IDs
        mode (str)            : Maximal edge length (distance mode: set or centroid)
    Returns:
        torch.tensor: (C,C) Tensor of pair-wise cluster distances
    """
    from scipy.spatial.distance import cdist
    from mlreco.utils import local_cdist
    from scipy.linalg import block_diag

    if mode == 'set':
        dist_mats = []
        for b in np.unique(batch_ids):
            bclusts = np.array(clusts)[batch_ids == b]
            dist_mat = np.zeros(shape=(len(bclusts),len(bclusts)),dtype=np.float32)
            for i, ci in enumerate(bclusts):
                for j, cj in enumerate(bclusts):
                    if i < j:
                        if use_numpy:
                            dist_mat[i,j] = cdist(voxels[ci].detach().cpu().numpy(), voxels[cj].detach().cpu().numpy()).min()
                        else:
                            dist_mat[i,j] = local_cdist(voxels[ci], voxels[cj]).min()
                    else:
                        dist_mat[i,j] = dist_mat[j,i]
            dist_mats.append(dist_mat)
        dist_mat = block_diag(*dist_mats)

    elif mode == 'centroid':
        dist_mats = []
        for b in np.unique(batch_ids):
            bclusts = np.array(clusts)[batch_ids == b]
            centroids = torch.stack([torch.mean(voxels[c], dim=0) for c in bclusts])
            if use_numpy:
                centroids = centroids.detach().cpu().numpy()
                dist_mat = cdist(centroids, centroids)
            else:
                dist_mat = local_cdist(centroids, centroids)
            dist_mats.append(dist_mat)
        dist_mat = block_diag(*dist_mats)

    else:
        raise(ValueError('Distance mode not recognized: '+mode))

    return dist_mat


def get_fragment_edges(graph, clust_ids):
    """
    Function that converts a set of edges between cluster ids
    to a set of edges between fragment ids (ordering in list)

    Args:
        graph (torch.tensor)  : (E,2) Tensor of [clust_id_1, clust_id_2]
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
