import numpy as np
import numba as nb
import torch

from scipy.spatial import Delaunay
from scipy.sparse.csgraph import minimum_spanning_tree

import mlreco.utils.numba_local as nbl
from mlreco.utils.decorators import numbafy
from mlreco.utils.globals import COORD_COLS


@nb.njit(cache=True)
def loop_graph(n: nb.int64) -> nb.int64[:,:]:
    """
    Function that returns an incidence matrix of a graph that only
    connects nodes with themselves.

    Args:
        n (int): Number of nodes C
    Returns:
        np.ndarray: (2,C) Tensor of edges
    """
    # Create the incidence matrix
    ret = np.empty((2,n), dtype=np.int64)
    for k in range(n):
        ret[k] = [k,k]

    return ret.T


@nb.njit(cache=True)
def complete_graph(batch_ids: nb.int64[:],
                   directed: bool = False) -> nb.int64[:,:]:
    """
    Function that returns an incidence matrix of a complete graph
    that connects every node with ever other node.

    Args:
        batch_ids (np.ndarray): (C) List of batch ids
        directed (bool)       : If directed, only keep edges [i,j] for which j>=i
    Returns:
        np.ndarray: (2,E) Tensor of edges
    """
    # Count the number of edges in the graph
    edge_count = 0
    for b in np.unique(batch_ids):
        edge_count += np.sum(batch_ids == b)**2
    edge_count = int((edge_count-len(batch_ids))/2)
    if not edge_count:
        return np.empty((2,0), dtype=np.int64)

    # Create the sparse incidence matrix
    ret = np.empty((edge_count,2), dtype=np.int64)
    k = 0
    for b in np.unique(batch_ids):
        clust_ids = np.where(batch_ids == b)[0]
        for i in range(len(clust_ids)):
            for j in range(i+1, len(clust_ids)):
                ret[k] = [clust_ids[i], clust_ids[j]]
                k += 1

    # Add the reciprocal edges as to create an undirected graph, if requested
    if not directed:
        ret = np.vstack((ret, ret[:,::-1]))

    return ret.T

@nb.njit(cache=True)
def delaunay_graph(data: nb.float64[:,:],
                   clusts: nb.types.List(nb.int64[:]),
                   batch_ids: nb.int64[:],
                   directed: bool = False) -> nb.int64[:,:]:
    """
    Function that returns an incidence matrix that connects nodes
    that share an edge in their corresponding Euclidean Delaunay graph.

    Args:
        data (np.ndarray)     : (N,4) [x, y, z, batchid]
        clusts ([np.ndarray]) : (C) List of arrays of voxel IDs in each cluster
        batch_ids (np.ndarray): (C) List of batch ids
        directed (bool)       : If directed, only keep edges [i,j] for which j>=i
    Returns:
        np.ndarray: (2,E) Tensor of edges
    """
    # For each batch, find the list of edges, append it
    ret = np.empty((0, 2), dtype=np.int64)
    for b in np.unique(batch_ids):
        clust_ids = np.where(batch_ids == b)[0]
        limits    = np.array([0]+[len(clusts[i]) for i in clust_ids]).cumsum()
        mask, labels = np.zeros(limits[-1], dtype=np.int64), np.zeros(limits[-1], dtype=np.int64)
        for i in range(len(clust_ids)):
            mask[limits[i]:limits[i+1]]   = clusts[clust_ids[i]]
            labels[limits[i]:limits[i+1]] = i*np.ones(len(clusts[clust_ids[i]]), dtype=np.int64)
        with nb.objmode(tri = 'int32[:,:]'): # Suboptimal. Ideally want to reimplement in Numba, but tall order...
            tri = Delaunay(data[mask][:, COORD_COLS], qhull_options='QJ').simplices # Joggled input guarantees simplical faces
        adj_mat = np.zeros((len(clust_ids),len(clust_ids)), dtype=np.bool_)
        for s in tri:
            for i in s:
                for j in s:
                    if labels[j] > labels[i]:
                        adj_mat[labels[i],labels[j]] = True
        edges = np.where(adj_mat)
        edges = np.vstack((clust_ids[edges[0]],clust_ids[edges[1]])).T
        ret   = np.vstack((ret, edges))

    # Add the reciprocal edges as to create an undirected graph, if requested
    if not directed:
        ret = np.vstack((ret, ret[:,::-1]))

    return ret.T


@nb.njit(cache=True)
def mst_graph(batch_ids: nb.int64[:],
              dist_mat: nb.float64[:,:],
              directed: bool = False) -> nb.int64[:,:]:
    """
    Function that returns an incidence matrix that connects nodes
    that share an edge in their corresponding Euclidean Minimum Spanning Tree (MST).

    Args:
        batch_ids (np.ndarray): (C) List of batch ids
        dist_mat (np.ndarray) : (C,C) Tensor of pair-wise cluster distances
        directed (bool)       : If directed, only keep edges [i,j] for which j>=i
    Returns:
        np.ndarray: (2,E) Tensor of edges
    """
    # For each batch, find the list of edges, append it
    ret = np.empty((0, 2), dtype=np.int64)
    for b in np.unique(batch_ids):
        clust_ids = np.where(batch_ids == b)[0]
        if len(clust_ids) > 1:
            submat = np.triu(nbl.submatrix(dist_mat, clust_ids, clust_ids))
            with nb.objmode(mst_mat = 'float32[:,:]'): # Suboptimal. Ideally want to reimplement in Numba, but tall order...
                mst_mat = minimum_spanning_tree(submat).toarray().astype(np.float32)
            edges = np.where(mst_mat > 0.)
            edges = np.vstack((clust_ids[edges[0]],clust_ids[edges[1]])).T
            ret   = np.vstack((ret, edges))

    # Add the reciprocal edges as to create an undirected graph, if requested
    if not directed:
        ret = np.vstack((ret, ret[:,::-1]))

    return ret.T


@nb.njit(cache=True)
def knn_graph(batch_ids: nb.int64[:],
              k: nb.int64,
              dist_mat: nb.float64[:,:],
              directed: bool = False) -> nb.int64[:,:]:
    """
    Function that returns an incidence matrix that connects nodes
    that are k nearest neighbors. Sorts the distance matrix.

    Args:
        batch_ids (np.ndarray): (C) List of batch ids
        k (int)               : Number of connected neighbors for each node
        dist_mat (np.ndarray) : (C,C) Tensor of pair-wise cluster distances
        directed (bool)       : If directed, only keep edges [i,j] for which j>=i
    Returns:
        np.ndarray: (2,E) Tensor of edges
    """
    # Use the available distance matrix to build a kNN graph
    ret = np.empty((0, 2), dtype=np.int64)
    for b in np.unique(batch_ids):
        clust_ids = np.where(batch_ids == b)[0]
        if len(clust_ids) > 1:
            subk = min(k+1, len(clust_ids))
            submat = nbl.submatrix(dist_mat, clust_ids, clust_ids)
            for i in range(len(submat)):
                idxs = np.argsort(submat[i])[1:subk]
                edges = np.empty((subk-1,2), dtype=np.int64)
                for j, idx in enumerate(np.sort(idxs)):
                    edges[j] = [clust_ids[i], clust_ids[idx]]
                if len(edges):
                    ret = np.vstack((ret, edges))

    # Add the reciprocal edges as to create an undirected graph, if requested
    if not directed:
        ret = np.vstack((ret, ret[:,::-1]))

    return ret.T


@nb.njit(cache=True)
def bipartite_graph(batch_ids: nb.int64[:],
                    primaries: nb.boolean[:],
                    directed: nb.boolean = True,
                    directed_to: str = 'secondary') -> nb.int64[:,:]:
    """
    Function that returns an incidence matrix of the bipartite graph
    between primary nodes and non-primary nodes.

    Args:
        batch_ids (np.ndarray): (C) List of batch ids
        primaries (np.ndarray): (C) Primary mask (True if primary)
        directed (bool)       : True if edges only exist in one direction
        directed_to (str)     : Whether to point the edges to the primaries or the secondaries
    Returns:
        np.ndarray: (2,E) Tensor of edges
    """
    # Create the incidence matrix
    ret = np.empty((0,2), dtype=np.int64)
    for i in np.where(primaries)[0]:
        for j in np.where(~primaries)[0]:
            if batch_ids[i] ==  batch_ids[j]:
                ret = np.vstack((ret, np.array([[i,j]])))

    # Handle directedness, by default graph is directed towards secondaries
    if directed:
        if directed_to == 'primary':
            ret = ret[:,::-1]
        elif directed_to != 'secondary':
            raise ValueError('Graph orientation not recognized')
    else:
        ret = np.vstack((ret, ret[:,::-1]))

    return ret.T


@nb.njit(cache=True)
def restrict_graph(edge_index: nb.int64[:,:],
                   dist_mat: nb.float64[:,:],
                   max_dist: nb.float64[:,:],
                   classes: nb.int64[:] = None) -> nb.int64[:,:]:
    """
    Function that restricts an incidence matrix of a graph
    to the edges below a certain length.

    If `classes` are specified, the maximum edge length must be provided
    for each possible combination of node classes.

    Args:
        edge_index (np.ndarray): (2,E) Tensor of edges
        dist_mat (np.ndarray)  : (C,C) Tensor of pair-wise cluster distances
        max_dist (np.ndarray)  : (N_c, N_c) Maximum edge length for each class type
        classes (np.ndarray)   : (C) List of class for each cluster in the graph
    Returns:
        np.ndarray: (2,E) Restricted tensor of edges
    """
    if classes is None:
        assert max_dist.shape[0] == max_dist.shape[1] == 1
        max_dist = max_dist[0][0]
        edge_dists = np.empty(edge_index.shape[1], dtype=dist_mat.dtype)
        for k in range(edge_index.shape[1]):
            i, j = edge_index[0,k], edge_index[1,k]
            edge_dists[k] = dist_mat[i, j]
        return edge_index[:, edge_dists < max_dist]
    else:
        edge_max_dists = np.empty(edge_index.shape[1], dtype=dist_mat.dtype)
        edge_dists = np.empty(edge_index.shape[1], dtype=dist_mat.dtype)
        for k in range(edge_index.shape[1]):
            i, j = edge_index[0,k], edge_index[1,k]
            edge_max_dists[k] = max_dist[classes[i], classes[j]]
            edge_dists[k] = dist_mat[i, j]
        return edge_index[:, edge_dists < edge_max_dists]


@numbafy(cast_args=['data'], list_args=['clusts'], keep_torch=True, ref_arg='data')
def get_cluster_edge_features(data, clusts, edge_index, closest_index=None):
    """
    Function that returns a tensor of edge features for each of the
    edges connecting clusters in the graph.

    Args:
        data (np.ndarray)         : (N,8) [x, y, z, batchid, value, id, groupid, shape]
        clusts ([np.ndarray])     : (C) List of arrays of voxel IDs in each cluster
        edge_index (np.ndarray)   : (2,E) Incidence matrix
        closest_index (np.ndarray): (E) Index of closest pair of voxels for each edge
    Returns:
        np.ndarray: (E,19) Tensor of edge features (point1, point2, displacement, distance, orientation)
    """
    return _get_cluster_edge_features(data, clusts, edge_index, closest_index)
    #return _get_cluster_edge_features_vec(data, clusts, edge_index)

@nb.njit(parallel=True, cache=True)
def _get_cluster_edge_features(data: nb.float32[:,:],
                               clusts: nb.types.List(nb.int64[:]),
                               edge_index: nb.int64[:,:],
                               closest_index: nb.int64[:] = None) -> nb.float32[:,:]:

    feats = np.empty((len(edge_index), 19), dtype=data.dtype)
    for k in nb.prange(len(edge_index)):
        # Get the voxels in the clusters connected by the edge
        c1, c2 = edge_index[k]
        x1 = data[clusts[c1]][:, COORD_COLS]
        x2 = data[clusts[c2]][:, COORD_COLS]

        # Find the closest set point in each cluster
        imin = np.argmin(nbl.cdist(x1, x2)) if closest_index is None else closest_index[k]
        i1, i2 = imin//len(x2), imin%len(x2)
        v1 = x1[i1,:]
        v2 = x2[i2,:]

        # Displacement
        disp = v1 - v2

        # Distance
        lend = np.linalg.norm(disp)
        if lend > 0:
            disp = disp / lend

        # Outer product
        B = np.outer(disp, disp).flatten()

        feats[k] = np.concatenate((v1, v2, disp, np.array([lend]), B))

    return feats


@nb.njit(cache=True)
def _get_cluster_edge_features_vec(data: nb.float32[:,:],
                                   clusts: nb.types.List(nb.int64[:]),
                                   edge_index: nb.int64[:,:]) -> nb.float32[:,:]:

    # Get the closest points of approach IDs for each edge
    lend, idxs1, idxs2 = _get_edge_distances(data[:,COORD_COLS], clusts, edge_index)

    # Get the points that correspond to the first voxels
    v1 = data[idxs1][:, COORD_COLS]

    # Get the points that correspond to the second voxels
    v2 = data[idxs2][:, COORD_COLS]

    # Get the displacement
    disp = v1 - v2

    # Reshape the distance vector to a column vector
    lend = lend.reshape(-1,1)

    # Normalize the displacement vector
    disp = disp/(lend + (lend == 0))

    # Compute the outer product of the displacement
    B = np.empty((len(disp), 9), dtype=data.dtype)
    for k in range(len(disp)):
        B[k] = np.outer(disp, disp).flatten()
    #B = np.dot(disp.reshape(len(disp),-1,1), disp.reshape(len(disp),1,-1)).reshape(len(disp),-1)

    return np.hstack((v1, v2, disp, lend, B))


@numbafy(cast_args=['data'], keep_torch=True, ref_arg='data')
def get_voxel_edge_features(data, edge_index):
    """
    Function that returns a tensor of edge features for each of the
    edges connecting voxels in the graph.

    Args:
        data (np.ndarray)      : (N,8) [x, y, z, batchid, value, id, groupid, shape]
        edge_index (np.ndarray): (2,E) Incidence matrix
    Returns:
        np.ndarray: (E,19) Tensor of edge features (displacement, orientation)
    """
    return _get_voxel_edge_features(data, edge_index)


@nb.njit(parallel=True, cache=True)
def _get_voxel_edge_features(data: nb.float32[:,:],
                         edge_index: nb.int64[:,:]) -> nb.float32[:,:]:
    feats = np.empty((len(edge_index), 19), dtype=data.dtype)
    for k in nb.prange(len(edge_index)):
        # Get the voxel coordinates
        xi = data[edge_index[k,0]][:, COORD_COLS]
        xj = data[edge_index[k,1]][:, COORD_COLS]

        # Displacement
        disp = xj - xi

        # Distance
        lend = np.linalg.norm(disp)
        if lend > 0:
            disp = disp / lend

        # Outer product
        B = np.outer(disp, disp).flatten()

        feats[k] = np.concatenate([xi, xj, disp, np.array([lend]), B])

    return feats


@numbafy(cast_args=['voxels'], list_args=['clusts'])
def get_edge_distances(voxels, clusts, edge_index):
    """
    For each edge, finds the closest points of approach (CPAs) between the
    the two voxel clusters it connects, and the distance that separates them.

    Args:
        voxels (np.ndarray)    : (N,3) Tensor of voxel coordinates
        clusts ([np.ndarray])  : (C) List of arrays of voxel IDs in each cluster
        edge_index (np.ndarray): (E,2) Incidence matrix
    Returns:
        np.ndarray: (E) List of edge lengths
        np.ndarray: (E) List of voxel IDs corresponding to the first edge cluster CPA
        np.ndarray: (E) List of voxel IDs corresponding to the second edge cluster CPA
    """
    return _get_edge_distances(voxels, clusts, edge_index)

@nb.njit(parallel=True, cache=True)
def _get_edge_distances(voxels: nb.float32[:,:],
                        clusts: nb.types.List(nb.int64[:]),
                        edge_index:  nb.int64[:,:]) -> (nb.float32[:], nb.int64[:], nb.int64[:]):

    resi, resj = np.empty(len(edge_index), dtype=np.int64), np.empty(len(edge_index), dtype=np.int64)
    lend = np.empty(len(edge_index), dtype=np.float32)
    for k in nb.prange(len(edge_index)):
        i, j = edge_index[k]
        if i == j:
            ii = jj = 0
            lend[k] = 0.
        else:
            dist_mat = nbl.cdist(voxels[clusts[i]], voxels[clusts[j]])
            idx = np.argmin(dist_mat)
            ii, jj = idx//len(clusts[j]), idx%len(clusts[j])
            lend[k] = dist_mat[ii, jj]
        resi[k] = clusts[i][ii]
        resj[k] = clusts[j][jj]

    return lend, resi, resj


@numbafy(cast_args=['voxels'], list_args=['clusts'])
def inter_cluster_distance(voxels, clusts, batch_ids=None, mode='voxel', algorithm='brute', return_index=False):
    """
    Finds the inter-cluster distance between every pair of clusters within
    each batch, returned as a block-diagonal matrix.

    Args:
        voxels (torch.tensor) : (N,3) Tensor of voxel coordinates
        clusts ([np.ndarray]) : (C) List of arrays of voxel IDs in each cluster
        batch_ids (np.ndarray): (C) List of cluster batch IDs
        mode (str)            : Eiher use closest voxel distance (`voxel`) or centroid distance (`centroid`)
        algorithm (str)       : `brute` is exact but slow, `recursive` uses a fast but approximate proxy
        return_index (bool)   : If True, returns the combined index of the closest voxel pair
    Returns:
        torch.tensor: (C,C) Tensor of pair-wise cluster distances
    """
    # If there is no batch_ids provided, assign 0 to all clusters
    if batch_ids is None:
        batch_ids = np.zeros(len(clusts), dtype=np.int64) 

    if not return_index:
        return _inter_cluster_distance(voxels, clusts, batch_ids, mode, algorithm)
    else:
        assert mode == 'voxel', 'Cannot return index for centroid method'
        return _inter_cluster_distance_index(voxels, clusts, batch_ids, algorithm)

@nb.njit(parallel=True, cache=True)
def _inter_cluster_distance(voxels: nb.float32[:,:],
                            clusts: nb.types.List(nb.int64[:]),
                            batch_ids: nb.int64[:],
                            mode: str = 'voxel',
                            algorithm: str = 'brute') -> nb.float32[:,:]:

    assert len(clusts) == len(batch_ids)
    dist_mat = np.zeros((len(batch_ids), len(batch_ids)), dtype=voxels.dtype)
    indxi, indxj = complete_graph(batch_ids, directed=True)
    if mode == 'voxel':
        for k in nb.prange(len(indxi)):
            i, j = indxi[k], indxj[k]
            dist_mat[i,j] = dist_mat[j,i] = nbl.closest_pair(voxels[clusts[i]], voxels[clusts[j]], algorithm)[-1]
    elif mode == 'centroid':
        centroids = np.empty((len(batch_ids), voxels.shape[1]), dtype=voxels.dtype)
        for i in nb.prange(len(batch_ids)):
            centroids[i] = nbl.mean(voxels[clusts[i]], axis=0)
        for k in nb.prange(len(indxi)):
            i, j = indxi[k], indxj[k]
            dist_mat[i,j] = dist_mat[j,i] = np.sqrt(np.sum((centroids[j]-centroids[i])**2))
    else:
        raise ValueError('Inter-cluster distance mode not supported')

    return dist_mat


@nb.njit(parallel=True, cache=True)
def _inter_cluster_distance_index(voxels: nb.float32[:,:],
                                  clusts: nb.types.List(nb.int64[:]),
                                  batch_ids: nb.int64[:],
                                  algorithm: str = 'brute') -> (nb.float32[:,:], nb.int64[:,:]):

    assert len(clusts) == len(batch_ids)
    dist_mat = np.zeros((len(batch_ids), len(batch_ids)), dtype=voxels.dtype)
    closest_index = np.empty((len(batch_ids), len(batch_ids)), dtype=nb.int64)
    for i in range(len(clusts)):
        closest_index[i,i] = i
    indxi, indxj = complete_graph(batch_ids, directed=True)
    for k in nb.prange(len(indxi)):
        i, j = indxi[k], indxj[k]
        ii, jj, dist = nbl.closest_pair(voxels[clusts[i]], voxels[clusts[j]], algorithm)
        index = ii*len(clusts[j]) + jj

        closest_index[i,j] = closest_index[j,i] = index
        dist_mat[i,j] = dist_mat[j,i] = dist

    return dist_mat, closest_index


@numbafy(cast_args=['graph'])
def get_fragment_edges(graph, clust_ids):
    """
    Function that converts a set of edges between cluster ids
    to a set of edges between fragment ids (ordering in list)

    Args:
        graph (np.ndarray)    : (E,2) Tensor of [clust_id_1, clust_id_2]
        clust_ids (np.ndarray): (C) List of fragment cluster ids
        batch_ids (np.ndarray): (C) List of fragment batch ids
    Returns:
        np.ndarray: (E,2) Tensor of true edges [frag_id_1, frag_id2]
    """
    return _get_fragment_edges(graph, clust_ids)

@nb.njit(cache=True)
def _get_fragment_edges(graph: nb.int64[:,:],
                        clust_ids: nb.int64[:]) -> nb.int64[:,:]:
    # Loop over the graph edges, find the fragment ids, append
    true_edges = np.empty((0,2), dtype=np.int64)
    for e in graph:
        n1 = np.where(clust_ids == e[0])[0]
        n2 = np.where(clust_ids == e[1])[0]
        if len(n1) and len(n2):
            true_edges = np.vstack((true_edges, np.array([[n1[0], n2[0]]], dtype=np.int64)))

    return true_edges
