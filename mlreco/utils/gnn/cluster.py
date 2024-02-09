import numpy as np
import numba as nb
import torch
from typing import List

import mlreco.utils.numba_local as nbl
from mlreco.utils.decorators import numbafy
from mlreco.utils.globals import (BATCH_COL, COORD_COLS, VALUE_COL, CLUST_COL,
        PART_COL, GROUP_COL, MOM_COL, SHAPE_COL)


@numbafy(cast_args=['data'], list_args=['cluster_classes'], keep_torch=True, ref_arg='data')
def form_clusters(data, min_size=-1, column=CLUST_COL, cluster_classes=[-1]):
    """
    Function that returns a list of of arrays of voxel IDs
    that make up each of the clusters in the input tensor.

    Args:
        data (np.ndarray)      : (N,6-10) [x, y, z, batchid, value, id(, groupid, intid, nuid, shape)]
        min_size (int)         : Minimal cluster size
        column (int)           : Column in the tensor which contains cluster IDs
        cluster_classes ([int]): List of classes to include in the list of clusters
    Returns:
        [np.ndarray]: (C) List of arrays of voxel IDs in each cluster
    """
    return _form_clusters(data, min_size, column, cluster_classes)

@nb.njit(cache=True)
def _form_clusters(data: nb.float64[:,:],
                   min_size: nb.int64 = -1,
                   column: nb.int64 = CLUST_COL,
                   cluster_classes: nb.types.List(nb.int64) = nb.typed.List([-1])) -> nb.types.List(nb.int64[:]):

    # Create a mask which restricts the voxels to those with shape in cluster_classes
    restrict = False
    if cluster_classes[0] != -1:
        mask = np.zeros(len(data), dtype=np.bool_)
        for s in cluster_classes:
            mask |= (data[:, SHAPE_COL] == s)
        mask = np.where(mask)[0]
        restrict = True
    subdata = data[mask] if restrict else data

    # Loop over batches and cluster IDs, append cluster voxel lists
    clusts = []
    batch_ids = subdata[:, BATCH_COL]
    for b in np.unique(batch_ids):
        binds = np.where(batch_ids == b)[0]
        clust_ids = subdata[binds, column]
        if restrict:
            binds = mask[binds]
        for c in np.unique(clust_ids):
            if c < 0:
                continue
            clust = np.where(clust_ids == c)[0]
            if len(clust) < min_size:
                continue
            clusts.append(binds[clust])

    return clusts


@numbafy(cast_args=['data'], keep_torch=True, ref_arg='data')
def reform_clusters(data, clust_ids, batch_ids, column=CLUST_COL):
    """
    Function that returns a list of of arrays of voxel IDs
    that make up the requested clusters.

    Args:
        data (np.ndarray)     : (N,8) [x, y, z, batchid, value, id, groupid, shape]
        clust_ids (np.ndarray): (C) List of cluster ids
        batch_ids (np.ndarray): (C) List of batch ids
        column (int)          : Column in the tensor which contains cluster IDs
    Returns:
        [np.ndarray]: (C) List of arrays of voxel IDs in each cluster
    """
    return _reform_clusters(data, clust_ids, batch_ids, column)

@nb.njit(cache=True)
def _reform_clusters(data: nb.float64[:,:],
                     clust_ids: nb.int64[:],
                     batch_ids: nb.int64[:],
                     column: nb.int64 = CLUST_COL) -> nb.types.List(nb.int64[:]):
    clusts = []
    for i in range(len(batch_ids)):
        clusts.append(np.where((data[:, BATCH_COL] == batch_ids[i]) & (data[:, column] == clust_ids[i]))[0])

    return clusts


@numbafy(cast_args=['data'], list_args=['clusts'])
def get_cluster_batch(data, clusts):
    """
    Function that returns the batch ID of each cluster.
    This should be unique for each cluster, assert that it is.

    Args:
        data (np.ndarray)    : (N,8) [x, y, z, batchid, value, id, groupid, shape]
        clusts ([np.ndarray]): (C) List of arrays of voxel IDs in each cluster
    Returns:
        np.ndarray: (C) List of batch IDs
    """
    if not len(clusts):
        return np.empty(0, dtype=data.dtype)

    return _get_cluster_batch(data, clusts)

@nb.njit(cache=True)
def _get_cluster_batch(data: nb.float64[:,:],
                       clusts: nb.types.List(nb.int64[:])) -> nb.int64[:]:

    labels = np.empty(len(clusts), dtype=np.int64)
    for i, c in enumerate(clusts):
        assert len(np.unique(data[c, BATCH_COL])) == 1
        labels[i] = data[c[0], BATCH_COL]
    return labels


@numbafy(cast_args=['data'], list_args=['clusts'])
def get_cluster_label(data, clusts, column=CLUST_COL):
    """
    Function that returns the majority label of each cluster,
    as specified in the requested data column.

    Args:
        data (np.ndarray)    : (N,8) [x, y, z, batchid, value, id, groupid, shape]
        clusts ([np.ndarray]): (C) List of arrays of voxel IDs in each cluster
        column (int)         : Column which specifies the cluster label
    Returns:
        np.ndarray: (C) List of cluster labels
    """
    if not len(clusts):
        return np.empty(0, dtype=data.dtype)

    return _get_cluster_label(data, clusts, column)

@nb.njit(cache=True)
def _get_cluster_label(data: nb.float64[:,:],
                       clusts: nb.types.List(nb.int64[:]),
                       column: nb.int64 = CLUST_COL) -> nb.float64[:]:

    labels = np.empty(len(clusts), dtype=data.dtype)
    for i, c in enumerate(clusts):
        v, cts = nbl.unique(data[c, column])
        labels[i] = v[np.argmax(cts)]

    return labels


@numbafy(cast_args=['data'], list_args=['clusts'])
def get_cluster_primary_label(data, clusts, column, cluster_column=CLUST_COL, group_column=GROUP_COL):
    """
    Function that returns the majority label of the primary component
    of a cluster, as specified in the requested data column.

    The primary component is identified by picking the set of label
    voxels that have a cluster_id identical to the cluster group_id.

    Args:
        data (np.ndarray)    : (N,8) [x, y, z, batchid, value, id, groupid, shape]
        clusts ([np.ndarray]): (C) List of arrays of voxel IDs in each cluster
        column (int)         : Column which specifies the cluster label
        cluster_column (int) : Column which specifies the cluster ID
        group_column (int)   : Column which specifies the cluster group ID
    Returns:
        np.ndarray: (C) List of cluster primary labels
    """
    if not len(clusts):
        return np.empty(0, dtype=data.dtype)

    return _get_cluster_primary_label(data, clusts, column, cluster_column, group_column)

@nb.njit(cache=True)
def _get_cluster_primary_label(data: nb.float64[:,:],
                               clusts: nb.types.List(nb.int64[:]),
                               column: nb.int64,
                               cluster_column: nb.int64 = CLUST_COL,
                               group_column: nb.int64 = GROUP_COL) -> nb.float64[:]:
    labels = np.empty(len(clusts), dtype=data.dtype)
    group_ids = _get_cluster_label(data, clusts, group_column)
    for i in range(len(clusts)):
        cluster_ids  = data[clusts[i], cluster_column]
        primary_mask = cluster_ids == group_ids[i]
        if len(data[clusts[i][primary_mask]]):
            v, cts = nbl.unique(data[clusts[i][primary_mask], column])
        else: # If the primary is empty, use group
            v, cts = nbl.unique(data[clusts[i], column])
        labels[i] = v[np.argmax(cts)]

    return labels


@numbafy(cast_args=['data'], list_args=['clusts'], keep_torch=True, ref_arg='data')
def get_momenta_label(data, clusts):
    """
    Function that returns the momentum value of each cluster

    Args:
        data (np.ndarray)    : (N,12) [x, y, z, batchid, value, id, groupid, px, py, pz, p, pdg]
        clusts ([np.ndarray]): (C) List of arrays of voxel IDs in each cluster
    Returns:
        np.ndarray: (C) List of cluster IDs
    """
    if not len(clusts):
        return np.empty(0, dtype=data.dtype)

    return _get_momenta_label(data, clusts)

@nb.njit(cache=True)
def _get_momenta_label(data: nb.float64[:,:],
                       clusts: nb.types.List(nb.int64[:])) -> nb.float64[:]:
    labels = np.empty(len(clusts), dtype=data.dtype)
    for i, c in enumerate(clusts):
        labels[i] = np.mean(data[c, MOM_COL])

    return labels


@numbafy(cast_args=['data'], list_args=['clusts'], keep_torch=True, ref_arg='data')
def get_cluster_centers(data, clusts):
    """
    Function that returns the coordinate of the centroid
    associated with the listed clusters.

    Args:
        data (np.ndarray)    : (N,8) [x, y, z, batchid, value, id, groupid, shape]
        clusts ([np.ndarray]): (C) List of arrays of voxel IDs in each cluster
    Returns:
        np.ndarray: (C,3) tensor of cluster centers
    """
    if not len(clusts):
        return np.empty(0, dtype=data.dtype)

    return _get_cluster_centers(data, clusts)

@nb.njit(cache=True)
def _get_cluster_centers(data: nb.float64[:,:],
                         clusts: nb.types.List(nb.int64[:])) -> nb.float64[:,:]:
    centers = np.empty((len(clusts), 3), dtype=data.dtype)
    for i, c in enumerate(clusts):
        centers[i] = np.sum(data[c][:, COORD_COLS], axis=0)/len(c)

    return centers


@numbafy(cast_args=['data'], list_args=['clusts'])
def get_cluster_sizes(data, clusts):
    """
    Function that returns the sizes of
    each of the listed clusters.

    Args:
        data (np.ndarray)    : (N,5) [x, y, z, batchid, value]
        clusts ([np.ndarray]): (C) List of arrays of voxel IDs in each cluster
    Returns:
        np.ndarray: (C) List of cluster sizes
    """
    if not len(clusts):
        return np.empty(0, dtype=data.dtype)

    return _get_cluster_sizes(data, clusts)

@nb.njit(cache=True)
def _get_cluster_sizes(data: nb.float64[:,:],
                       clusts: nb.types.List(nb.int64[:])) -> nb.int64[:]:
    sizes = np.empty(len(clusts), dtype=np.int64)
    for i, c in enumerate(clusts):
        sizes[i] = len(c)

    return sizes


@numbafy(cast_args=['data'], list_args=['clusts'], keep_torch=True, ref_arg='data')
def get_cluster_energies(data, clusts):
    """
    Function that returns the energies deposited by
    each of the listed clusters.

    Args:
        data (np.ndarray)    : (N,5) [x, y, z, batchid, value]
        clusts ([np.ndarray]): (C) List of arrays of voxel IDs in each cluster
    Returns:
        np.ndarray: (C) List of cluster energies
    """
    if not len(clusts):
        return np.empty(0, dtype=data.dtype)

    return _get_cluster_energies(data, clusts)

@nb.njit(cache=True)
def _get_cluster_energies(data: nb.float64[:,:],
                          clusts: nb.types.List(nb.int64[:])) -> nb.float64[:]:
    energies = np.empty(len(clusts), dtype=data.dtype)
    for i, c in enumerate(clusts):
        energies[i] = np.sum(data[c, VALUE_COL])

    return energies


@numbafy(cast_args=['data'], list_args=['clusts'], keep_torch=True, ref_arg='data')
def get_cluster_features(data: nb.float64[:,:],
                         clusts: nb.types.List(nb.int64[:])) -> nb.float64[:,:]:
    """
    Function that returns an array of 16 geometric features for
    each of the clusters in the provided list.

    Args:
        data (np.ndarray)    : (N,3) Voxel coordinates [x, y, z]
        clusts ([np.ndarray]): (C) List of arrays of voxel IDs in each cluster
    Returns:
        np.ndarray: (C,16) tensor of cluster features (center, orientation, direction, size)
    """
    if not len(clusts):
        return np.empty((0, 16), dtype=data.dtype) # Cannot type empty list

    return _get_cluster_features(data, clusts)

@nb.njit(parallel=True, cache=True)
def _get_cluster_features(data: nb.float64[:,:],
                          clusts: nb.types.List(nb.int64[:])) -> nb.float64[:,:]:
    feats = np.empty((len(clusts), 16), dtype=data.dtype)
    ids = np.arange(len(clusts)).astype(np.int64) # prange creates a uint64 iterator which is cast to int64 to access a list,
                                                  # and throws a warning. To avoid this, use a separate counter to acces clusts.
    for k in nb.prange(len(clusts)):
        # Get list of voxels in the cluster
        clust = clusts[ids[k]]
        x = data[clust][:, COORD_COLS]

        # Get cluster center
        center = nbl.mean(x, 0)

        # Get orientation matrix
        A = np.cov(x.T, ddof = len(x) - 1).astype(x.dtype)

        # Center data
        x = x - center

        # Get eigenvectors, normalize orientation matrix and eigenvalues to largest
        # If points are superimposed, i.e. if the largest eigenvalue != 0, no need to keep going
        w, v = np.linalg.eigh(A)
        if w[2] == 0.:
            feats[k] = np.concatenate((center, np.zeros(12), np.array([len(clust)])))
            continue
        dirwt = 1.0 - w[1] / w[2]
        B = A / w[2]

        # Get the principal direction, identify the direction of the spread
        v0 = v[:,2]

        # Projection all points, x, along the principal axis
        x0 = np.dot(x, v0)

        # Evaluate the distance from the points to the principal axis
        xp0 = x - np.outer(x0, v0)
        np0 = np.empty(len(xp0), dtype=data.dtype)
        for i in range(len(xp0)):
            np0[i] = np.linalg.norm(xp0[i])

        # Flip the principal direction if it is not pointing towards the maximum spread
        sc = np.dot(x0, np0)
        if sc < 0:
            v0 = np.zeros(3, dtype=data.dtype)-v0 # (-)/negative doesn't work with numba for now...

        # Weight direction
        v0 = dirwt * v0

        # Append
        feats[k] = np.concatenate((center, B.flatten(), v0, np.array([len(clust)])))

    return feats


@numbafy(cast_args=['data'], list_args=['clusts'], keep_torch=True, ref_arg='data')
def get_cluster_features_extended(data, clusts, add_value=True, add_shape=True):
    """
    Function that returns the an array of 3 additional features for
    each of the clusters in the provided list.

    Args:
        data (np.ndarray)    : (N,X) Data tensor [x,y,z,batch_id,value,...,sem_type]
        clusts ([np.ndarray]): (C) List of arrays of voxel IDs in each cluster
        add_value (bool)     : Whether or not to add the pixel value mean/std to the features
        add_shape (bool)     : Whether or not to add the dominant semantic type to the features
    Returns:
        np.ndarray: (C,3) tensor of cluster features (mean value, std value, major sem_type)
    """
    assert add_value or add_shape
    if not len(clusts):
        return np.empty((0, add_value*2+add_shape), dtype=data.dtype)

    return _get_cluster_features_extended(data, clusts, add_value, add_shape)

@nb.njit(parallel=True, cache=True)
def _get_cluster_features_extended(data: nb.float64[:,:],
                                   clusts: nb.types.List(nb.int64[:]),
                                   add_value: bool = True,
                                   add_shape: bool = True) -> nb.float64[:,:]:
    feats = np.empty((len(clusts), add_value*2+add_shape), dtype=data.dtype)
    ids = np.arange(len(clusts)).astype(np.int64)
    for k in nb.prange(len(clusts)):
        # Get cluster
        clust = clusts[ids[k]]

        # Get mean and RMS energy in the cluster, if requested
        if add_value:
            mean_value = np.mean(data[clust, VALUE_COL])
            std_value = np.std(data[clust, VALUE_COL])
            feats[k, :2] = np.array([mean_value, std_value], dtype=data.dtype)

        # Get the cluster semantic class, if requested
        if add_shape:
            types, cnts = nbl.unique(data[clust, SHAPE_COL])
            major_sem_type = types[np.argmax(cnts)]
            feats[k, -1] = major_sem_type

    return feats


@numbafy(cast_args=['data','particles'], list_args=['clusts'], keep_torch=True, ref_arg='data')
def get_cluster_points_label(data, particles, clusts, random_order=True):
    """
    Function that gets label points for each cluster.
    Returns start point of primary shower fragment twice if shower, delta or Michel
    and end points of tracks if track.

    Args:
        data (torch.tensor)     : (N,X) Voxel coordinates [batch_id, x, y, z, ...]
        particles (torch.tensor): (N,9) Point coordinates [batch_id, start_x, start_y, start_z, last_x, last_y, last_z, start_t, shape_id]
                                (obtained with parse_particle_coords)
        clusts ([np.ndarray])   : (C) List of arrays of voxel IDs in each cluster
        random_order (bool)     : Whether or not to shuffle the start and end points randomly
    Returns:
        np.ndarray: (N,6) cluster-wise start and end points (in RANDOMIZED ORDER by default)
    """
    if not len(clusts):
        return np.empty((0, 6), dtype=data.dtype)

    return _get_cluster_points_label(data, particles, clusts, random_order)

@nb.njit(cache=True)
def _get_cluster_points_label(data: nb.float64[:,:],
                              particles: nb.float64[:,:],
                              clusts: nb.types.List(nb.int64[:]),
                              random_order: nb.boolean = True) -> nb.float64[:,:]:

    # Get start and end points (one and the same for all but track class)
    batch_ids = _get_cluster_batch(data, clusts)
    points = np.empty((len(clusts), 6), dtype=data.dtype)
    for b in np.unique(batch_ids):
        batch_particles = particles[particles[:, BATCH_COL] == b]
        for i in np.where(batch_ids == b)[0]:
            c = clusts[i]
            clust_ids = np.unique(data[c, PART_COL]).astype(np.int64)
            minid = np.argmin(batch_particles[clust_ids, -2])
            order = np.arange(6) if (np.random.choice(2) or not random_order) else np.array([3, 4, 5, 0, 1, 2])
            points[i] = batch_particles[clust_ids[minid]][order+1] # The first column is the batch ID

    # Bring the start points to the closest point in the corresponding cluster
    for i, c in enumerate(clusts):
        dist_mat = nbl.cdist(points[i].reshape(-1,3), data[c][:, COORD_COLS])
        argmins  = np.empty(len(dist_mat), dtype=np.int64)
        for j in range(len(dist_mat)):
            argmins[j] = np.argmin(dist_mat[j])
        points[i] = data[c][argmins][:, COORD_COLS].reshape(-1)

    return points


@numbafy(cast_args=['data'], list_args=['clusts'], keep_torch=True, ref_arg='data')
def get_cluster_start_points(data, clusts):
    """
    Function that estimates the start point of clusters
    based on their PCA and local curvature.

    Args:
        data (np.ndarray)    : (N,3) Voxel coordinates [x, y, z]
        clusts ([np.ndarray]): (C) List of arrays of voxel IDs in each cluster
    Returns:
        np.ndarray: (C,3) tensor of cluster start points
    """
    if not len(clusts):
        return np.empty((0, 3), dtype=data.dtype)

    return _get_cluster_start_points(data, clusts)

@nb.njit(parallel=True, cache=True)
def _get_cluster_start_points(data: nb.float64[:,:],
                              clusts: nb.types.List(nb.int64[:])) -> nb.float64[:,:]:
    points = np.empty((len(clusts), 3))
    for k in nb.prange(len(clusts)):
        vid = cluster_end_points(data[clusts[k]][:, COORD_COLS])[-1]

    return points


@numbafy(cast_args=['data','starts'], list_args=['clusts'], keep_torch=True, ref_arg='data')
def get_cluster_directions(data, starts, clusts, max_dist=-1, optimize=False):
    """
    Finds the orientation of all the clusters.

    Args:
        data (torch.tensor)  : (N,3) Voxel coordinates [x, y, z]
        starts (torch.tensor): (C,3) Coordinates of the start points
        clusts ([np.ndarray]): (C) List of arrays of voxel IDs in each cluster
        max_dist (float)     : Max distance between start voxel and other voxels
        optimize (bool)      : Optimizes the number of points involved in the estimate
    Returns:
        torch.tensor: (3) Orientation
    """
    if not len(clusts):
        return np.empty(starts.shape, dtype=data.dtype)

    return _get_cluster_directions(data, starts, clusts, max_dist, optimize)

@nb.njit(parallel=True, cache=True)
def _get_cluster_directions(data: nb.float64[:,:],
                            starts: nb.float64[:,:],
                            clusts: nb.types.List(nb.int64[:]),
                            max_dist: nb.float64 = -1,
                            optimize: nb.boolean = False) -> nb.float64[:,:]:

    dirs = np.empty(starts.shape, data.dtype)
    ids  = np.arange(len(clusts)).astype(np.int64)
    for k in nb.prange(len(clusts)):
        # Weird bug here: without the cast (astype), throws a strange noncontiguous error on reshape...
        dirs[k] = cluster_direction(data[clusts[ids[k]]], starts[k].astype(np.float64), max_dist, optimize)

    return dirs


@numbafy(cast_args=['data','values','starts'], list_args=['clusts'], keep_torch=True, ref_arg='data')
def get_cluster_dedxs(data, values, starts, clusts, max_dist=-1):
    """
    Finds the start dEdxs of all the clusters.

    Args:
        data (torch.tensor)  : (N,3) Voxel coordinates [x, y, z]
        values (torch.tensor): (N) Voxel values
        starts (torch.tensor): (C,3) Coordinates of the start points
        clusts ([np.ndarray]): (C) List of arrays of voxel IDs in each cluster
        max_dist (float)     : Max distance between start voxel and other voxels
    Returns:
        torch.tensor: (N) dEdx values for each cluster
    """
    if not len(clusts):
        return np.empty(0, dtype=data.dtype)

    return _get_cluster_dedxs(data, values, starts, clusts, max_dist)

@nb.njit(parallel=True, cache=True)
def _get_cluster_dedxs(data: nb.float64[:,:],
                       values: nb.float64[:],
                       starts: nb.float64[:,:],
                       clusts: nb.types.List(nb.int64[:]),
                       max_dist: nb.float64 = -1) -> nb.float64[:,:]:

    dedxs = np.empty(len(clusts), data.dtype)
    ids   = np.arange(len(clusts)).astype(np.int64)
    for k in nb.prange(len(clusts)):
        # Weird bug here: without the cast (astype), throws a strange noncontiguous error on reshape...
        dedxs[k] = cluster_dedx(data[clusts[ids[k]]], values[clusts[ids[k]]], starts[k].astype(np.float64), max_dist)

    return dedxs


# @nb.njit(cache=True)
def cluster_end_points(voxels: nb.float64[:,:]) -> (nb.float64[:], nb.float64[:]):
    """
    Finds the start point of a cluster by:
    1. Find the principal axis a of the point cloud
    2. Find the coordinate a_i of each point along this axis
    3. Find the points with minimum and maximum coordinate
    4. Find the point that has the largest umbrella curvature

    Args:
        voxels (np.ndarray): (N,3) Voxel coordinates [x, y, z]
    Returns:
        int: ID of the start voxel
    """
    # Get the axis of maximum spread
    axis = nbl.principal_components(voxels)[0]

    # Compute coord values along that axis
    coords = np.empty(len(voxels))
    for i in range(len(coords)):
        coords[i] = np.dot(voxels[i], axis)

    # Compute curvature of the extremities
    ids = [np.argmin(coords), np.argmax(coords)]

    # Sort the voxel IDs by increasing order of curvature order
    curvs = [umbrella_curv(voxels, ids[0]), umbrella_curv(voxels, ids[1])]
    curvs = np.array(curvs, dtype=np.int64)
    ids = np.array(ids, dtype=np.int64)
    ids[np.argsort(curvs)]

    # Return extrema
    return voxels[ids[0]], voxels[ids[1]]


@nb.njit(cache=True)
def cluster_direction(voxels: nb.float64[:,:],
                      start: nb.float64[:],
                      max_dist: nb.float64 = -1,
                      optimize: nb.boolean = False) -> nb.float64[:]:
    """
    Estimates the orientation of a cluster:
    - By default takes the normalized mean direction
      from the cluster start point to the cluster voxels
    - If max_dist is specified, restricts the cluster voxels
      to those within a max_dist radius from the start point
    - If optimize is True, selects the neighborhood which
      minimizes the transverse spread w.r.t. the direction

    Args:
        voxels (torch.tensor): (N,3) Voxel coordinates [x, y, z]
        start (torch.tensor) : (C,3) Coordinates of the start point
        max_dist (float)     : Max distance between start voxel and other voxels
        optimize (bool)      : Optimizes the number of points involved in the estimate
    Returns:
        torch.tensor: (3) Orientation
    """
    # If max_dist is set, limit the set of voxels to those within a sphere
    # of radius max_dist
    if max_dist > 0:
        dist_mat = nbl.cdist(start.reshape(1,-1), voxels).flatten()
        voxels = voxels[dist_mat <= max(max_dist, np.min(dist_mat))]

    # If optimize is set, select the radius by minimizing the transverse spread
    if optimize and len(voxels) > 2:
        # Order the cluster points by increasing distance to the start point
        dist_mat = nbl.cdist(start.reshape(1,-1), voxels).flatten()
        order = np.argsort(dist_mat)
        voxels = voxels[order]
        dist_mat = dist_mat[order]

        # Find the PCA relative secondary spread for each point
        labels = -np.ones(len(voxels), dtype=voxels.dtype)
        meank = nbl.mean(voxels[:3], 0)
        covk = (np.transpose(voxels[:3] - meank) @ (voxels[:3] - meank))/3
        for i in range(2, len(voxels)):
            # Get the eigenvalues, compute relative transverse spread
            w, _ = np.linalg.eigh(covk)
            labels[i] = np.sqrt(w[2] / (w[0] + w[1])) \
                    if (w[0] + w[1]) / w[2] > 1e-9 else 0.

            # If the value is the same as the previous, choose this one
            if dist_mat[i] == dist_mat[i-1]:
                labels[i-1] = -1.

            # Increment mean and matrix
            if i != len(voxels) - 1:
                meank = ((i + 1) * meank + voxels[i+1]) / (i + 2)
                covk = (i + 1) * covk / (i + 2) \
                        + (voxels[i+1] - meank).reshape(-1,1) \
                        * (voxels[i+1] - meank) / (i + 1)

        # Subselect voxels that are most track-like
        max_id = np.argmax(labels)
        voxels = voxels[:max_id+1]

    # If no voxels were selected, return dummy value
    if not len(voxels) or (len(voxels) == 1 and np.all(voxels[0] == start)):
        return np.array([1., 0., 0.], dtype=voxels.dtype)

    # Compute mean direction with respect to start point, normalize it
    rel_voxels = np.empty((len(voxels), 3), dtype=voxels.dtype)
    for i in range(len(voxels)):
        rel_voxels[i] = voxels[i] - start

    mean = nbl.mean(rel_voxels, 0)
    norm = np.sqrt(np.sum(mean**2))
    if norm:
        return mean/norm

    return mean


# @nb.njit(cache=True)
def umbrella_curv(voxels: nb.float64[:,:],
                  voxid: nb.int64) -> nb.float64:
    """
    Computes the umbrella curvature as in equation 9 of "Umbrella Curvature:
    A New Curvature Estimation Method for Point Clouds" by A.Foorginejad and K.Khalili
    (https://www.sciencedirect.com/science/article/pii/S2212017313006828)

    Args:
        voxels (np.ndarray): (N,3) Voxel coordinates [x, y, z]
        voxid  (int)       : Voxel ID in which to compute the curvature
    Returns:
        int: Value of the curvature in voxid with respect to the rest of the point cloud
    """
    # Find the mean direction from that point
    refvox = voxels[voxid]
    axis = np.mean([v-refvox for v in voxels], axis=0)
    axis /= np.linalg.norm(axis)

    # Find the umbrella curvature (mean angle from the mean direction)
    return abs(np.mean([np.dot((voxels[i]-refvox)/np.linalg.norm(voxels[i]-refvox), axis) for i in range(len(voxels)) if i != voxid]))


@nb.njit(cache=True)
def cluster_dedx(voxels: nb.float64[:,:],
                 values: nb.float64[:],
                 start: nb.float64[:],
                 max_dist: nb.float64 = 5) -> nb.float64[:]:
    """
    Estimates the initial dEdx of a cluster

    Args:
        voxels (torch.tensor): (N,4) Voxel coordinates [x, y, z]
        values (torch.tensor): (N) Voxel values
        starts (torch.tensor): (C,3) Coordinates of the start points
        max_dist (float)     : Max distance between start voxel and other voxels
    Returns:
        torch.tensor: (3) Orientation
    """
    # If max_dist is set, limit the set of voxels to those within a sphere of radius max_dist
    dist_mat = nbl.cdist(start.reshape(1,-1), voxels).flatten()
    if max_dist > 0:
        voxels = voxels[dist_mat <= max_dist]
        if len(voxels) < 2:
            return 0.
        values = values[dist_mat <= max_dist]
        dist_mat = dist_mat[dist_mat <= max_dist]

    # Compute the total energy in the neighborhood and the max distance, return ratio
    return np.sum(values)/np.max(dist_mat)
