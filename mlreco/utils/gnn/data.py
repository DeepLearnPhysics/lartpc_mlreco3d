# Defines inputs to the GNN networks
import numpy as np
import torch
from mlreco.utils.gnn.cluster import get_cluster_voxels, get_cluster_features, get_cluster_features_extended, get_cluster_dirs
from .voxels import get_voxel_features

def cluster_vtx_features(data, clusts, delta=0.0, whether_adjust_direction=False):
    """
    Function that returns the an array of 16 features for
    each of the clusters in the provided list.

    Args:
        data (np.ndarray)    : (N,8) [x, y, z, batchid, value, id, groupid, shape]
        clusts ([np.ndarray]): (C) List of arrays of voxel IDs in each cluster
        delta (float)        : Orientation matrix regularization
    Returns:
        np.ndarray: (C,16) tensor of cluster features (center, orientation, direction, size)
    """
    return get_cluster_features(data, clusts, delta, whether_adjust_direction)


def cluster_vtx_features_extended(data_values, data_sem_types, clusts):
    return get_cluster_features_extended(data_values, data_sem_types, clusts)


def cluster_vtx_dirs(data, cs, delta=0.0):
    """
    Function that returns the direction of the listed clusters,
    expressed as its normalized covariance matrix.

    Args:
        data (np.ndarray)    : (N,8) [x, y, z, batchid, value, id, groupid, shape]
        clusts ([np.ndarray]): (C) List of arrays of voxel IDs in each cluster
        delta (float)          : Orientation matrix regularization
    Returns:
        np.ndarray: (C,9) tensor of cluster directions
    """
    return get_cluster_dirs(data, clusts, delta)


def cluster_edge_dir(data, c1, c2):
    """
    Function that returns the edge direction between for a
    given pair of connected clusters.

    Args:
        data (np.ndarray): (N,8) [x, y, z, batchid, value, id, groupid, shape]
        c1 (np.ndarray)  : (M1) Array of voxel IDs associated with the first cluster
        c2 (np.ndarray)  : (M2) Array of voxel IDs associated with the second cluster
    Returns:
        np.ndarray: (10) Array of edge direction (orientation, distance)
    """
    from scipy.spatial.distance import cdist
    x1 = get_cluster_voxels(data, c1)
    x2 = get_cluster_voxels(data, c2)
    d12 = cdist(x1, x2)
    imin = np.argmin(d12)
    i1, i2 = np.unravel_index(imin, d12.shape)
    v1 = x1[i1,:] # closest point in c1
    v2 = x2[i2,:] # closest point in c2
    disp = v1 - v2 # displacement
    lend = np.linalg.norm(disp) # length of displacement
    if lend > 0:
        disp = disp / lend
    B = np.outer(disp, disp).flatten()

    return np.concatenate([B, [lend]])


def cluster_edge_dirs(data, clusts, edge_index):
    """
    Function that returns a tensor of edge directions for each of the
    edges in the graph.

    Args:
        data (np.ndarray)      : (N,8) [x, y, z, batchid, value, id, groupid, shape]
        clusts ([np.ndarray])  : (C) List of arrays of voxel IDs in each cluster
        edge_index (np.ndarray): (2,E) Incidence matrix
    Returns:
        np.ndarray: (E,10) Tensor of edge directions (orientation, distance)
    """
    return np.vstack([cluster_edge_dir(data, clusts[e[0]], clusts[e[1]]) for e in edge_index.T])


def cluster_edge_feature(data, c1, c2):
    """
    Function that returns the edge features for a
    given pair of connected clusters.

    Args:
        data (np.ndarray): (N,8) [x, y, z, batchid, value, id, groupid, shape]
        c1 (np.ndarray)  : (M1) Array of voxel IDs associated with the first cluster
        c2 (np.ndarray)  : (M2) Array of voxel IDs associated with the second cluster
    Returns:
        np.ndarray: (19) Array of edge features (point1, point2, displacement, distance, orientation)
    """
    from scipy.spatial.distance import cdist
    x1 = get_cluster_voxels(data, c1)
    x2 = get_cluster_voxels(data, c2)
    d12 = cdist(x1, x2)
    imin = np.argmin(d12)
    i1, i2 = np.unravel_index(imin, d12.shape)
    v1 = x1[i1,:] # closest point in c1
    v2 = x2[i2,:] # closest point in c2
    disp = v1 - v2 # displacement
    lend = np.linalg.norm(disp) # length of displacement
    if lend > 0:
        disp = disp / lend
    B = np.outer(disp, disp).flatten()
    return np.concatenate([v1, v2, disp, [lend], B])


def cluster_edge_features(data, clusts, edge_index):
    """
    Function that returns a tensor of edge features for each of the
    edges connecting clusters in the graph.

    Args:
        data (np.ndarray)      : (N,8) [x, y, z, batchid, value, id, groupid, shape]
        clusts ([np.ndarray])  : (C) List of arrays of voxel IDs in each cluster
        edge_index (np.ndarray): (2,E) Incidence matrix
    Returns:
        np.ndarray: (E,19) Tensor of edge features (point1, point2, displacement, distance, orientation)
    """
    return np.vstack([cluster_edge_feature(data, clusts[e[0]], clusts[e[1]]) for e in edge_index.T])


def vtx_features(data, max_dist=5.0, delta=0.0):
    """
    Function that returns the an array of 16 features for
    each of the clusters in the provided list.

    Args:
        data (np.ndarray)    : (N,8) [x, y, z, batchid, value, id, groupid, shape]
        max_dist (float)     : Defines "local", max distance to look at
        delta (float)        : Orientation matrix regularization
    Returns:
        np.ndarray: (N,16) tensor of voxel features (coords, local orientation, local direction, local count)
    """
    return get_voxel_features(data, max_dist, delta)


def edge_feature(data, i, j):
    """
    Function that returns the edge features for a
    given pair of connected voxels.

    Args:
        data (np.ndarray): (N,8) [x, y, z, batchid, value, id, groupid, shape]
        i (int)            : Index of the first voxel
        j (int)            : Index of the second voxel
    Returns:
        np.ndarray: (19) Array of edge features (displacement, orientation)
    """
    xi = data[i,:3]
    xj = data[j,:3]
    disp = xj - xi
    lend = np.linalg.norm(disp)
    if lend > 0:
        disp = disp / lend
    B = np.outer(disp, disp).flatten()
    return np.concatenate([xi, xj, disp, [lend], B])


def edge_features(data, edge_index):
    """
    Function that returns a tensor of edge features for each of the
    edges connecting voxels in the graph.

    Args:
        data (np.ndarray)      : (N,8) [x, y, z, batchid, value, id, groupid, shape]
        edge_index (np.ndarray): (2,E) Incidence matrix
    Returns:
        np.ndarray: (E,19) Tensor of edge features (displacement, orientation)
    """
    return np.vstack([edge_feature(data, e[0], e[1]) for e in edge_index.T])

def regulate_to_data(data, particles):
    """
    Function for regulating particles ids as to data ids
    deleting any particles entries that has id that doesn't appear in data ids
    Inputs:
        - data: (tensor) (N,8)->[x,y,z,batchids,value,ids,group_ids,sem.types]
        - particles: (tensor) (M,8)->[start_x,start_y,start_z,end_x,end_y,end_z,batchids,ids]
        - data_id_index
        - particles_id_index
    Output:
        - regulated_particles: (tensor) (M',8)
    """
    # check if particles is tensor
    if type(particles)!=torch.Tensor:
        return particles
    # get the batch from data and particles
    data_batch_ids = data[:,3].unique().view(-1)
    part_batch_ids = particles[:,6].unique().view(-1)
    if not torch.equal(data_batch_ids, part_batch_ids):
        raise ValueError('Data and Particles have no identical batch ids!')
    # Loop over batches
    regulated_particles = []
    for batch_id in data_batch_ids:
        data_selection = data[:,3]==batch_id
        part_selection = particles[:,6]==batch_id
        batched_parts = particles[part_selection,:]
        # get ids from data_selection
        data_ids = data[data_selection,5].unique().view(-1)
        for p in batched_parts:
            if p[7] in data_ids:
                regulated_particles.append(p.view(-1, p.size()[0]))
    return torch.cat(
        regulated_particles
    )