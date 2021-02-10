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
    Returns:
        np.ndarray: (C,16) tensor of cluster features (center, orientation, direction, size)
    """
    return get_cluster_features(data, clusts, whether_adjust_direction)


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
    # Get the voxels in the clusters connected by the edge
    x1 = get_cluster_voxels(data, c1)
    x2 = get_cluster_voxels(data, c2)

    # Find the closest set point in each cluster
    from scipy.spatial.distance import cdist
    d12 = cdist(x1, x2)
    imin = np.argmin(d12)
    i1, i2 = np.unravel_index(imin, d12.shape)
    v1 = x1[i1,:] # closest point in c1
    v2 = x2[i2,:] # closest point in c2

    # Displacement
    disp = v1 - v2

    # Distance
    lend = np.linalg.norm(disp)
    if lend > 0:
        disp = disp / lend

    # Outer product
    B = np.outer(disp, disp).flatten()

    return np.concatenate([B, lend.reshape(1)])


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
    # Get the voxels in the clusters connected by the edge
    x1 = get_cluster_voxels(data, c1)
    x2 = get_cluster_voxels(data, c2)

    # Find the closest set point in each cluster
    from scipy.spatial.distance import cdist
    d12 = cdist(x1, x2)
    imin = np.argmin(d12)
    i1, i2 = np.unravel_index(imin, d12.shape)
    v1 = x1[i1,:] # closest point in c1
    v2 = x2[i2,:] # closest point in c2

    # Displacement
    disp = v1 - v2

    # Distance
    lend = np.linalg.norm(disp)
    if lend > 0:
        disp = disp / lend

    # Outer product
    B = np.outer(disp, disp).flatten()

    return np.concatenate([v1, v2, disp, lend.reshape(1), B])


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

    # Get the voxel coordinates
    xi = data[i,:3]
    xj = data[j,:3]

    # Displacement
    disp = xj - xi

    # Distance
    lend = np.linalg.norm(disp)
    if lend > 0:
        disp = disp / lend

    # Outer product
    B = np.outer(disp, disp).flatten()

    return np.concatenate([xi, xj, disp, lend.reshape(1), B])


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


def form_merging_batches(batch_ids, mean_merge_size):
    """
    Function that returns a list of updated batch_ids for merging.

    Args:
        batch_ids (np.ndarray) : (B) batch IDs in the batch
        mean_merge_size (int)  : Mean number of event to combine
    Returns:
        np.ndarray: (B) Merged batch IDs
    """
    # Get enough Poisson samples to cover the full batch size exactly
    batch_size = len(batch_ids)
    event_cnts = []
    while np.sum(event_cnts) < batch_size:
        cnt = np.random.poisson(mean_merge_size)
        if cnt > 0:
            event_cnts.append(cnt)
    if np.sum(event_cnts) > batch_size:
        event_cnts[-1] -= np.sum(event_cnts)-batch_size

    return np.concatenate([np.full(n,i) for i,n in enumerate(event_cnts)])


def merge_batch(data, particles, merge_size=2, whether_fluctuate=False, data_type='cluster'):
    """
    Merge events in same batch. For example, if batch size = 16 and merge_size = 2,
    output data has a batch size of 8 with each adjacent 2 batches in input data merged.

    Args:
        data (np.ndarray)       : (N,10) [x, y, z, batchid, value, id, groupid, intid, nuid,shape]
        particles (np.ndarray)  : (N,8) [start_x, start_y, start_z, batch_id, last_x, last_y, last_z, start_t]
        merge_size (int)        : How many batches to be merged if whether_fluctuate=False,
                                  otherwise sample the number of merged batches using Poisson with mean of merge_size
        whether_fluctuate (bool): Whether not using a constant merging size

    Returns:
        np.ndarray: (B) Relabeled tensor
    """
    # Get the batch IDs
    batch_ids = data[:,3].unique()

    # Get the list that dictates how to merge events
    batch_size = len(batch_ids)
    if whether_fluctuate:
        merging_batch_id_list = form_merging_batches(batch_ids, merge_size)
    else:
        event_cnts = np.full(int(batch_size/merge_size), merge_size)
        if np.sum(event_cnts) < batch_size:
            event_cnts = np.append(event_cnts, batch_size-np.sum(event_cnts))
        merging_batch_id_list = np.concatenate([np.full(n,i) for i,n in enumerate(event_cnts)])

    # Merge batches, relabel everything to prevent any repeated indices
    data = data
    particles = particles
    for i in np.unique(merging_batch_id_list):
        # Find the list of voxels that belong to the new batch
        merging_batch_ids = np.where(merging_batch_id_list == i)[0]
        data_selections = [data[:,3] == j for j in merging_batch_ids]
        part_selections = [particles[:,3] == j for j in merging_batch_ids]

        # Relabel the batch column to the new batch id
        batch_selection = torch.sum(torch.stack(data_selections), dim=0).type(torch.bool)
        data[batch_selection,3] = int(i)

        # Relabel the cluster and group IDs by offseting by the number of particles
        clust_offset, group_offset, int_offset, nu_offset = 0, 0, 0, 0
        for j, sel in enumerate(data_selections):
            if j:
                data[sel & (data[:,5] > -1),5] += clust_offset
                data[sel & (data[:,6] > -1),6] += group_offset
                data[sel & (data[:,7] > -1),7] += int_offset
                data[sel & (data[:,8] > -1),8] += nu_offset
            clust_offset += torch.sum(part_selections[j])
            group_offset += torch.max(data[sel,6])+1
            int_offset = torch.max(data[sel,7])+1
            nu_offset = torch.max(data[sel,8])+1

        # Relabel the particle batch column
        batch_selection = torch.sum(torch.stack(part_selections), dim=0).type(torch.bool)
        particles[batch_selection,3] = int(i)

    return data, particles, merging_batch_id_list
