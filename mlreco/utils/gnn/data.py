# Defines inputs to the GNN networks
import numpy as np
import numba as nb
import torch

from typing import Tuple

from mlreco.utils import local_cdist
from mlreco.utils.numba import numba_wrapper, unique_nb
from mlreco.utils.ppn import get_track_endpoints_geo

from .cluster import get_cluster_features, get_cluster_features_extended
from .network import get_cluster_edge_features, get_voxel_edge_features
from .voxels  import get_voxel_features

def cluster_features(data, clusts, extra=False, **kwargs):
    """
    Function that returns an array of 16/19 geometric features for
    each of the clusters in the provided list.

    Args:
        data (torch.Tensor)  : (N,3) Voxel coordinates [x, y, z]
        clusts ([np.ndarray]): (C) List of arrays of voxel IDs in each cluster
        extra (bool)         : Whether or not to include extended features
    Returns:
        np.ndarray: (C,16/19) tensor of cluster features (center, orientation, direction, size)
    """
    if extra:
        return torch.cat([get_cluster_features(data.float(), clusts, **kwargs),
                          get_cluster_features_extended(data.float(), clusts, **kwargs)], dim=1)
    return get_cluster_features(data.float(), clusts, **kwargs)


def cluster_edge_features(data, clusts, edge_index, **kwargs):
    """
    Function that returns a tensor of 19 geometric edge features for each of the
    edges connecting clusters in the graph.

    Args:
        data (torch.Tensor)    : (N,8) [x, y, z, batchid, value, id, groupid, shape]
        clusts ([np.ndarray])  : (C) List of arrays of voxel IDs in each cluster
        edge_index (np.ndarray): (E,2) Incidence matrix
    Returns:
        np.ndarray: (E,19) Tensor of edge features (point1, point2, displacement, distance, orientation)
    """
    return get_cluster_edge_features(data.float(), clusts, edge_index, **kwargs)


def voxel_features(data, max_dist=5.0):
    """
    Function that returns the an array of 16 features for
    each of the voxels in the provided tensor.

    Args:
        data (torch.Tensor)  : (N,8) [x, y, z, batchid, value, id, groupid, shape]
        max_dist (float)     : Defines "local", i.e. max distance to look at
    Returns:
        np.ndarray: (N,16) tensor of voxel features (coords, local orientation, local direction, local count)
    """
    return get_voxel_features(data.float(), max_dist)


def voxel_edge_features(data, edge_index):
    """
    Function that returns a tensor of edge features for each of the
    edges connecting voxels in the graph.

    Args:
        data (torch.Tensor)    : (N,8) [x, y, z, batchid, value, id, groupid, shape]
        edge_index (np.ndarray): (2,E) Incidence matrix
    Returns:
        np.ndarray: (E,19) Tensor of edge features (displacement, orientation)
    """
    return get_voxel_edge_features(data.float(), edge_index)


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


def merge_batch(data, particles, merge_size=2, whether_fluctuate=False, batch_col=0):
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
    batch_ids = data[:,batch_col].unique()

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
        data_selections = [data[:,batch_col] == j for j in merging_batch_ids]
        part_selections = [particles[:,batch_col] == j for j in merging_batch_ids]

        # Relabel the batch column to the new batch id
        batch_selection = torch.sum(torch.stack(data_selections), dim=0).type(torch.bool)
        data[batch_selection,batch_col] = int(i)

        # Relabel the cluster and group IDs by offseting by the number of particles
        clust_offset, group_offset, int_offset, nu_offset = 0, 0, 0, 0
        for j, sel in enumerate(data_selections):
            if j:
                data[sel & (data[:,5] > -1),5] += clust_offset
                data[sel & (data[:,6] > -1),6] += group_offset
                data[sel & (data[:,7] > -1),7] += int_offset
                data[sel & (data[:,8] >  0),8] += nu_offset
            clust_offset += torch.sum(part_selections[j])
            group_offset += torch.max(data[sel,6])+1
            int_offset = torch.max(data[sel,7])+1
            nu_offset = torch.max(data[sel,8])+1

        # Relabel the particle batch column
        batch_selection = torch.sum(torch.stack(part_selections), dim=0).type(torch.bool)
        particles[batch_selection,batch_col] = int(i)

    return data, particles, merging_batch_id_list


def _get_extra_gnn_features(fragments,
                           frag_seg,
                           classes,
                           input,
                           result,
                           use_ppn=False,
                           use_supp=False,
                           enhance=False,
                           allow_outside=False,
                           coords_col=(1, 4)):
    """
    Extracting extra features to feed into the GNN particle aggregators

    - PPN: Most likely PPN point for showers,
            end points for tracks (+ direction estimate)
    - Supplemental: Mean/RMS energy in the fragment + semantic class

    If the `enhance` parameter is `True`, tracks leverage PPN predictions
    to provide a more accurate estimate of the end points. This needs to be 
    avoided for track fragments, as PPN is not trained to find end points for them.
    If set to `False`, the two voxels furthest away from each other are picked.

    If the `allow_outside` parameter is `True`, the end point estimates
    are *not* brought back to the closest fragment voxel.

    Parameters
    ==========
    fragments: np.ndarray
    frag_seg: np.ndarray
    classes: list
    input: list
    result: dictionary
    use_ppn: bool
    use_supp: bool
    enhance: bool
    allow_outside: bool

    Returns
    =======
    mask: np.ndarray
        Boolean mask to select fragments belonging to one
        of the requested classes.
    kwargs: dictionary
        Keys can include `points` (if `use_ppn` is `True`)
        and `extra_feats` (if `use_supp` is True).
    """
    # Build a mask for the requested classes
    mask = np.zeros(len(frag_seg), dtype=bool)
    for c in classes:
        mask |= (frag_seg == c)
    mask = np.where(mask)[0]

    #print("INPUT = ", input)

    # If requested, extract PPN-related features
    kwargs = {}
    if use_ppn:
        ppn_points = torch.empty((0,6), device=input[0].device,
                                        dtype=torch.float)
        points_tensor = result['points'][0].detach()
        for i, f in enumerate(fragments[mask]):
            fragment_voxels = input[0][f][:,coords_col[0]:coords_col[1]]
            if frag_seg[mask][i] == 1:
                end_points = get_track_endpoints_geo(input[0], f, points_tensor if enhance else None)
            else:
                scores = torch.softmax(points_tensor[f, -2:], dim=1)[:,-1]
                # scores = torch.sigmoid(points_tensor[f, -1])
                dmask  = torch.nonzero((scores > 0.5) & (torch.max(
                    torch.abs(points_tensor[f,:3]), dim=1).values < 1.),
                    as_tuple=True)[0]
                argmax = dmask[torch.argmax(scores[dmask])] \
                            if len(dmask) else torch.argmax(scores)
                start  = fragment_voxels[argmax] + points_tensor[f][argmax,:3] + 0.5
                end_points = torch.cat([start, start])

            if not allow_outside and (frag_seg[mask][i] != 1 or (frag_seg[mask][i] == 1 and enhance)):
                dist_mat   = local_cdist(end_points.reshape(-1,3), fragment_voxels)
                argmins    = torch.argmin(dist_mat, dim=1)
                end_points = torch.cat([fragment_voxels[argmins[0]], fragment_voxels[argmins[1]]])

            ppn_points = torch.cat((ppn_points, end_points.reshape(1,-1)), dim=0)

        kwargs['points'] = ppn_points

    # If requested, add energy and semantic related features
    if use_supp:
        supp_feats = torch.empty((0,3), device=input[0].device,
                                        dtype=torch.float)
        for i, f in enumerate(fragments[mask]):
            values = torch.cat((input[0][f,4].mean().reshape(1),
                                input[0][f,4].std().reshape(1))).float()
            if torch.isnan(values[1]): # Handle size-1 particles
                values[1] = input[0][f,4] - input[0][f,4]
            sem_type = torch.tensor([frag_seg[mask][i]],
                                    dtype=torch.float,
                                    device=input[0].device)
            supp_feats = torch.cat((supp_feats,
                torch.cat([values, sem_type.reshape(1)]).reshape(1,-1)), dim=0)

        kwargs['extra_feats'] = supp_feats

    return mask, kwargs


@numba_wrapper(list_args=['clusts'])
def split_clusts(clusts, batch_ids, batches, counts):
    """
    Splits a batched list of clusters into individual
    lists of clusters, one per batch ID.

    Args:
        clusts ([np.ndarray]) : (C) List of arrays of global voxel IDs in each cluster
        batch_ids (np.ndarray): (C) List of cluster batch ids
        batches (np.ndarray)  : (B) List of batch ids in this batch
        counts (np.ndarray)   : (B) Number of voxels in each batch
    Returns:
        [[np.ndarray]]: (B) List of list of arrays of batchwise voxels IDs in each cluster
        [np.ndarray]  : (B) List of cluster IDs in each batch
    """
    clusts_split, cbids = _split_clusts(clusts, batch_ids, batches, counts)

    # Cast the list of clusters to np.array (object type)
    same_length = [np.all([len(c) == len(bclusts[0]) for c in bclusts]) for bclusts in clusts_split]
    return [np.array(clusts_split[b], dtype=object if not sl else np.int64) for b, sl in enumerate(same_length)], cbids

@nb.njit(cache=True)
def _split_clusts(clusts: nb.types.List(nb.int64[:]),
                  batch_ids: nb.int64[:],
                  batches: nb.int64[:],
                  counts: nb.int64[:]) -> Tuple[nb.types.List(nb.types.List(nb.int64[:])), nb.types.List(nb.int64[:])]:

    # Get the batchwise voxel IDs for all pixels in the clusters
    cvids = np.empty(np.sum(counts), dtype=np.int64)
    index = 0
    for n in counts:
        cvids[index:index+n] = np.arange(n, dtype=np.int64)
        index += n

    # For each batch ID, get the list of clusters that belong to it
    cbids = [np.where(batch_ids == b)[0] for b in batches]

    # Split the cluster list into a list of list, one per batch iD
    return [[cvids[clusts[k]] for k in cids] for cids in cbids], cbids


@nb.njit(cache=True)
def split_edge_index(edge_index: nb.int64[:,:],
                     batch_ids: nb.int64[:],
                     batches: nb.int64[:]) -> Tuple[nb.types.List(nb.int64[:,:]), nb.types.List(nb.int64[:])]:
    """
    Splits a batched list of edges into individual
    lists of edges, one per batch ID.

    Args:
        edge_index (np.ndarray): (E,2) List of edges
        batch_ids (np.ndarray) : (C) List of cluster batch ids
        batches (np.ndarray)   : (B) List of batch ids in this batch
    Returns:
        [np.ndarray]: (B) List of list of edges
        [np.ndarray]: (B) List of edge IDs in each batch
    """
    # If the input is empty, simply return defaults
    if not edge_index.shape[1]:
        return [np.empty((0,2), dtype=np.int64) for b in batches], [np.empty(0, dtype=np.int64) for b in batches]

    # For each batch ID, get the list of edges that belong to it
    ebids = [np.where(batch_ids[edge_index[0]] == b)[0] for b in batches]

    # For each batch ID, find the cluster IDs within that batch
    ecids = np.empty(len(batch_ids), dtype=np.int64)
    index = 0
    for n in unique_nb(batch_ids)[1]:
        ecids[index:index+n] = np.arange(n, dtype=np.int64)
        index += n

    # Split the edge index into a list of edge indices
    return [np.ascontiguousarray(np.vstack((ecids[edge_index[0][b]], ecids[edge_index[1][b]])).T) for b in ebids], ebids
