import numpy as np
import numba as nb

import mlreco.utils.numba_local as nbl
from mlreco.utils.metrics import SBD, AMI, ARI, purity_efficiency

int_array = nb.int64[:]


@nb.njit(cache=True)
def edge_assignment(edge_index: nb.int64[:,:],
                    groups: nb.int64[:]) -> nb.int64[:]:
    """
    Function that determines which edges are turned on based
    on the group ids of the clusters they are connecting.

    Args:
        edge_index (np.ndarray): (E,2) Incidence matrix
        groups (np.ndarray)    : (C) List of group ids
    Returns:
        np.ndarray: (E) Array specifying on/off edges
    """
    # Set the edge as true if it connects two nodes that belong to the same batch and the same group
    return groups[edge_index[:,0]] == groups[edge_index[:,1]]


@nb.njit(cache=True)
def edge_assignment_from_graph(edge_index: nb.int64[:,:],
                               true_edge_index: nb.int64[:,:]) -> nb.int64[:]:
    """
    Function that determines which edges are turned on based
    on the group ids of the clusters they are connecting.

    Args:
        edge_index (np.ndarray): (E,2) Constructed incidence matrix
        edge_index (np.ndarray): (E,2) True incidence matrix
    Returns:
        np.ndarray: (E) Array specifying on/off edges
    """
    # Set the edge as true if it connects two nodes that are connected by a true dependency
    edge_assn = np.empty(len(edge_index), dtype=np.int64)
    for k, e in enumerate(edge_index):
        edge_assn[k] = (e == true_edge_index).any()

    return edge_assn


@nb.njit(cache=True)
def union_find(edge_index: nb.int64[:,:],
               n: nb.int64) -> (nb.int64[:], nb.types.DictType(nb.int64, nb.int64[:])):
    """
    Implementation of the Union-Find algorithm.

    Args:
        edge_index (np.ndarray): (E,2) Edges in the graph
        n (int)                : Total number of clusters C
    Returns:
        np.ndarray: (C) Updated list of group ids
        np.ndarray: (C) Updated dictionary of partition groups
    """
    # Find the group_ids by merging groups when they are connected
    group_ids = np.arange(n, dtype=np.int64)
    for e in edge_index:
        if group_ids[e[0]] != group_ids[e[1]]:
            group_ids[group_ids == group_ids[e[1]]] = group_ids[e[0]]

    # Build group dictionary
    groups = nb.typed.Dict.empty(nb.int64, int_array)
    for g in np.unique(group_ids):
        groups[g] = np.where(group_ids == g)[0]

    return group_ids, groups


@nb.njit(cache=True)
def node_assignment(edge_index: nb.int64[:,:],
                    edge_label: nb.int64[:],
                    n: nb.int64) -> nb.int64[:]:
    """
    Function that assigns each node to a group, based
    on the edge assigment provided. This uses a local
    union find implementation.

    Args:
        edge_index (np.ndarray): (E,2) Incidence matrix
        edge_assn (np.ndarray) : (E) Boolean array (1 if edge is on)
        n (int)                : Total number of clusters C
    Returns:
        np.ndarray: (C) List of group ids
    """
    # Loop over on edges, reset the group IDs of connected node
    on_edges = edge_index[np.where(edge_label)[0]]
    return union_find(on_edges, n)[0]


@nb.njit(cache=True)
def node_assignment_bipartite(edge_index: nb.int64[:,:],
                              edge_label: nb.int64[:],
                              primaries: nb.int64[:],
                              n: nb.int64) -> nb.int64[:]:
    """
    Function that assigns each node to a group represented
    by a primary node. This function loops over secondaries and
    associates it to the primary with that is connected to it
    with the strongest edge.

    Args:
        edge_index (np.ndarray): (E,2) Incidence matrix
        edge_label (np.ndarray): (E) Array of edge scores
        primaries (np.ndarray) : (P) List of primary ids
        n (int)                : Total number of clusters C
    Returns:
        np.ndarray: (C) List of group ids
    """
    group_ids = np.arange(n, dtype=np.int64)
    others = [i for i in range(n) if i not in primaries]
    for i in others:
        inds = edge_index[:,1] == i
        if np.sum(inds) == 0:
            continue
        indmax = np.argmax(edge_label[inds])
        group_ids[i] = edge_index[inds,0][indmax]

    return group_ids


@nb.njit(cache=True)
def primary_assignment(node_scores: nb.float32[:,:],
                       group_ids: nb.int64[:] = None) -> nb.boolean[:]:
    """
    Function that select shower primary fragments based
    on the node-score (and optionally an a priori grouping).

    Args:
        node_scores (np.ndarray): (C,2) Node scores
        group_ids (array)       : (C) List of group ids
    Returns:
        np.ndarray: (C) Primary labels
    """
    if group_ids is None:
        return nbl.argmax(node_scores, axis=1).astype(np.bool_)

    primary_labels = np.zeros(len(node_scores), dtype=np.bool_)
    node_scores = nbl.softmax(node_scores, axis=1)
    for g in np.unique(group_ids):
        mask = np.where(group_ids == g)[0]
        idx  = np.argmax(node_scores[mask][:,1])
        primary_labels[mask[idx]] = True

    return primary_labels


@nb.njit(cache=True)
def adjacency_matrix(edge_index: nb.int64[:,:],
                     n: nb.int64) -> nb.boolean[:,:]:
    """
    Function that creates an adjacency matrix from a list
    of connected edges in a graph (densify adjacency).

    Args:
        edge_index (np.ndarray): (E,2) Incidence matrix
        n (int)                : Total number of clusters C
    Returns:
        np.ndarray: (C,C) Adjacency matrix
    """
    adj_mat = np.eye(n, dtype=np.bool_)
    for e in edge_index:
        adj_mat[e[0],e[1]] = True
    return adj_mat


@nb.njit(cache=True)
def grouping_loss(pred_mat: nb.float32[:],
                  target_mat: nb.boolean[:],
                  loss: str = 'ce') -> np.float32:
    """
    Function that defines the graph clustering score.
    Given a target adjacency matrix A and a predicted
    adjacency P, the score is evaluated the average CE,
    L1 or L2 distance between truth and prediction.

    Args:
        pred_mat (np.ndarray)  : (C*C) Predicted adjacency matrix scores (flattened)
        target_mat (np.ndarray): (C*C) Target adjacency matrix (flattened)
        loss (str)             : Loss used to compute the graph score
    Returns:
        int: Graph grouping loss
    """
    if loss == 'ce':
        return nbl.log_loss(target_mat, pred_mat)
    elif loss == 'l1':
        return np.mean(np.absolute(pred_mat-target_mat))
    elif loss == 'l2':
        return np.mean((pred_mat-target_mat)*(pred_mat-target_mat))
    else:
        raise ValueError('Loss type not recognized')


@nb.njit(cache=True)
def edge_assignment_score(edge_index: nb.int64[:,:],
                          edge_scores: nb.float32[:,:],
                          n: nb.int64) -> (nb.int64[:,:], nb.int64[:], nb.float32):
    """
    Function that finds the graph that produces the lowest
    grouping score iteratively adding the most likely edges,
    if they improve the the score (builds a spanning tree).

    Args:
        edge_index (np.ndarray) : (E,2) Incidence matrix
        edge_scores (np.ndarray): (E,2) Two-channel edge score
        n (int)                 : Total number of clusters C
    Returns:
        np.ndarray: (E',2) Optimal incidence matrix
        np.ndarray: (C) Optimal group ID for each node
        float     : Score for the optimal incidence matrix
    """
    # If there is no edge, do not bother
    if not len(edge_index):
        return np.empty((0,2), dtype=np.int64), np.arange(n, dtype=np.int64), 0.

    # Build an input adjacency matrix to constrain the edge selection to the input graph
    adj_mat = adjacency_matrix(edge_index, n)

    # Interpret the softmax score as a dense adjacency matrix probability
    edge_scores = nbl.softmax(edge_scores, axis=1)
    pred_mat    = np.eye(n, dtype=np.float32)
    for k, e in enumerate(edge_index):
        pred_mat[e[0],e[1]] = edge_scores[k,1]

    # Remove edges with a score < 0.5 and sort the remainder by increasing order of OFF score
    on_mask   = edge_scores[:,1] >= 0.5
    args      = np.argsort(edge_scores[on_mask,0])
    ord_index = edge_index[on_mask][args]

    # Now iteratively identify the best edges, until the total score cannot be improved any longer
    best_ids    = np.empty(0, dtype=np.int64)
    best_groups = np.arange(n, dtype=np.int64)
    best_loss   = grouping_loss(pred_mat.flatten(), np.eye(n, dtype=np.bool_).flatten())
    for k, e in enumerate(ord_index):
        # If the edge connect two nodes already in the same group, proceed
        group_a, group_b = best_groups[e[0]], best_groups[e[1]]
        if group_a == group_b:
            continue

        # Restrict the adjacency matrix and the predictions to the nodes in the two candidate groups
        node_mask = np.where((best_groups == group_a) | (best_groups == group_b))[0]
        sub_pred = nbl.submatrix(pred_mat, node_mask, node_mask).flatten()
        sub_adj  = nbl.submatrix(adj_mat, node_mask, node_mask).flatten()

        # Compute the current adjacency matrix between the two groups
        current_adj = (best_groups[node_mask] == best_groups[node_mask].reshape(-1,1)).flatten()

        # Join the two groups if it minimizes the loss
        current_loss  = grouping_loss(sub_pred, sub_adj*current_adj)
        combined_loss = grouping_loss(sub_pred, sub_adj)
        if combined_loss < current_loss:
            best_groups[best_groups == group_b] = group_a
            best_loss += combined_loss - current_loss
            best_ids = np.append(best_ids, k)

    # Build the edge index
    best_index = ord_index[best_ids]

    return best_index, best_groups, best_loss


@nb.njit(cache=True)
def node_assignment_score(edge_index: nb.int64[:,:],
                          edge_scores: nb.float32[:,:],
                          n: nb.int64) -> nb.int64[:]:
    """
    Function that finds the graph that produces the lowest
    grouping score by building a score MST and by
    iteratively removing edges that improve the score.

    Args:
        edge_index (np.ndarray) : (E,2) Incidence matrix
        edge_scores (np.ndarray): (E,2) Two-channel edge score
        n (int)                : Total number of clusters C
    Returns:
        np.ndarray: (E',2) Optimal incidence matrix
    """
    return edge_assignment_score(edge_index, edge_scores, n)[1]


#@nb.njit(cache=True)
def cluster_to_voxel_label(clusts: nb.types.List(nb.int64[:]),
                           node_label: nb.int64[:]) -> nb.int64[:]:
    """
    Function that turns a list of labels on clusters
    to an array of labels on voxels.

    Args:
        clusts ([np.ndarray])  : (C) List of arrays of voxel IDs in each cluster
        node_label (np.ndarray): (C) List of node labels
    Returns:
        np.ndarray: (N) List of voxel labels
    """
    nvoxels = np.sum([len(c) for c in clusts])
    vlabel = np.empty(nvoxels, dtype=np.int64)
    stptr = 0
    for i, c in enumerate(clusts):
        endptr = stptr + len(c)
        vlabel[stptr:endptr] = node_label[i]
        stptr = endptr

    return vlabel


@nb.njit(cache=True)
def node_purity_mask(group_ids: nb.int64[:],
                     primary_ids: nb.int64[:]) -> nb.boolean[:]:
    """
    Function which creates a mask that is False only for nodes
    which belong to a group with more than a single clear primary.

    Note: It is possible that the single true primary has been
    broken into several nodes. In that case, the primary is
    also ambiguous, skip. TODO: pick the most sensible primary
    in that case, too restrictive otherwise.

    Args:
        group_ids (np.ndarray)  : (C) Array of cluster group IDs
        primary_ids (np.ndarray): (C) Array of cluster primary IDs
    Returns:
        np.ndarray: (E) High purity node mask
    """
    purity_mask = np.zeros(len(group_ids), dtype=np.bool_)
    for g in np.unique(group_ids):
        group_mask = group_ids == g
        if np.sum(group_mask) > 1 and np.sum(primary_ids[group_mask] == 1) == 1:
            purity_mask[group_mask] = np.ones(np.sum(group_mask))

    return purity_mask


@nb.njit(cache=True)
def edge_purity_mask(edge_index: nb.int64[:,:],
                     part_ids: nb.int64[:],
                     group_ids: nb.int64[:],
                     primary_ids: nb.int64[:]) -> nb.boolean[:]:
    """
    Function which creates a mask that is False only for edges
    which connect two nodes that both belong to a common group
    without a single clear primary.

    Note: It is possible that the single true primary has been
    broken into several nodes.

    Args:
        edge_index (np.ndarray) : (E,2) Incidence matrix
        part_ids (np.ndarray)   : (C) Array of cluster particle IDs
        group_ids (np.ndarray)  : (C) Array of cluster group IDs
        primary_ids (np.ndarray): (C) Array of cluster primary IDs
    Returns:
        np.ndarray: (E) High purity edge mask
    """
    purity_mask = np.ones(len(edge_index), dtype=np.bool_)
    for g in np.unique(group_ids):
        group_mask = np.where(group_ids == g)[0]
        if np.sum(primary_ids[group_mask]) != 1 and len(np.unique(part_ids[group_mask][primary_ids[group_mask] == 1])) != 1:
            edge_mask = np.empty(len(edge_index), dtype=np.bool_)
            for k, e in enumerate(edge_index):
                edge_mask[k] = (e[0] == group_mask).any() & (e[1] == group_mask).any()
            purity_mask[edge_mask] = np.zeros(np.sum(edge_mask))

    return purity_mask


def clustering_metrics(clusts, node_assn, node_pred):
    """
    Function that assigns each node to a group, based
    on the edge assigment provided.

    Args:
        clusts ([np.ndarray]) : (C) List of arrays of voxel IDs in each cluster
        node_assn (np.ndarray): (C) List of true node group labels
        node_pred (np.ndarray): (C) List of predicted node group labels
    Returns:
        double: Adjusted Rand Index
        double: Adjusted Mutual Information
        double: Symmetric Best Dice
        double: Purity
        double: Efficiency
    """
    pred_vox = cluster_to_voxel_label(clusts, node_pred)
    true_vox = cluster_to_voxel_label(clusts, node_assn)
    ari = ARI(pred_vox, true_vox)
    ami = AMI(pred_vox, true_vox)
    sbd = SBD(pred_vox, true_vox)
    pur, eff = purity_efficiency(pred_vox, true_vox)
    return ari, ami, sbd, pur, eff


def voxel_efficiency_bipartite(clusts, node_assn, node_pred, primaries):
    """
    Function that evaluates the fraction of secondary
    voxels that are associated to the corresct primary.

    Args:
        clusts ([np.ndarray]) : (C) List of arrays of voxel IDs in each cluster
        node_assn (np.ndarray): (C) List of true node group labels
        node_pred (np.ndarray): (C) List of predicted node group labels
        primaries (np.ndarray): (P) List of primary ids
    Returns:
        double: Fraction of correctly assigned secondary voxels
    """
    others = [i for i in range(n) if i not in primaries]
    tot_vox = np.sum([len(clusts[i]) for i in others])
    int_vox = np.sum([len(clusts[i]) for i in others if node_pred[i] == node_assn[i]])
    return int_vox * 1.0 / tot_vox
