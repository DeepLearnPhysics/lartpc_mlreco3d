# Defines GNN network accuracy metrics
import numpy as np
from mlreco.utils.metrics import SBD, AMI, ARI, purity_efficiency

def edge_assignment(edge_index, batches, groups, binary=False):
    """
    Function that determines which edges are turned on based
    on the group ids of the clusters they are connecting.

    Args:
        edge_index (np.ndarray): (2,E) Incidence matrix
        batches (np.ndarray)   : (C) List of batch ids
        groups (np.ndarray)    : (C) List of group ids
        binary (bool)          : True if the assigment must be adapted to binary loss
    Returns:
        np.ndarray: (E) Boolean array specifying on/off edges
    """
    # Set the edge as true if it connects two nodes that belong to the same batch and the same group
    edge_assn = np.array([(batches[e[0]] == batches[e[1]] and groups[e[0]] == groups[e[1]]) for e in edge_index.T], dtype=int)

    # If binary loss will be used, transform to -1,+1 instead of 0,1
    if binary:
        edge_assn = 2*edge_assn - 1
    return edge_assn


def edge_assignment_from_graph(edge_index, true_edge_index, binary=False):
    """
    Function that determines which edges are turned on based
    on the group ids of the clusters they are connecting.

    Args:
        edge_index (np.ndarray): (2,E) Constructed incidence matrix
        edge_index (np.ndarray): (2,E) True incidence matrix
    Returns:
        np.ndarray: (E) Boolean array specifying on/off edges
    """
    # Set the edge as true if it connects two nodes that belong to the same batch and the same group
    edge_assn = np.array([np.any([(e == pair).all() for pair in true_edge_index]) for e in edge_index.T], dtype=int)

    # If binary loss will be used, transform to -1,+1 instead of 0,1
    if binary:
        edge_assn = 2*edge_assn - 1
    return edge_assn


def cluster_to_voxel_label(clusts, node_label):
    """
    Function that turns an array of labels on clusters
    to an array of labels on voxels.

    Args:
        clusts ([np.ndarray])  : (C) List of arrays of voxel IDs in each cluster
        node_label (np.ndarray): (C) List of node labels
    Returns:
        np.ndarray: (N) List of voxel labels
    """
    nvoxels = np.sum([len(c) for c in clusts])
    vlabel = np.empty(nvoxels)
    stptr = 0
    for i, c in enumerate(clusts):
        endptr = stptr + len(c)
        vlabel[stptr:endptr] = node_label[i]
        stptr = endptr

    return vlabel


def find_parent(parent, i):
    """
    Function that recursivey finds the parent node id.

    Args:
        parent (np.ndarray): (C) List of current group ids for all the node
        i (int)              : Index of the node of which to find the parent
    Returns:
        int: Parent id
    """
    if i != parent[i]:
        parent[i] = find_parent(parent, parent[i])

    return parent[i]


def node_assignment(edge_index, edge_label, n):
    """
    Function that assigns each node to a group, based
    on the edge assigment provided.

    Args:
        edge_index (np.ndarray): (2,E) Incidence matrix
        edge_assn (np.ndarray) : (E) Boolean array (1 if edge is on)
        n (int)                  : Total number of clusters C
    Returns:
        np.ndarray: (C) List of group ids
    """
    # Loop over on edges, reset the group IDs of connected node
    group_ids = np.arange(n)
    on_edges = edge_index.T[np.where(edge_label)[0]]
    for a, b in on_edges:
        p1 = find_parent(group_ids, a)
        p2 = find_parent(group_ids, b)
        if p1 != p2:
            group_ids[p2] = p1

    return group_ids


def node_assignment_UF(edge_index, edge_wt, n, thresh=0.0):
    """
    Function that assigns each node to a group, based on the edge
    weights provided, by using the topologylayer implementation
    of union find.

    Args:
        edge_index (np.ndarray): (2,E) Incidence matrix
        edge_wt (np.ndarray)   : (E) Array of edge weights
        n (int)                : Total number of clusters C
        thresh (double)        : Threshold for edge association
    Returns:
        np.ndarray: (C) List of group ids
    """
    from topologylayer.functional.persistence import getClustsUF_raw

    edges = edge_index
    edges = edges.T # transpose
    edges = edges.flatten()

    val = edge_wt

    cs = getClustsUF_raw(edges, val, n, thresh)
    un, cinds = np.unique(cs, return_inverse=True)
    return cinds


def node_assignment_bipartite(edge_index, edge_label, primaries, n):
    """
    Function that assigns each node to a group represented
    by a primary node.

    Args:
        edge_index (np.ndarray): (2,E) Incidence matrix
        edge_label (np.ndarray): (E) Boolean array (1 if edge is on)
        primaries (np.ndarray) : (P) List of primary ids
        n (int)                : Total number of clusters C
    Returns:
        np.ndarray: (C) List of group ids
    """
    # Set the group ID to the cluster ID if primary
    group_ids = np.zeros(n)
    for i in primaries:
        clust[i] = i

    # Assign the secondary clusters to primaries
    others = [i for i in range(n) if i not in primaries]
    for i in others:
        inds = edge_index[1,:] == i
        if sum(inds) == 0:
            clust[i] = -1
            continue
        indmax = np.argmax(edge_label[inds])
        group_ids[i] = edge_index[0,inds][indmax].item()

    return group_ids


def node_assignment_group(group_ids, batch_ids):
    """
    Function that assigns each node to a group, given
    group ids at each batch and corresponding batch ids

    Args:
        group_ids (np.ndarray): (C) List of cluster group ids within each batch
        batch_ids (np.ndarray): (C) List of cluster batch ids
    Returns:
        np.ndarray: (C) List of unique group ids
    """
    # Loop over on edges, reset the group IDs of connected node
    joined = np.vstack((group_ids, batch_ids))
    _, unique_ids = np.unique(joined, axis=1, return_inverse=True)
    return unique_ids


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

