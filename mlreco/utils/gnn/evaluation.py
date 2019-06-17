# utility to evaluate the network accuracy
import numpy as np
import torch

def assign_clusters(edge_index, edge_label, primaries, others, n):
    """
    assigns each node to a cluster represented by the primary node
    """
    clust = np.zeros(n)
    for i in primaries:
        clust[i] = i
    for i in others:
        inds = edge_index[1,:] == i
        if sum(inds) == 0:
            clust[i] = -1
            continue
        indmax = torch.argmax(edge_label[inds])
        clust[i] = edge_index[0,inds][indmax].item()
    return clust

def secondary_vox_matching_efficiency(edge_index, true_labels, pred_labels, primaries, clusters, n):
    """
    fraction of secondary voxels that are correctly assigned
    """
    mask = np.array([(i not in primaries) for i in range(n)])
    others = np.arange(n)[mask]
    true_nodes = assign_clusters(edge_index, true_labels, primaries, others, n)
    pred_nodes = assign_clusters(edge_index, pred_labels, primaries, others, n)
    tot_vox = np.sum([len(clusters[i]) for i in others])
    int_vox = np.sum([len(clusters[i]) for i in others if true_nodes[i] == pred_nodes[i]])
    return int_vox * 1.0 / tot_vox