# utility to evaluate the network accuracy
import numpy as np
import torch
from mlreco.utils.metrics import SBD, AMI, ARI, purity_efficiency


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


def assign_clusters_UF(edge_index, edge_wt, n, thresh=0.0):
    """
    assigns clusters using Union Find on edges
    """
    from topologylayer.functional.persistence import getClustsUF_raw
    
    edges = edge_index.detach().cpu().numpy()
    edges = edges.T # transpose
    edges = edges.flatten()
    
    val = edge_wt.detach().cpu().numpy()
    
    cs = getClustsUF_raw(edges, val, n, thresh)
    un, cinds = np.unique(cs, return_inverse=True)
    return cinds


def secondary_matching_vox_efficiency(edge_index, true_labels, pred_labels, primaries, clusters, n):
    """
    fraction of secondary voxels that are correctly assigned
    """
    # mask = np.array([(i not in primaries) for i in range(n)])
    # others = np.arange(n)[mask]
    others = np.array([i for i in range(n) if i not in primaries])
    true_nodes = assign_clusters(edge_index, true_labels, primaries, others, n)
    pred_nodes = assign_clusters(edge_index, pred_labels, primaries, others, n)
    tot_vox = np.sum([len(clusters[i]) for i in others])
    int_vox = np.sum([len(clusters[i]) for i in others if true_nodes[i] == pred_nodes[i]])
    return int_vox * 1.0 / tot_vox


def secondary_matching_vox_efficiency2(matched, group, primaries, clusters):
    """
    fraction of secondary voxels that are correctly assigned
    uses matched array
    """
    n = len(matched)
    others = np.array([i for i in range(n) if i not in primaries])
    others_matched = np.array([i for i in others if matched[i] > -1])
    tot_vox = np.sum([len(clusters[i]) for i in others])
    int_vox = np.sum([len(clusters[i]) for i in others_matched if  group[i] == group[matched[i]]])
    return int_vox * 1.0 / tot_vox


def secondary_matching_vox_efficiency3(edge_index, true_labels, pred_labels, primaries, clusters, n):
    """
    fraction of secondary voxels that are correctly assigned
    pred_labels is N x C
    """
    # mask = np.array([(i not in primaries) for i in range(n)])
    # others = np.arange(n)[mask]
    others = np.array([i for i in range(n) if i not in primaries])
    true_nodes = assign_clusters(edge_index, true_labels, primaries, others, n)
    pred_labels = torch.argmax(pred_labels, 1) # get argmax predicted
    pred_nodes = assign_clusters(edge_index, pred_labels, primaries, others, n)
    tot_vox = np.sum([len(clusters[i]) for i in others])
    int_vox = np.sum([len(clusters[i]) for i in others if true_nodes[i] == pred_nodes[i]])
    return int_vox * 1.0 / tot_vox


def primary_assign_vox_efficiency(true_nodes, pred_nodes, clusters):
    """
    fraction of secondary voxels that are correctly assigned
    """
    tot_vox = np.sum([len(c) for c in clusters])
    int_vox = np.sum([len(clusters[i]) for i in range(len(clusters)) if np.sign(true_nodes[i].detach().cpu().numpy()) == np.sign(pred_nodes[i].detach().cpu().numpy())])
    return int_vox * 1.0 / tot_vox


def cluster_to_voxel_label(label, clusters):
    """
    turn an array of labels on clusters to an array of labels on voxels
    """
    nvoxels = np.sum([len(c) for c in clusters])
    vlabel = np.empty(nvoxels, dtype=np.int)
    stptr = 0
    for i, c in enumerate(clusters):
        endptr = stptr + len(c)
        vlabel[stptr:endptr] = label[i]
        stptr = endptr
    return vlabel


def form_groups(edge_index, edge_pred, n):
    """. 
    Assign a group ID to each of the clusters. 
    """
    group_ids = np.arange(n)
    on_edges = edge_index.transpose(0, 1)[torch.nonzero(edge_pred)].reshape(-1, 2)
    for e in on_edges:
        group_ids[e[1]] = group_ids[e[0]]
    return group_ids


def DBSCAN_cluster_metrics(edge_index, true_labels, pred_labels, primaries, clusters, n):
    """
    return ARI, AMI, SBD, purity, efficiency
    of matching
    """
    others = np.array([i for i in range(n) if i not in primaries])
    true_nodes = assign_clusters(edge_index, true_labels, primaries, others, n)
    pred_labels = torch.argmax(pred_labels, 1) # get argmax predicted
    pred_nodes = assign_clusters(edge_index, pred_labels, primaries, others, n)
    pred_vox = cluster_to_voxel_label(pred_nodes, clusters)
    true_vox = cluster_to_voxel_label(true_nodes, clusters)
    ari = ARI(pred_vox, true_vox)
    ami = AMI(pred_vox, true_vox)
    sbd = SBD(pred_vox, true_vox)
    pur, eff = purity_efficiency(pred_vox, true_vox)
    return ari, ami, sbd, pur, eff


def DBSCAN_cluster_metrics2(matched, clusters, group):
    """
    return ARI, AMI, SBD, purity, efficiency
    of matching.  Use matched array
    """
    pred_vox = cluster_to_voxel_label(matched, clusters)
    true_vox = cluster_to_voxel_label(group, clusters)
    ari = ARI(pred_vox, true_vox)
    ami = AMI(pred_vox, true_vox)
    sbd = SBD(pred_vox, true_vox)
    pur, eff = purity_efficiency(pred_vox, true_vox)
    return ari, ami, sbd, pur, eff


def DBSCAN_cluster_metrics3(edge_index, edge_assn, edge_pred, clusters):
    """ 
    return ARI, AMI, SBD, purity, efficiency
    of matching. Use complete graph
    """
    pred_group_ids = form_groups(edge_index, edge_pred, len(clusters))
    pred_vox = cluster_to_voxel_label(pred_group_ids, clusters) 
    true_group_ids = form_groups(edge_index, edge_assn, len(clusters))
    true_vox = cluster_to_voxel_label(true_group_ids, clusters) 
    ari = ARI(pred_vox, true_vox)
    ami = AMI(pred_vox, true_vox)
    sbd = SBD(pred_vox, true_vox)
    pur, eff = purity_efficiency(pred_vox, true_vox)
    return ari, ami, sbd, pur, eff

