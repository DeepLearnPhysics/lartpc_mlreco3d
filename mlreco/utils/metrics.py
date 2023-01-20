"""
Various metrics used for evaluating clustering
"""

import numpy as np


def unique_with_batch(label, bid):
    """
    merge 1D arrays of label and bid into array of new labels for unique (label, bid) pairs

    Parameters
    ----------
    label : array_like
        input labels
    bid : array_like
        input batch ids

    Returns
    -------
    labels2 : ndarray
        new unique labels
    """
    label = np.array(label)
    bid = np.array(bid)
    lb = np.stack((label, bid))
    _, label2, cts = np.unique(lb, axis=1, return_inverse=True, return_counts=True)
    return label2, cts


def unique_label(label):
    """
    transform label array into new label array where labels are between 0 and nlabels
    """
    label = np.array(label)
    _, label2, cts = np.unique(label, return_inverse=True, return_counts=True)
    return label2, cts


def ARI(pred, truth, bid=None):
    """
    Compute the Adjusted Rand Index (ARI) score for two clusterings
    """
    from sklearn.metrics import adjusted_rand_score
    if bid:
        pred, = unique_with_batch(pred, bid)
        truth, = unique_with_batch(truth, bid)
    return adjusted_rand_score(pred, truth)


def AMI(pred, truth, bid=None):
    """
    Compute the Adjusted Mutual Information (AMI) score for two clusterings
    """
    from sklearn.metrics import adjusted_mutual_info_score
    if bid:
        pred, = unique_with_batch(pred, bid)
        truth, = unique_with_batch(truth, bid)
    return adjusted_mutual_info_score(pred, truth, average_method='arithmetic')


def BD(data_sum, clusters_sum, clusters_sum_counts, data_fixed, clusters_fixed, clusters_fixed_counts):
    """
    Helper function for SBD function.
    """
    bd = 0
    for i in range(len(clusters_sum)):
        c = clusters_sum[i]
        c_len = clusters_sum_counts[i]
        unique, counts = np.unique(data_fixed[np.where(data_sum == c)], return_counts=True)
        best_dice = 0
        for j in range(len(unique)):
            dice = 2 * counts[j] / (c_len + clusters_fixed_counts[np.searchsorted(clusters_fixed, unique[j])])
            if dice > best_dice:
                best_dice = dice
        bd += best_dice
    bd /= len(clusters_sum)
    return bd


# pred, truth are 1D arrays of labels in the same order
def SBD(pred, truth, bid=None):
    '''
    Compute the Symmetric Best Dice (SBD) Score for Instance Segmentation.
    '''
    if bid:
        pred, = unique_with_batch(pred, bid)
        truth, = unique_with_batch(truth, bid)
    pred_clusters, pred_counts = np.unique(pred, return_counts=True)
    truth_clusters, truth_counts = np.unique(truth, return_counts=True)

    bd1 = BD(pred, pred_clusters, pred_counts, truth, truth_clusters, truth_counts)
    bd2 = BD(truth, truth_clusters, truth_counts, pred, pred_clusters, pred_counts)
    sbd = np.minimum(bd1, bd2)

    return sbd


def contingency_table(a, b, na=None, nb=None):
    """
    build contingency table for a and b
    assume a and b have labels between 0 and na and 0 and nb respectively
    """
    if not na:
        na = np.max(a)
    if not nb:
        nb = np.max(b)
    table = np.zeros((na, nb), dtype=int)
    for i, j in zip(a,b):
        table[i,j] += 1
    return table


def purity(pred, truth, bid=None):
    """
    cluster purity:
    intersection(pred, truth)/pred
    number in [0,1] - 1 indicates everything in the cluster is in the same ground-truth cluster
    """
    if bid:
        pred, pcts = unique_with_batch(pred, bid)
        truth, tcts = unique_with_batch(truth, bid)
    else:
        pred, pcts = (pred)
        truth, tcts = unique_label(truth)
    table = contingency_table(pred, truth, len(pcts), len(tcts))
    purities = table.max(axis=1) / pcts
    return purities.mean()


def global_purity(pred, truth, bid=None):
    """
    cluster purity as defined in https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html:
    intersection(pred, truth)/pred
    number in [0,1] - 1 indicates everything in the cluster is in the same ground-truth cluster
    """
    if bid:
        pred, pcts = unique_with_batch(pred, bid)
        truth, tcts = unique_with_batch(truth, bid)
    else:
        pred, pcts = unique_label(pred)
        truth, tcts = unique_label(truth)
    table = contingency_table(pred, truth, len(pcts), len(tcts))
    return np.sum(table.max(axis=1))/len(pred)


def efficiency(pred, truth, bid=None):
    """
    cluster efficiency:
    intersection(pred, truth)/truth
    number in [0,1] - 1 indicates everything is found in cluster
    """
    if bid:
        pred, pcts = unique_with_batch(pred, bid)
        truth, tcts = unique_with_batch(truth, bid)
    else:
        pred, pcts = unique_label(pred)
        truth, tcts = unique_label(truth)
    table = contingency_table(pred, truth, len(pcts), len(tcts))
    efficiencies = table.max(axis=0) / tcts
    return efficiencies.mean()


def global_efficiency(pred, truth, bid=None):
    """
    cluster efficiency as defined in https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html:
    intersection(pred, truth)/truth
    number in [0,1] - 1 indicates everything is found in cluster
    """
    if bid:
        pred, pcts = unique_with_batch(pred, bid)
        truth, tcts = unique_with_batch(truth, bid)
    else:
        pred, pcts = unique_label(pred)
        truth, tcts = unique_label(truth)
    table = contingency_table(pred, truth, len(pcts), len(tcts))
    return np.sum(table.max(axis=0))/len(pred)


def purity_efficiency(pred, truth, bid=None, mean=True):
    """
    function that combines purity and efficiency calculation into one go
    """
    if bid:
        pred, pcts = unique_with_batch(pred, bid)
        truth, tcts = unique_with_batch(truth, bid)
    else:
        pred, pcts = unique_label(pred)
        truth, tcts = unique_label(truth)
    table = contingency_table(pred, truth, len(pcts), len(tcts))
    efficiencies = table.max(axis=0) / tcts
    purities = table.max(axis=1) / pcts
    if mean:
        return purities.mean(), efficiencies.mean()
    else:
        return purities, efficiencies
