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
    _, label2 = np.unqiqe(lb, axis=1, return_inverse=True)
    return label2


def ARI(pred, truth, bid=None):
    """
    Compute the Adjusted Rand Index (ARI) score for two clusterings
    """
    from sklearn.metrics import adjusted_rand_score
    if bid:
        pred = unique_with_batch(pred, bid)
        truth = unique_with_batch(truth, bid)
    return adjusted_rand_score(pred, truth)


def AMI(pred, truth, bid=None):
    """
    Compute the Adjusted Mutual Information (AMI) score for two clusterings
    """
    from sklearn.metrics import adjusted_mutual_info_score
    if bid:
        pred = unique_with_batch(pred, bid)
        truth = unique_with_batch(truth, bid)
    return adjusted_mutual_info_score(pred, truth)


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
        pred = unique_with_batch(pred, bid)
        truth = unique_with_batch(truth, bid)
    pred_clusters, pred_counts = np.unique(pred, return_counts=True)
    truth_clusters, truth_counts = np.unique(truth, return_counts=True)
    
    bd1 = BD(pred, pred_clusters, pred_counts, truth, truth_clusters, truth_counts)
    bd2 = BD(truth, truth_clusters, truth_counts, pred, pred_clusters, pred_counts)
    sbd = np.minimum(bd1, bd2)

    return sbd


def purity(pred, truth, bid=None):
    """
    cluster purity
    """
    pass


def efficiency(pred, truth, bid=None):
    """
    cluster efficiency
    """
    pass

