import numpy as np

# pred, truth are 1D arrays of labels in the same order
def SBD(pred, truth):
    '''
    Compute the Symmetric Best Dice (SBD) Score for Instance Segmentation.
    '''
    pred_clusters, pred_counts = np.unique(pred, return_counts=True)
    truth_clusters, truth_counts = np.unique(truth, return_counts=True)
    
    bd1 = BD(pred, pred_clusters, pred_counts, truth, truth_clusters, truth_counts)
    bd2 = BD(truth, truth_clusters, truth_counts, pred, pred_clusters, pred_counts)
    sbd = np.minimum(bd1, bd2)

    return sbd

def BD(data_sum, clusters_sum, clusters_sum_counts, data_fixed, clusters_fixed, clusters_fixed_counts):
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