import numpy as np
import torch
from mlreco.models.layers.dbscan import distances
from scipy.spatial.distance import cdist


def adapt_labels(result, label_seg, label_clustering, num_classes=5):
    """
    Returns new cluster labels that have the same size as the input w/ ghost points.
    Points predicted as nonghost but that are true ghosts get the cluster label of
    the closest cluster.
    Points that are true ghosts and predicted as ghosts get "emtpy" (-1) values.
    Return shape: (input w/ ghost points, label_clusters_features)
    """
    complete_label_clustering = []
    for i in range(len(label_seg)):
        coords = label_seg[i][:, :4]
        label_c = []
        #print(len(coords[:, -1].unique()))
        for batch_id in coords[:, -1].unique():
            batch_mask = coords[:, -1] == batch_id
            batch_coords = coords[batch_mask]
            #print(torch.unique(label_clustering[i][:, 3]), batch_id)
            batch_clustering = label_clustering[i][label_clustering[i][:, 3] == batch_id]
            nonghost_mask = (result['ghost'][i][batch_mask].argmax(dim=1) == 0)
            # Select voxels predicted as nonghost, but true ghosts
            mask = nonghost_mask & (label_seg[i][:, -1][batch_mask] == num_classes)
            # Assign them to closest cluster
            d = distances(batch_coords[mask, :3], batch_clustering[:, :3]).argmin(dim=1)
            additional_label_clustering = torch.cat([batch_coords[mask], batch_clustering[d, 4:]], dim=1)
            # Prepare new labels
            new_label_clustering = -1. * torch.ones((batch_coords.size(0), batch_clustering.size(1))).double()
            if torch.cuda.is_available():
                new_label_clustering = new_label_clustering.cuda()
            new_label_clustering[mask] = additional_label_clustering
            #print(new_label_clustering.size(), label_seg[i][batch_mask, -1].size(), batch_clustering.size())
            new_label_clustering[label_seg[i][batch_mask, -1] < num_classes] = batch_clustering
            #print(label_seg[i][batch_mask][[label_seg[i][batch_mask, -1] < num_classes]][:10, :3], batch_clustering[:10, :3])
            label_c.append(new_label_clustering[nonghost_mask])
        label_c = torch.cat(label_c, dim=0)
        complete_label_clustering.append(label_c)
    return complete_label_clustering


def adapt_labels_numpy(result, label_seg, label_clustering, num_classes=5):
    """
    Returns new cluster labels that have the same size as the input w/ ghost points.
    Points predicted as nonghost but that are true ghosts get the cluster label of
    the closest cluster.
    Points that are true ghosts and predicted as ghosts get "emtpy" (-1) values.
    Return shape: (input w/ ghost points, label_clusters_features)
    """
    complete_label_clustering = []
    for i in range(len(label_seg)):
        coords = label_seg[i][:, :4]
        label_c = []
        #print(len(coords[:, -1].unique()))
        for batch_id in np.unique(coords[:, -1]):
            batch_mask = coords[:, -1] == batch_id
            batch_coords = coords[batch_mask]
            batch_clustering = label_clustering[i][label_clustering[i][:, 3] == batch_id]
            nonghost_mask = (result['ghost'][i][batch_mask].argmax(axis=1) == 0)
            # Select voxels predicted as nonghost, but true ghosts
            mask = nonghost_mask & (label_seg[i][:, -1][batch_mask] == num_classes)
            # Assign them to closest cluster
            d = cdist(batch_coords[mask, :3], batch_clustering[:, :3]).argmin(axis=1)
            additional_label_clustering = np.concatenate([batch_coords[mask], batch_clustering[d, 4:]], axis=1)
            # Prepare new labels
            new_label_clustering = -1. * np.ones((batch_coords.shape[0], batch_clustering.shape[1]))
            new_label_clustering[mask] = additional_label_clustering
            #print(new_label_clustering.size(), label_seg[i][batch_mask, -1].size(), batch_clustering.size())
            new_label_clustering[label_seg[i][batch_mask, -1] < num_classes] = batch_clustering
            label_c.append(new_label_clustering[nonghost_mask])
        label_c = np.concatenate(label_c, axis=0)
        complete_label_clustering.append(label_c)
    return complete_label_clustering
