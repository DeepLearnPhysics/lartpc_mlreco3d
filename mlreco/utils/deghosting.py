import numpy as np
import torch
from mlreco.models.layers.common.dbscan import distances
from scipy.spatial.distance import cdist


def adapt_labels(result, label_seg, label_clustering,
                 num_classes=5,
                 batch_column=0,
                 coords_column_range=(1, 4),
                 true_mask=None):
    """
    Returns new cluster labels that have the same size as the input w/ ghost points.
    Points predicted as nonghost but that are true ghosts get the cluster label of
    the closest cluster.
    Points that are true ghosts and predicted as ghosts get "emtpy" (-1) values.
    Return shape: (input w/ ghost points, label_clusters_features)
    """
    complete_label_clustering = []
    c1, c2 = coords_column_range

    if true_mask is not None:
        assert true_mask.shape[0] == label_seg[0].shape[0]

    c3 = max(c2, batch_column+1)
    for i in range(len(label_seg)):
        coords = label_seg[i][:, :c3]
        label_c = []
        for batch_id in coords[:, batch_column].int().unique():
            batch_mask = coords[:, batch_column] == batch_id
            batch_coords = coords[batch_mask]
            batch_clustering = label_clustering[i][label_clustering[i][:, batch_column] == batch_id]
            if len(batch_clustering) == 0:
                continue

            # Prepare new labels
            new_label_clustering = -1. * torch.ones((batch_coords.size(0),
                                                     batch_clustering.size(1)))
            if torch.cuda.is_available():
                new_label_clustering = new_label_clustering.cuda()

            if true_mask is None:
                nonghost_mask = (result['ghost'][i][batch_mask].argmax(dim=1) == 0)
                # Select voxels predicted as nonghost, but true ghosts
                mask = nonghost_mask & (label_seg[i][:, -1][batch_mask] == num_classes)
                # Assign them to closest cluster
                #print(batch_coords.shape, batch_clustering.shape)
                d = distances(batch_coords[mask, c1:c2],
                            batch_clustering[:, c1:c2]).argmin(dim=1)
                additional_label_clustering = torch.cat([batch_coords[mask],
                                                        batch_clustering[d, c3:]], dim=1).float()
                new_label_clustering[mask] = additional_label_clustering
            else:
                nonghost_mask = true_mask[batch_mask]

            new_label_clustering[label_seg[i][batch_mask, -1] < num_classes] = batch_clustering.float()
            label_c.append(new_label_clustering[nonghost_mask])
        label_c = torch.cat(label_c, dim=0)
        complete_label_clustering.append(label_c)
    return complete_label_clustering


def adapt_labels_numpy(result, label_seg, label_clustering, num_classes=5, batch_col=0, coords_col=(1, 4)):
    """
    Returns new cluster labels that have the same size as the input w/ ghost points.
    Points predicted as nonghost but that are true ghosts get the cluster label of
    the closest cluster.
    Points that are true ghosts and predicted as ghosts get "emtpy" (-1) values.
    Return shape: (input w/ ghost points, label_clusters_features)
    """
    c1, c2 = coords_col
    complete_label_clustering = []
    for i in range(len(label_seg)):
        coords = label_seg[i][:, :4]
        label_c = []
        #print(len(coords[:, -1].unique()))
        for batch_id in np.unique(coords[:, batch_col]):
            batch_mask = coords[:, batch_col] == batch_id
            batch_coords = coords[batch_mask]
            batch_clustering = label_clustering[i][label_clustering[i][:, batch_col] == batch_id]
            if len(batch_clustering) == 0:
                continue
            nonghost_mask = (result['ghost'][i][batch_mask].argmax(axis=1) == 0)
            # Select voxels predicted as nonghost, but true ghosts
            mask = nonghost_mask & (label_seg[i][:, -1][batch_mask] == num_classes)
            # Assign them to closest cluster
            d = cdist(batch_coords[mask, c1:c2], batch_clustering[:, c1:c2]).argmin(axis=1)
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
