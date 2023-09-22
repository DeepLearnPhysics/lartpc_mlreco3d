import numpy as np
import torch

from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from torch_cluster import knn

from .globals import *


def compute_rescaled_charge(input_data,
        deghost_mask, last_index, collection_only=False):
    """
    Computes rescaled charge after deghosting.

    Notes
    -----
    This function should work on Numpy arrays or Torch tensors.

    Parameters
    ----------
    input_data: Union[np.ndarray, torch.Tensor]
        (N, 4+N_f+6) Input tensor
    deghost_mask: Union[np.ndarray, torch.Tensor]
        (N) Ghost mask
    last_index: int
        Index where hit-related features start (4+N_f)
    collection_only : bool, default False
        Only use the collection plane to estimate the rescaled charge

    Returns
    -------
    Union[np.ndarray, torch.Tensor]
        (N_deghost) Rescaled charge array for input data
    """
    # Define operations on the basis of the input type
    if torch.is_tensor(input_data):
        unique = torch.unique
        empty = lambda shape: torch.empty(shape, dtype=torch.long,
                device=input_data.device)
        sum = lambda x: torch.sum(x, dim=1)
    else:
        unique = np.unique
        empty = np.empty
        sum = lambda x: np.sum(x, axis=1)

    # Count how many times each wire hit is used to form a space point
    hit_ids      = input_data[deghost_mask, last_index+3:last_index+6]
    multiplicity = empty(hit_ids.shape)
    for b in unique(input_data[:, BATCH_COL]):
        batch_mask = input_data[deghost_mask, BATCH_COL] == b
        _, inverse, counts = unique(hit_ids[batch_mask],
                return_inverse=True, return_counts=True)
        multiplicity[batch_mask] = counts[inverse].reshape(-1,3)

    # Rescale the charge on the basis of hit multiplicity
    hit_charges = input_data[deghost_mask, last_index  :last_index+3]
    if not collection_only:
        # Take the average of the charge estimates from each active plane
        pmask   = hit_ids > -1
        charges = sum((hit_charges*pmask)/multiplicity)/sum(pmask)
    else:
        # Only use the collection plane measurement
        charges = hit_charges[:,-1]/multiplicity[:,-1]

    return charges


def adapt_labels(cluster_label, segment_label, segmentation, deghost_mask=None,
        break_classes=[SHOWR_SHP,TRACK_SHP,MICHL_SHP,DELTA_SHP]):
    """
    Adapts the cluster labels to account for the predicted semantics.

    Points predicted as wrongly predicted get the cluster
    label of the closest touching cluster, if there is one. Points that are
    predicted as ghosts get "empty" (-1) cluster labels everywhere.

    Instances that have been broken up by the deghosting process get
    assigned distinct cluster labels for each effective fragment.

    Notes
    -----
    This function should work on Numpy arrays or Torch tensors.

    Uses GPU version from `torch_cluster.knn` to speed up
    the label adaptation computation.

    Parameters
    ----------
    cluster_label : Union[np.ndarray, torch.Tensor]
        (N, N_l) Cluster label tensor
    segment_label : List[Union[np.ndarray, torch.Tensor]]
        (M, 5) Segmentation label tensor
    segmentation : Union[np.ndarray, torch.Tensor]
        (N_deghost, N_c) Segmentation score prediction tensor
    deghost_mask : Union[np.ndarray, torch.Tensor], optional
        (M) Predicted deghost mask
    break_classes : List[int], default [SHOWR_SHP, TRACK_SHP, MICHL_SHP, DELTA_SHP]
        Classes to run DBSCAN on to break up

    Returns
    -------
    Union[np.ndarray, torch.Tensor]
        (N_deghost, N_l) Adapted cluster label tensor
    """
    # Define operations on the basis of the input type
    if torch.is_tensor(segment_label):
        where             = torch.where
        ones              = lambda shape: torch.ones(shape, dtype=segment_label.dtype, device=segment_label.device)
        unique            = lambda x: x.int().unique()
        argmax            = lambda x, dim: torch.argmax(x, dim=dim)
        concatenate       = lambda x, dim: torch.cat(x, dim=dim)
        compute_neighbor  = lambda X_true, X_pred: knn(X_true[:, COORD_COLS].float(), X_pred[:, COORD_COLS].float(), 1)[1]
        compute_distances = lambda X_true, X_pred: torch.amax(torch.abs(X_true[:, COORD_COLS] - X_pred[:, COORD_COLS]), dim=1)
        to_long           = lambda x: x.long()
        to_bool           = lambda x: x.bool()
    else:
        where             = np.where
        ones              = np.ones
        unique            = np.unique
        argmax            = lambda x, axis: np.argmax(x, axis=axis)
        concatenate       = lambda x, axis: np.concatenate(x, axis=axis)
        compute_neighbor  = lambda X_true, X_pred: cdist(X_pred[:, COORD_COLS], X_true[:, COORD_COLS]).argmin(axis=1)
        compute_distances = lambda X_true, X_pred: np.amax(np.abs(X_true[:, COORD_COLS] - X_pred[:, COORD_COLS]), axis=1)
        to_long           = lambda x: x.astype(np.int64)
        to_bool           = lambda x: x.astype(bool)

    # Build a tensor of predicted segmentation that includes ghost points
    coords = segment_label[:, :VALUE_COL]
    if deghost_mask is not None and len(deghost_mask) != len(segmentation):
        segment_pred = to_long(GHOST_SHP*ones(len(coords)))
        segment_pred[deghost_mask] = argmax(segmentation, 1)
    else:
        segment_pred = argmax(segmentation, 1)

    # Initialize the DBSCAN algorithm (finds connected groups)
    dbscan = DBSCAN(eps=1.1, min_samples=1, metric='chebyshev')

    # Loop over individual images in the batch
    new_cluster_label = []
    for batch_id in unique(coords[:, BATCH_COL]):
        # Restrict tensors to a specific batch_id
        batch_mask      = where(coords[:, BATCH_COL] == batch_id)[0]
        if not len(batch_mask):
            new_cluster_label.append(-1 * ones((0, clusts_label.shape[1])))
        if deghost_mask is not None:
            deghost_mask_b = deghost_mask[batch_mask]
            if not deghost_mask_b.sum():
                new_cluster_label.append(-1. * ones((0, cluster_label.shape[1])))
                continue

        coords_b        = coords[batch_mask]
        cluster_label_b = cluster_label[cluster_label[:, BATCH_COL] == batch_id]
        segment_label_b = segment_label[batch_mask, VALUE_COL]
        segment_pred_b  = segment_pred[batch_mask]

        # Prepare new labels
        new_label = -1. * ones((coords_b.shape[0], cluster_label_b.shape[1]))
        new_label[:, :VALUE_COL] = coords_b

        # Check if the segment labels and predictions are compatible.
        # If they are compatible, store the cluster labels as is
        true_deghost = segment_label_b < GHOST_SHP
        incompatible_segment = (segment_pred_b == TRACK_SHP) \
                ^ (segment_label_b == TRACK_SHP)
        new_label[true_deghost] = cluster_label_b
        new_label[true_deghost & incompatible_segment, VALUE_COL:] = -1.

        # Loop over semantic classes separately
        for s in unique(segment_pred_b):
            # Skip predicted ghosts
            if s == GHOST_SHP:
                continue

            # Restrict to points in this class that have incompatible segment
            # labels. If there are none, skip to the next class
            bad_index = where((segment_pred_b == s) \
                    & (~true_deghost | incompatible_segment))[0]
            if coords_b[bad_index].shape[0] == 0:
                continue

            # Find points in cluster_label that have compatible segment labels
            cluster_seg = cluster_label_b[:, SHAPE_COL]
            compatible_segment = cluster_seg == TRACK_SHP if s == TRACK_SHP \
                    else cluster_seg != TRACK_SHP
            X_true = cluster_label_b[compatible_segment]
            if X_true.shape[0] == 0:
                continue

            # Loop over the set of unlabeled predicted points
            X_pred = coords_b[bad_index]
            tagged_voxels_count = 1
            while tagged_voxels_count > 0 and X_pred.shape[0] > 0:
                # Find the nearest neighbor to each predicted point
                d = compute_neighbor(X_true, X_pred)

                # Compute Chebyshev distance between predicted and closest true.
                distances = compute_distances(X_true[d], X_pred)

                # Label unlabeled voxels that touch a compatible true voxel
                select_mask = distances <= 1
                tagged_voxels_count = select_mask.sum()
                if tagged_voxels_count > 0:
                    # Use the label of the touching true voxel
                    additional_cluster_label = \
                            concatenate([X_pred[select_mask],
                                X_true[d[select_mask], VALUE_COL:]], 1)
                    new_label[bad_index[select_mask]] = additional_cluster_label

                    # Update the mask to not include the new assigned points
                    bad_index = bad_index[~select_mask]

                    # The new true available points are the ones we just added.
                    # The new pred points are those not yet labeled
                    X_true = additional_cluster_label
                    X_pred = X_pred[~select_mask]

        # At this point, get rid of predicted ghosts.
        if deghost_mask is not None:
            new_label = new_label[deghost_mask_b]
            new_label[:, SHAPE_COL] = segment_pred_b[deghost_mask_b]
        else:
            new_label[:, SHAPE_COL] = segment_pred_b

        # Find the current largest cluster ID to avoid duplicates
        cluster_count = int(cluster_label_b[:, CLUST_COL].max()) + 1

        for break_class in break_classes:
            # Restrict to the set of labels associated with this class
            break_index = where(new_label[:, SHAPE_COL] == break_class)[0]
            restricted_label = new_label[break_index]

            # Loop over true cluster instances in the new label tensor, break
            for c in unique(restricted_label[:, CLUST_COL]):
                # Skip invalid cluster ID
                if c < 0:
                    continue

                # Restrict tensor to a specific cluster, get voxel coordinates
                cluster_index = where(restricted_label[:, CLUST_COL] == c)[0]
                coordinates = restricted_label[cluster_index][:, COORD_COLS]
                if torch.is_tensor(coordinates):
                    coordinates = coordinates.detach().cpu().numpy()

                # Run DBSCAN on the cluster, update labels
                break_labels = dbscan.fit(coordinates).labels_
                break_labels += cluster_count
                if torch.is_tensor(new_label):
                    break_labels = torch.tensor(break_labels,
                            dtype=new_label.dtype, device=new_label.device)
                new_label[break_index[cluster_index], CLUST_COL] = break_labels
                cluster_count = int(break_labels.max()) + 1

        # Append the new set of labels associated with this image
        new_cluster_label.append(new_label)

    # Stack the tensors obtained from each batch_id
    new_cluster_label = concatenate(new_cluster_label, 0)

    return new_cluster_label


def deghost_labels_and_predictions(data_blob, result):
    '''
    Given dictionaries <data_blob> and <result>, apply deghosting to
    uresnet predictions and labels for use in later reconstruction stages.

    Warning
    -------
    Modifies in place the input data and result dictionaries.

    Note
    ----
    Used in analysis tools (decorator).

    Parameters
    ----------
    data_blob: dict
    result: dict
    '''

    result['ghost_mask'] = [
            result['ghost'][i][:,0] > result['ghost'][i][:,1] \
            for i in range(len(result['ghost']))]

    if 'segment_label' in data_blob:
        data_blob['true_ghost_mask'] = [
            data_blob['segment_label'][i][:, -1] < 5 \
                for i in range(len(data_blob['segment_label']))]

    data_blob['input_data_pre_deghost'] = data_blob['input_data']

    if 'segment_label' in data_blob:
        data_blob['input_data_true_nonghost'] = [data_blob['input_data'][i][mask] \
            for i, mask in enumerate(data_blob['true_ghost_mask'])]

    data_blob['input_data'] = [data_blob['input_data'][i][mask] \
        for i, mask in enumerate(result['ghost_mask'])]

    if 'cluster_label' in data_blob \
        and data_blob['cluster_label'] is not None:
        # Save the clust_data before deghosting
        data_blob['cluster_label_true_nonghost'] = data_blob['cluster_label']
        data_blob['cluster_label'] = adapt_labels_numpy(
            result,
            data_blob['segment_label'],
            data_blob['cluster_label'])

    if 'seg_prediction' in result \
        and result['seg_prediction'] is not None:
        result['seg_prediction'] = [
            result['seg_prediction'][i][result['ghost_mask'][i]] \
                for i in range(len(result['seg_prediction']))]

    if 'segmentation' in result \
        and result['segmentation'] is not None:
        result['segmentation_true_nonghost'] = result['segmentation']
        result['segmentation'] = [
            result['segmentation'][i][result['ghost_mask'][i]] \
                for i in range(len(result['segmentation']))]

    if 'kinematics_label' in data_blob \
        and data_blob['kinematics_label'] is not None:
        data_blob['kinematics_label_true_nonghost'] = data_blob['kinematics_label']
        data_blob['kinematics_label'] = adapt_labels_numpy(
            result,
            data_blob['segment_label'],
            data_blob['kinematics_label'])

    # This needs to come last - in adapt_labels seg_label is the original one
    if 'segment_label' in data_blob \
        and data_blob['segment_label'] is not None:
        data_blob['segment_label_true_nonghost'] = data_blob['segment_label']
        data_blob['segment_label'] = [
            data_blob['segment_label'][i][result['ghost_mask'][i]] \
                for i in range(len(data_blob['segment_label']))]
