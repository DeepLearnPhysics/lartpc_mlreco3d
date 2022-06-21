import numpy as np
import torch
from mlreco.models.layers.common.dbscan import distances
from scipy.spatial.distance import cdist
from torch_cluster import knn


def compute_rescaled_charge(input_data, deghost_mask, last_index = 6, batch_col = 0):
    """
    Computes rescaled charge after deghosting

    Note
    ----
    This function should work on Numpy arrays or Torch tensors.

    Parameters
    ----------
    input_data: np.ndarray or torch.Tensor
        Shape (N, 4+num_features) where 4 corresponds to batch_id,x,y,z
    deghost_mask: np.ndarray or torch.Tensor
        Shape (N,), N_deghost is the predicted deghosted voxel count
    last_index: int, default 6
        Indexes where hit-related features start @ 4 + deghost_input_features
    batch_col: int, default 0

    Returns
    -------
    np.ndarray or torch.Tensor
        Shape (N_deghost,) contains rescaled charge array for input data.
        Includes deghosted mask already.
    """
    if torch.is_tensor(input_data):
        unique = torch.unique
        empty = lambda n: torch.empty(n, dtype=torch.long, device=hit_charges.device)
        sum = lambda x: torch.sum(x, dim=1)
    else:
        unique = np.unique
        empty = np.empty
        sum = lambda x: np.sum(x, axis=1)

    batches = unique(input_data[:, batch_col])
    hit_charges  = input_data[deghost_mask, last_index  :last_index+3]
    hit_ids      = input_data[deghost_mask, last_index+3:last_index+6]
    multiplicity = empty(hit_charges.shape, )
    for b in batches:
        batch_mask = input_data[deghost_mask, batch_col] == b
        _, inverse, counts = unique(hit_ids[batch_mask], return_inverse=True, return_counts=True)
        multiplicity[batch_mask] = counts[inverse].reshape(-1,3)
    pmask   = hit_ids > -1
    charges = sum((hit_charges*pmask)/multiplicity)/sum(pmask) # Take average estimate
    return charges


def adapt_labels_knn(result, label_seg, label_clustering,
                      num_classes=5,
                      batch_column=0,
                      coords_column_range=(1, 4),
                      true_mask=None,
                      use_numpy=False):
    """
    Returns new cluster labels that have the same size as the input w/ ghost points.
    Points predicted as nonghost but that are true ghosts get the cluster label of
    the closest cluster.
    Points that are true ghosts and predicted as ghosts get "empty" (-1) values.

    Note
    ----
    Uses GPU version from `torch_cluster.knn` to speed up
    the label adaptation computation.

    Parameters
    ----------
    result: dict
    label_seg: list of torch.Tensor
    label_clustering: list of torch.Tensor
    num_classes: int, default 5
        Semantic classes count.
    batch_column: int, default 0
    coords_column_range: tuple, default (1, 4)
    true_mask: torch.Tensor, default None
        True nonghost mask. If None, will use the intersection
        of predicted nonghost and true nonghost. This option is
        useful to do "cheat ghost predictions" (i.e. mimic deghosting
        predictions using true ghost mask, to run later stages
        of the chain independently of the deghosting stage).

    Returns
    -------
    np.ndarray
        shape: (input w/ ghost points, label_clusters_features)

    See Also
    --------
    adapt_labels, adapt_labels_numpy
    """
    complete_label_clustering = []
    c1, c2 = coords_column_range

    if use_numpy:
        unique            = np.unique
        ones              = np.ones
        argmax            = lambda x: np.argmax(x, axis=1)
        where             = np.where
        concatenate0      = lambda x: np.concatenate(x, axis=1)
        concatenate1      = lambda x: np.concatenate(x, axis=0)
        compute_neighbor  = lambda X_true, X_pred: cdist(X_pred[:, c1:c2], X_true[:, c1:c2]).argmin(axis=1)
        compute_distances = lambda X_true, X_pred: np.amax(np.abs(X_true[:, c1:c2] - X_pred[:, c1:c2]), axis=1)
        make_float        = lambda x : x
        get_shape         = lambda x, y: (x.shape[0], y.shape[1])
    else:
        unique            = lambda x: x.int().unique()
        ones              = torch.ones
        argmax            = lambda x: torch.argmax(x, dim=1)
        where             = torch.where
        concatenate0      = lambda x: torch.cat(x, dim=1).float()
        concatenate1      = lambda x: torch.cat(x, dim=0)
        compute_neighbor  = lambda X_true, X_pred: knn(X_true[:, c1:c2].float(), X_pred[:, c1:c2].float(), 1)[1]
        compute_distances = lambda X_true, X_pred: torch.amax(torch.abs(X_true[:, c1:c2] - X_pred[:, c1:c2]), dim=1)
        make_float        = lambda x: x.float()
        get_shape         = lambda x, y: (x.size(0), y.size(1))

    if true_mask is not None:
        assert true_mask.shape[0] == label_seg[0].shape[0]
    c3 = max(c2, batch_column+1)

    for i in range(len(label_seg)):
        coords = label_seg[i][:, :c3]
        label_c = []
        for batch_id in unique(coords[:, batch_column]):
            batch_mask = coords[:, batch_column] == batch_id
            batch_coords = coords[batch_mask]
            batch_clustering = label_clustering[i][label_clustering[i][:, batch_column] == batch_id]
            if len(batch_clustering) == 0:
                continue

            if true_mask is None:
                nonghost_mask = argmax(result['ghost'][i][batch_mask]) == 0
            else:
                nonghost_mask = true_mask[batch_mask]

            # Prepare new labels
            new_label_clustering = -1. * ones(get_shape(batch_coords, batch_clustering))
            if (not use_numpy) and torch.cuda.is_available():
                new_label_clustering = new_label_clustering.cuda()
            new_label_clustering[:, :c3] = batch_coords

            # Loop over predicted semantics
            # print(result['segmentation'][i].shape, batch_mask.shape, batch_mask.sum())
            if result['segmentation'][i].shape[0] == batch_mask.shape[0]:
                semantic_pred = argmax(result['segmentation'][i][batch_mask])
            else: # adapt_labels was called from analysis tools (see below deghost_labels_and_predictions)
                # the problem in this case is that `segmentation` has already been deghosted
                semantic_pred = argmax(result['segmentation_noghost'][i][batch_mask])

            for semantic in unique(semantic_pred):
                semantic_mask = semantic_pred == semantic

                if true_mask is not None:
                    continue
                # Select voxels predicted as nonghost, but true ghosts
                mask = nonghost_mask & (label_seg[i][:, -1][batch_mask] == num_classes) & semantic_mask
                mask = where(mask)[0]
                # Now we need a special treatment for these, if there are any.
                if batch_coords[mask].shape[0] == 0:
                    continue
                tagged_voxels_count = 1 # to get the loop started
                X_true = batch_clustering[batch_clustering[:, -1] == semantic]
                if X_true.shape[0] == 0:
                    continue
                X_pred = batch_coords[mask]
                while tagged_voxels_count > 0 and X_pred.shape[0] > 0:
                    # print(batch_id, "while", X_true.shape, X_pred.shape, tagged_voxels_count)
                    #neighbors = knn(X_true[:, c1:c2].float(), X_pred[:, c1:c2].float(), 1)
                    #_, d = neighbors[0], neighbors[1]
                    d = compute_neighbor(X_true, X_pred)

                    # compute Chebyshev distance between predicted and true
                    # distances = torch.amax(torch.abs(X_true[neighbors[1], c1:c2] - X_pred[neighbors[0], c1:c2]), dim=1)
                    distances =compute_distances(X_true[d], X_pred)
                    #print(distances)
                    select_mask = distances <= 1

                    tagged_voxels_count = select_mask.sum()
                    if tagged_voxels_count > 0:
                        # We assign label of closest true nonghost voxel to those within Chebyshev distance 1
                        additional_label_clustering = concatenate0([X_pred[select_mask],
                                                                X_true[d[select_mask], c3:]])

                        new_label_clustering[mask[select_mask]] = additional_label_clustering
                        mask = mask[~select_mask]
                        # Update for next iteration
                        X_true = additional_label_clustering
                        X_pred = X_pred[~select_mask]

            # Include true nonghost voxels by default
            new_label_clustering[label_seg[i][batch_mask, -1] < num_classes] = make_float(batch_clustering)
            # Now we save - need only to keep predicted nonghost voxels.
            label_c.append(new_label_clustering[nonghost_mask])
        label_c = concatenate1(label_c)
        complete_label_clustering.append(label_c)

    return complete_label_clustering


def adapt_labels(*args, **kwargs):
    """
    Kept for backward compatibility, to deprecate soon.

    See Also
    --------
    adapt_labels_knn, adapt_labels_numpy
    """
    return adapt_labels_knn(*args, **kargs)


def adapt_labels_numpy(*args, **kwargs):
    """
    Numpy version of `adapt_labels`.

    See Also
    --------
    adapt_labels, adapt_labels_knn
    """
    return adapt_labels_knn(*args, **kwargs, use_numpy=True)
    # c1, c2 = coords_col
    # complete_label_clustering = []
    #
    # c3 = max(c2, batch_col+1)
    # for i in range(len(label_seg)):
    #     coords = label_seg[i][:, :4]
    #     label_c = []
    #     #print(len(coords[:, -1].unique()))
    #     for batch_id in np.unique(coords[:, batch_col]):
    #         batch_mask = coords[:, batch_col] == batch_id
    #         batch_coords = coords[batch_mask]
    #         batch_clustering = label_clustering[i][label_clustering[i][:, batch_col] == batch_id]
    #
    #         # Prepare new labels
    #         new_label_clustering = -1. * np.ones((batch_coords.shape[0], label_clustering[i].shape[1]))
    #         new_label_clustering[:, :c3] = batch_coords
    #
    #         nonghost_mask = (result['ghost'][i][batch_mask].argmax(axis=1) == 0)
    #         # Select voxels predicted as nonghost, but true ghosts
    #         mask = nonghost_mask & (label_seg[i][:, -1][batch_mask] == num_classes)
    #         if len(batch_clustering):
    #             # Assign them to closest cluster
    #             d = cdist(batch_coords[mask, c1:c2], batch_clustering[:, c1:c2]).argmin(axis=1)
    #             additional_label_clustering = np.concatenate([batch_coords[mask], batch_clustering[d, 4:]], axis=1)
    #             new_label_clustering[mask] = additional_label_clustering
    #
    #         if len(batch_clustering):
    #             new_label_clustering[label_seg[i][batch_mask, -1] < num_classes] = batch_clustering
    #
    #         label_c.append(new_label_clustering[nonghost_mask])
    #     label_c = np.concatenate(label_c, axis=0)
    #     complete_label_clustering.append(label_c)
    # return complete_label_clustering


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
        result['ghost'][i].argmax(axis=1) == 0 \
            for i in range(len(result['ghost']))]

    if 'segment_label' in data_blob:
        data_blob['true_ghost_mask'] = [
            data_blob['segment_label'][i][:, -1] < 5 \
                for i in range(len(data_blob['segment_label']))]

    data_blob['input_data_noghost'] = data_blob['input_data']

    if 'segment_label' in data_blob:
        data_blob['input_data_trueghost'] = [data_blob['input_data'][i][mask] \
            for i, mask in enumerate(data_blob['true_ghost_mask'])]

    data_blob['input_data'] = [data_blob['input_data'][i][mask] \
        for i, mask in enumerate(result['ghost_mask'])]


    if 'cluster_label' in data_blob \
        and data_blob['cluster_label'] is not None:
        # Save the clust_data before deghosting
        data_blob['cluster_label_noghost'] = data_blob['cluster_label']
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
        result['segmentation_noghost'] = result['segmentation']
        result['segmentation'] = [
            result['segmentation'][i][result['ghost_mask'][i]] \
                for i in range(len(result['segmentation']))]

    if 'kinematics_label' in data_blob \
        and data_blob['kinematics_label'] is not None:
        data_blob['kinematics_label'] = adapt_labels_numpy(
            result,
            data_blob['segment_label'],
            data_blob['kinematics_label'])

    # This needs to come last - in adapt_labels seg_label is the original one
    if 'segment_label' in data_blob \
        and data_blob['segment_label'] is not None:
        data_blob['segment_label_noghost'] = data_blob['segment_label']
        data_blob['segment_label'] = [
            data_blob['segment_label'][i][result['ghost_mask'][i]] \
                for i in range(len(data_blob['segment_label']))]
