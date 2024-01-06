import numpy as np
import scipy
import torch

from mlreco.utils import numba_local as nbl
from mlreco.utils import local_cdist
from mlreco.utils.dbscan import dbscan_types, dbscan_points
from mlreco.utils.globals import (BATCH_COL, COORD_COLS, PPN_RTYPE_COLS,
        PPN_RPOS_COLS, PPN_END_COLS, TRACK_SHP, LOWES_SHP, UNKWN_SHP)


def get_ppn_labels(particle_v, meta, dim=3, min_voxel_count=5,
        min_energy_deposit=0, include_point_tagging=True):
    '''
    Gets particle point coordinates and informations for running PPN.

    We skip some particles under specific conditions (e.g. low energy deposit,
    low voxel count, nucleus track, etc.)

    Parameters
    ----------
    particle_v : List[larcv.Particle]
        List of LArCV particle objects in the image
    meta : larcv::Voxel3DMeta or larcv::ImageMeta
        Metadata information
    dim : int, default 3
        Number of dimensions of the image
    min_voxel_count : int, default 5
        Minimum number of voxels associated with a particle to be included
    min_energy_deposit : float, default 0
        Minimum energy deposition associated with a particle to be included
    include_point_tagging : bool, default True
        If True, include an a label of 0 for start points and 1 for end points

    Returns
    -------
    np.array
        Array of points of shape (N, 5/6) where 5/6 = x,y,z + point type
        + particle index [+ start (0) or end (1) point tagging]
    '''
    # Check on dimension
    if dim not in [2, 3]:
        raise ValueError('The image dimension must be either 2 or 3, ',
                f'got {dim} instead.')

    # Loop over true particles
    part_info = []
    for part_index, particle in enumerate(particle_v):
        # Check that the particle has the expected index
        assert part_index == particle.id()

        # If the particle does not meet minimum energy/size requirements, skip
        if particle.energy_deposit() < min_energy_deposit or \
                particle.num_voxels() < min_voxel_count:
            continue

        # If the particle is a nucleus, skip.
        # TODO: check if it's useful
        pdg_code = abs(particle.pdg_code())
        if pdg_code > 1000000000:  # skipping nucleus trackid
            continue

        # If a shower has its first step outside of detector boundaries, skip
        # TODO: check if it's useful
        if pdg_code == 11 or pdg_code == 22:
            if not image_contains(meta, particle.first_step(), dim):
                continue

        # Skip low energy scatters and unknown shapes
        shape = particle.shape()
        if particle.shape() in [LOWES_SHP, UNKWN_SHP]:
            continue

        # Append the start point with the rest of the particle information
        first_step = image_coordinates(meta, particle.first_step(), dim)
        part_extra = [shape, part_index, 0] if include_point_tagging else [shape, part_index]
        part_info.append(first_step + part_extra)

        # Append the end point as well, for tracks only
        if shape == TRACK_SHP:
            last_step  = image_coordinates(meta, particle.last_step(), dim)
            part_extra = [shape, part_index, 1] if include_point_tagging else [shape, part_index]
            part_info.append(last_step + part_extra)

    if not len(part_info):
        return np.empty((0,6), dtype=np.float32)
    return np.array(part_info)


def get_ppn_predictions(data, out, score_threshold=0.5, type_score_threshold=0.5,
                        type_threshold=1.999, entry=0, score_pool='max', enforce_type=True,
                        selection=None, num_classes=5, apply_deghosting=True, **kwargs):
    '''
    Converts the raw output of PPN to a set of proposed points.

    Parameters
    ----------
    data - 5-types sparse tensor
    out - output dictionary of the full chain
    score_threshold - minimal detection score
    type_score_threshold - minimal score for a point type prediction to be considered
    type_threshold - distance threshold for matching w/ semantic type prediction
    entry - which index to look at (within a batch of events)
    score_pool - which operation to use to pool PPN points scores (max/min/mean)
    enforce_type - whether to force PPN points predicted of type X
                    to be within N voxels of a voxel with same predicted semantic
    selection - list of list of indices to consider exclusively (eg to get PPN predictions
                within a cluster). Shape Batch size x N_voxels (not square)
    Returns
    -------
    [bid,x,y,z,score softmax values (2 columns), occupancy,
    type softmax scores (5 columns), predicted type,
    (optional) endpoint type]
    1 row per ppn-predicted points
    '''
    unwrapped = len(out['ppn_points']) == len(out['ppn_coords'])
    event_data = data#.cpu().detach().numpy()
    points = out['ppn_points'][entry]
    ppn_coords = out['ppn_coords'][entry] if unwrapped else out['ppn_coords']
    enable_classify_endpoints = 'ppn_classify_endpoints' in out
    if enable_classify_endpoints:
        classify_endpoints = out['ppn_classify_endpoints'][entry]

    ppn_mask = out['ppn_masks'][entry][-1] if unwrapped else out['ppn_masks'][-1]
    uresnet_predictions = np.argmax(out['segmentation'][entry], -1)

    if 'ghost' in out and apply_deghosting:
        mask_ghost = np.argmax(out['ghost'][entry], axis=1) == 0
        event_data = event_data[mask_ghost]
        #points = points[mask_ghost]
        #if enable_classify_endpoints:
        #    classify_endpoints = classify_endpoints[mask_ghost]
        #ppn_mask = ppn_mask[mask_ghost]
        uresnet_predictions = uresnet_predictions[mask_ghost]
        #scores = scores[mask_ghost]

    scores = scipy.special.softmax(points[:, PPN_RPOS_COLS], axis=1)
    pool_op = None
    if   score_pool == 'max'  : pool_op=np.amax
    elif score_pool == 'mean' : pool_op = np.amean
    else: raise ValueError('score_pool must be either "max" or "mean"!')
    all_points = []
    all_occupancy = []
    all_types  = []
    all_scores = []
    all_batch  = []
    all_softmax = []
    all_endpoints = []
    batch_ids  = event_data[:, BATCH_COL]
    for b in np.unique(batch_ids):
        final_points = []
        final_scores = []
        final_types = []
        final_softmax = []
        final_endpoints = []
        batch_index = batch_ids == b
        batch_index2 = ppn_coords[-1][:, 0] == b
        # print(batch_index.shape, batch_index2.shape, ppn_mask.shape, scores.shape)
        mask = ((~(ppn_mask[batch_index2] == 0)).any(axis=1)) & (scores[batch_index2][:, 1] > score_threshold)
        # If we want to restrict the postprocessing to specific voxels
        # (e.g. within a particle cluster, not the full event)
        # then use the argument `selection`.
        if selection is not None:
            new_mask = np.zeros(mask.shape, dtype=np.bool)
            if len(selection) > 0 and isinstance(selection[0], list):
                indices = np.array(selection[int(b)])
            else:
                indices = np.array(selection)
            new_mask[indices] = mask[indices]
            mask = new_mask

        ppn_type_predictions = np.argmax(scipy.special.softmax(points[batch_index2][mask][:, PPN_RTYPE_COLS], axis=1), axis=1)
        ppn_type_softmax = scipy.special.softmax(points[batch_index2][mask][:, PPN_RTYPE_COLS], axis=1)
        if enable_classify_endpoints:
            ppn_classify_endpoints = scipy.special.softmax(classify_endpoints[batch_index2][mask], axis=1)
        if enforce_type:
            for c in range(num_classes):
                uresnet_points = uresnet_predictions[batch_index][mask] == c
                ppn_points = ppn_type_softmax[:, c] > type_score_threshold #ppn_type_predictions == c
                if np.count_nonzero(ppn_points) > 0 and np.count_nonzero(uresnet_points) > 0:
                    d = scipy.spatial.distance.cdist(points[batch_index2][mask][ppn_points][:, :3] + event_data[batch_index][mask][ppn_points][:, COORD_COLS] + 0.5, event_data[batch_index][mask][uresnet_points][:, COORD_COLS])
                    ppn_mask2 = (d < type_threshold).any(axis=1)
                    final_points.append(points[batch_index2][mask][ppn_points][ppn_mask2][:, :3] + 0.5 + event_data[batch_index][mask][ppn_points][ppn_mask2][:, COORD_COLS])
                    final_scores.append(scores[batch_index2][mask][ppn_points][ppn_mask2])
                    final_types.append(ppn_type_predictions[ppn_points][ppn_mask2])
                    final_softmax.append(ppn_type_softmax[ppn_points][ppn_mask2])
                    if enable_classify_endpoints:
                        final_endpoints.append(ppn_classify_endpoints[ppn_points][ppn_mask2])
        else:
            final_points = [points[batch_index2][mask][:, :3] + 0.5 + event_data[batch_index][mask][:, COORD_COLS]]
            final_scores = [scores[batch_index2][mask]]
            final_types = [ppn_type_predictions]
            final_softmax =  [ppn_type_softmax]
            if enable_classify_endpoints:
                final_endpoints = [ppn_classify_endpoints]
        if len(final_points)>0:
            final_points = np.concatenate(final_points, axis=0)
            final_scores = np.concatenate(final_scores, axis=0)
            final_types  = np.concatenate(final_types,  axis=0)
            final_softmax = np.concatenate(final_softmax, axis=0)
            if enable_classify_endpoints:
                final_endpoints = np.concatenate(final_endpoints, axis=0)
            if final_points.shape[0] > 0:
                clusts = dbscan_points(final_points, epsilon=1.99,  minpts=1)
                for c in clusts:
                    # append mean of points
                    all_points.append(np.mean(final_points[c], axis=0))
                    all_occupancy.append(len(c))
                    all_scores.append(pool_op(final_scores[c], axis=0))
                    all_types.append (pool_op(final_types[c],  axis=0))
                    all_softmax.append(pool_op(final_softmax[c], axis=0))
                    if enable_classify_endpoints:
                        all_endpoints.append(pool_op(final_endpoints[c], axis=0))
                    all_batch.append(b)
    result = (all_batch, all_points, all_scores, all_occupancy, all_softmax, all_types,)
    if enable_classify_endpoints:
        result = result + (all_endpoints,)
    result = np.column_stack( result )
    if len(result) == 0:
        if enable_classify_endpoints:
            return np.empty((0, 15), dtype=np.float32)
        else:
            return np.empty((0, 13), dtype=np.float32)
    return result


def get_particle_points(coords, clusts, clusts_seg, ppn_points, classes=None,
        anchor_points=True, enhance_track_points=False, approx_farthest_points=True):
    '''
    Given a list particle or fragment clusters, leverage the raw PPN output
    to produce a list of start points for shower objects and of end points
    for track objects:
    - For showers, pick the most likely PPN point
    - For tracks, pick the two points furthest away from each other

    Parameters
    ----------
    coords : numpy.ndarray
        Array of coordinates of voxels in the image
    clusts : List[numpy.ndarray]
        List of clusters representing the fragment or particle objects
    clusts_seg : numpy.ndarray
        Array of cluster semantic types
    ppn_points : numpy.ndarray
        Raw output of PPN
    anchor_points : bool, default True
        If `True`, the point estimates are brought to the closest cluster voxel
    approx_farthest_points: bool, default True
        If `True`, approximate the computation of the two farthest points
    enhance_track_points, default False
        If `True`, tracks leverage PPN predictions to provide a more
        accurate estimate of the end points. This needs to be avoided for
        track fragments, as PPN is typically not trained to find end points
        for them. If set to `False`, the two voxels furthest away from each
        other are picked.
    '''

    # Loop over the relevant clusters
    points = np.empty((len(clusts), 6), dtype=np.float32)
    for i, c in enumerate(clusts):
        # Get cluster coordinates
        clust_coords = coords[c]

        # Deal with tracks
        if clusts_seg[i] == TRACK_SHP:
            # Get the two most separated points in the cluster
            idxs = [0, 0]
            method = 'brute' if not approx_farthest_points else 'recursive'
            idxs[0], idxs[1], _ = nbl.farthest_pair(clust_coords, method)
            end_points = clust_coords[idxs]

            # If requested, enhance using the PPN predictions. Only consider
            # points in the cluster that have a positive score
            if enhance_track_points:
                pos_mask = ppn_points[c][idxs, PPN_RPOS_COLS[1]] \
                        >= ppn_points[c][idxs, PPN_RPOS_COLS[0]]
                end_points += pos_mask * (points_tensor[idxs, :3] + 0.5)

            # If needed, anchor the track endpoints to the track cluster
            if anchor_points and enhance_track_points:
                dist_mat   = nbl.cdist(end_points, clust_coords)
                end_points = clust_coords[np.argmin(dist_mat, axis=1)]

            # Store
            points[i] = end_points.flatten()

        # Deal with the rest (EM activity)
        else:
            # Only use positive voxels and give precedence to predictions
            # that are contained within the voxel making the prediction.
            ppn_scores = nbl.softmax(ppn_points[c][:, PPN_RPOS_COLS], axis=1)[:,-1]
            val_index  = np.where(np.all(np.abs(ppn_points[c, :3] < 1.)))[0]
            best_id    = val_index[np.argmax(ppn_scores[val_index])] \
                    if len(val_index) else np.argmax(ppn_scores)
            start_point = clust_coords[best_id] \
                    + ppn_points[c][best_id, :3] + 0.5

            # If needed, anchor the shower start point to the shower cluster
            if anchor_points:
                dists = nbl.cdist(np.atleast_2d(start_point), clust_coords)
                start_point = clust_coords[np.argmin(dists)]

            # Store twice to preserve the feature vector length
            points[i] = np.concatenate([start_point, start_point])

    # Return points
    return points


def check_track_orientation_ppn(start_point, end_point, ppn_candidates):
    '''
    Use the PPN point assignments as a basis to orient a track. Match
    the end points of a track to the closest PPN candidate and pick the
    candidate with the highest start score as the start point

    Parameters
    ----------
    start_point : np.ndarray
        (3) Start point of the track
    end_point : np.ndarray
        (3) End point of the track
    ppn_candidates : np.ndarray
        (N, 10)  PPN point candidates and their associated scores

    Returns
    -------
    bool
       Returns `True` if the start point provided is correct, `False`
       if the end point is more likely to be the start point.
    '''
    # If there's no PPN candidates, nothing to do here
    if not len(ppn_candidates):
        return True

    # Get the candidate coordinates and end point classification predictions
    ppn_points = ppn_candidates[:, COORD_COLS]
    end_scores = ppn_candidates[:, PPN_END_COLS]

    # Compute the distance between the track end points and the PPN candidates
    end_points = np.vstack([start_point, end_point])
    dist_mat = nbl.cdist(end_points, ppn_points)

    # If both track end points are closest to the same PPN point, the start
    # point must be closest to it if the score is high, farthest otherwise
    argmins = np.argmin(dist_mat, axis=1)
    if argmins[0] == argmins[1]:
        label = np.argmax(end_scores[argmins[0]])
        dists = dist_mat[[0,1], argmins]
        return (label == 0 and dists[0] < dists[1]) or \
                (label == 1 and dists[1] < dists[0])

    # In all other cases, check that the start point is associated with the PPN
    # point with the lowest end score
    end_scores = end_scores[argmins, -1]
    return end_scores[0] < end_scores[1]


def image_contains(meta, point, dim=3):
    '''
    Checks whether a point is contained in the image box defined by meta.

    Parameters
    ----------
    meta : larcv::Voxel3DMeta or larcv::ImageMeta
        Metadata information
    point : larcv::Point3D or larcv::Point2D
        Point to check on
    dim: int, default 3
         Number of dimensions of the image

    Returns
    -------
    bool
        True if the point is contained in the image box
    '''
    if dim == 3:
        return point.x() >= meta.min_x() and point.y() >= meta.min_y() \
                and point.z() >= meta.min_z() and point.x() <= meta.max_x() \
                and point.y() <= meta.max_y() and point.z() <= meta.max_z()
    else:
        return point.x() >= meta.min_x() and point.x() <= meta.max_x() \
                and point.y() >= meta.min_y() and point.y() <= meta.max_y()


def image_coordinates(meta, point, dim=3):
    '''
    Returns the coordinates of a point in units of pixels with an image.

    Parameters
    ----------
    meta : larcv::Voxel3DMeta or larcv::ImageMeta
        Metadata information
    point : larcv::Point3D or larcv::Point2D
        Point to convert the units of
    dim: int, default 3
         Number of dimensions of the image

    Returns
    -------
    bool
        True if the point is contained in the image box
    '''
    x, y, z = point.x(), point.y(), point.z()
    if dim == 3:
        x = (x - meta.min_x()) / meta.size_voxel_x()
        y = (y - meta.min_y()) / meta.size_voxel_y()
        z = (z - meta.min_z()) / meta.size_voxel_z()
        return [x, y, z]
    else:
        x = (x - meta.min_x()) / meta.size_voxel_x()
        y = (y - meta.min_y()) / meta.size_voxel_y()
        return [x, y]


def uresnet_ppn_type_point_selector(*args, **kwargs):
        from warnings import warn
        warn('uresnet_ppn_type_point_selector is deprecated,'
             'use get_ppn_predictions instead')
        return get_ppn_predictions(*args, **kwargs)
