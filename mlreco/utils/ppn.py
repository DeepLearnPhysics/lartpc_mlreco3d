from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import scipy
from mlreco.utils.dbscan import dbscan_types
import torch


def contains(meta, point, point_type="3d"):
    """
    Decides whether a point is contained in the box defined by meta.

    Parameters
    ----------
    meta: larcv::Voxel3DMeta or larcv::ImageMeta
    point: larcv::Point3D or larcv::Point2D
    point_type: str, optional
        Has to be "3d" for 3D, otherwise anything else works for 2D.

    Returns
    -------
    bool
    """
    if point_type == '3d':
        return point.x() >= meta.min_x() and point.y() >= meta.min_y() \
            and point.z() >= meta.min_z() and point.x() <= meta.max_x() \
            and point.y() <= meta.max_y() and point.z() <= meta.max_z()
    else:
        return point.x() >= meta.min_x() and point.x() <= meta.max_x() \
            and point.y() >= meta.min_y() and point.y() <= meta.max_y()


def pass_particle(gt_type, start, end, energy_deposit, vox_count):
    """
    Filters particles based on their type, voxel count and energy deposit.

    Parameters
    ----------
    gt_type: int
    start: larcv::Point3D
    end: larcv::Point3D
    energy_deposit: float
    vox_count: int

    Returns
    -------
    bool

    Notes
    -----
    Made during DUNE Pi0 workshop (?), do we need to keep it here?
    Assumes 3D
    """
    if (np.power((start.x()-end.x()),2) + np.power((start.y()-end.y()),2) + np.power((start.z()-end.z()),2)) < 6.25:
        return True
    if gt_type == 0: return vox_count<7 or energy_deposit < 50.
    if gt_type == 1: return vox_count<7 or energy_deposit < 10.
    if gt_type == 2: return vox_count<7 or energy_deposit < 1.
    if gt_type == 3: return vox_count<5 or energy_deposit < 5.
    if gt_type == 4: return vox_count<5 or energy_deposit < 5.


def get_ppn_info(particle_v, meta, point_type="3d", min_voxel_count=7, min_energy_deposit=10):
    """
    Gets particle points coordinates and informations for running PPN.

    Parameters
    ----------
    particle_v:
    meta: larcv::Voxel3DMeta or larcv::ImageMeta
    point_type: str, optional
    min_voxel_count: int, optional
    min_energy_deposit: float, optional

    Returns
    -------
    np.array
        Array of points of shape (N, 10) where 10 = x,y,z + point type + pdg
        code + energy deposit + num voxels + energy_init + energy_deposit

    Notes
    -----
    We skip some particles under specific conditions (e.g. low energy deposit,
    low voxel count, nucleus track, etc.)
    For now in 2D we assume a specific 2d projection (plane).
    """
    if point_type not in ["3d", "xy", "yz", "zx"]:
        raise Exception("Point type not supported in PPN I/O.")
    # from larcv import larcv
    gt_positions = []
    for particle in particle_v:
        pdg_code = abs(particle.pdg_code())
        prc = particle.creation_process()
        # Skip particle under some conditions
        if particle.energy_deposit() < min_energy_deposit or particle.num_voxels() < min_voxel_count:
            continue
        if pdg_code > 1000000000:  # skipping nucleus trackid
            continue
        if pdg_code == 11 or pdg_code == 22:  # Shower
            if not contains(meta, particle.first_step(), point_type=point_type):
                continue
            # Skipping delta ray
            #if particle.parent_pdg_code() == 13 and particle.creation_process() == "muIoni":
            #    continue

        # Determine point type
        if (pdg_code == 2212):
            gt_type = 0 # proton
        elif pdg_code != 22 and pdg_code != 11:
            gt_type = 1
        elif pdg_code == 22:
            gt_type = 2
        else:
            if prc == "primary" or prc == "nCapture" or prc == "conv":
                gt_type = 2 # em shower
            elif prc == "muIoni" or prc == "hIoni":
                gt_type = 3 # delta
            elif prc == "muMinusCaptureAtRest" or prc == "muPlusCaptureAtRest" or prc == "Decay":
                gt_type = 4 # michel

        #if pass_particle(gt_type,particle.first_step(),particle.last_step(),particle.energy_deposit(),particle.num_voxels()):
        #    continue

        # TODO deal with different 2d projections
        record = [pdg_code,
                  particle.energy_deposit(),
                  particle.num_voxels(),
                  particle.energy_init()]
        # Register start point
        x = particle.first_step().x()
        y = particle.first_step().y()
        z = particle.first_step().z()
        if point_type == '3d':
            x = (x - meta.min_x()) / meta.size_voxel_x()
            y = (y - meta.min_y()) / meta.size_voxel_y()
            z = (z - meta.min_z()) / meta.size_voxel_z()
            gt_positions.append([x, y, z, gt_type] + record)
        else:
            x = (x - meta.min_x()) / meta.pixel_width()
            y = (y - meta.min_y()) / meta.pixel_height()
            gt_positions.append([x, y, gt_type] + record)

        # Register end point (for tracks only)
        if gt_type == 0 or gt_type == 1:
            x = particle.last_step().x()
            y = particle.last_step().y()
            z = particle.last_step().z()
            if point_type == '3d':
                x = (x - meta.min_x()) / meta.size_voxel_x()
                y = (y - meta.min_y()) / meta.size_voxel_y()
                z = (z - meta.min_z()) / meta.size_voxel_z()
                gt_positions.append([x, y, z, gt_type] + record)
            else:
                x = (x - meta.min_x()) / meta.pixel_width()
                y = (y - meta.min_y()) / meta.pixel_height()
                gt_positions.append([x, y, gt_type] + record)

    return np.array(gt_positions)


def nms_numpy(im_proposals, im_scores, threshold, size):
    """
    Runs NMS algorithm on a list of predicted points and scores.

    Parameters
    ----------
    im_proposals: np.array
        Shape (N, data_dim). Predicted points.
    im_scores: np.array
        Shape (N, 2). Predicted scores.
    threshold: float
        Threshold for overlap
    size: int
        Half side of square window defined around each point

    Returns
    -------
    np.array
        boolean array of same length as points/scores
    """
    # TODO: looks like this doesn't account for batches
    dim = im_proposals.shape[-1]
    coords = []
    for d in range(dim):
        coords.append(im_proposals[:, d] - size)
    for d in range(dim):
        coords.append(im_proposals[:, d] + size)
    coords = np.array(coords)

    areas = np.ones_like(coords[0])
    areas = np.prod(coords[dim:] - coords[0:dim] + 1, axis=0)

    order = im_scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx = np.maximum(coords[:dim, i][:, np.newaxis], coords[:dim, order[1:]])
        yy = np.minimum(coords[dim:, i][:, np.newaxis], coords[dim:, order[1:]])
        w = np.maximum(0.0, yy - xx + 1)
        inter = np.prod(w, axis=0)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]

    return keep


def group_points(ppn_pts, batch, label):
    """
    if there are multiple ppn points in a very similar location, return the average pos

    Parameters
    ----------
    ppn_pts: np.array
    batch: np.array
    label: np.array

    Returns
    -------
    np.array
    """
    ppn_pts_new = []
    batch_new = []
    label_new = []
    for b in np.unique(batch):
        bsel = batch == b
        ppn_pts_sel = ppn_pts[bsel]
        label_sel = label[bsel]
        clusts = dbscan_types(ppn_pts_sel, label_sel, epsilon=1.99,  minpts=1, typemin=0, typemax=5)
        for c in clusts:
            # append mean of points
            ppn_pts_new.append(np.mean(ppn_pts_sel[c],axis=0))
            # append batch
            batch_new.append(b)
            label_new.append(np.mean(label_sel[c]))

    return np.array(ppn_pts_new), np.array(batch_new), np.array(label_new)


def uresnet_ppn_type_point_selector(data, out, score_threshold=0.5,
                                    type_threshold=100, entry=0, **kwargs):
    """
    Postprocessing of PPN points.
    Parameters
    ----------
    data - 5-types sparse tensor
    out - uresnet_ppn_type output
    Returns
    -------
    [x,y,z,bid,label] of ppn-predicted points
    """
    event_data = data#.cpu().detach().numpy()
    points = out['points'][entry]#.cpu().detach().numpy()
    mask = out['mask_ppn2'][entry]#.cpu().detach().numpy()
    # predicted type labels
    # uresnet_predictions = torch.argmax(out['segmentation'][0], -1).cpu().detach().numpy()
    uresnet_predictions = np.argmax(out['segmentation'][entry], -1)
    scores = scipy.special.softmax(points[:, 3:5], axis=1)

    if 'ghost' in out:
        mask_ghost = np.argmax(out['ghost'][entry], axis=1) == 0
        event_data = event_data[mask_ghost]
        points = points[mask_ghost]
        mask = mask[mask_ghost]
        uresnet_predictions = uresnet_predictions[mask_ghost]
        scores = scores[mask_ghost]

    all_points = []
    all_batch = []
    all_labels = []
    batch_ids = event_data[:, 3]
    for b in np.unique(batch_ids):
        final_points = []
        final_scores = []
        final_labels = []
        batch_index = batch_ids == b
        mask = ((~(mask[batch_index] == 0)).any(axis=1)) & (scores[batch_index][:, 1] > score_threshold)
        num_classes = 5
        ppn_type_predictions = np.argmax(scipy.special.softmax(points[batch_index][mask][:, 5:], axis=1), axis=1)
        for c in range(num_classes):
            uresnet_points = uresnet_predictions[batch_index][mask] == c
            ppn_points = ppn_type_predictions == c
            if ppn_points.shape[0] > 0 and uresnet_points.shape[0] > 0:
                d = scipy.spatial.distance.cdist(points[batch_index][mask][ppn_points][:, :3] + event_data[batch_index][mask][ppn_points][:, :3] + 0.5, event_data[batch_index][mask][uresnet_points][:, :3])
                ppn_mask = (d < type_threshold).any(axis=1)
                final_points.append(points[batch_index][mask][ppn_points][ppn_mask][:, :3] + 0.5 + event_data[batch_index][mask][ppn_points][ppn_mask][:, :3])
                final_scores.append(scores[batch_index][mask][ppn_points][ppn_mask])
                final_labels.append(ppn_type_predictions[ppn_points][ppn_mask])
        final_points = np.concatenate(final_points, axis=0)
        final_scores = np.concatenate(final_scores, axis=0)
        final_labels = np.concatenate(final_labels, axis=0)
        clusts = dbscan_types(final_points, final_labels, epsilon=1.99,  minpts=1, typemin=0, typemax=5)
        for c in clusts:
            # append mean of points
            all_points.append(np.mean(final_points[c], axis=0))
            all_batch.append(b)
            all_labels.append(np.mean(final_labels[c]))

    return np.column_stack((all_points, all_batch, all_labels))


def uresnet_ppn_point_selector(data, out, nms_score_threshold=0.8, entry=0,
                               window_size=4, score_threshold=0.9, **kwargs):
    """
    Basic selection of PPN points.

    Parameters
    ----------
    data - 5-types sparse tensor
    out - ppn output

    Returns
    -------
    [x,y,z,bid,label] of ppn-predicted points
    """
    # analysis_keys:
    #  segmentation: 3
    #  points: 0
    #  mask: 5
    #  ppn1: 1
    #  ppn2: 2
    # FIXME assumes 3D for now
    points = out['points'][entry]#.cpu().detach().numpy()
    #ppn1 = out['ppn1'][entry]#.cpu().detach().numpy()
    #ppn2 = out[2][0].cpu().detach().numpy()
    mask = out['mask_ppn2'][entry]#.cpu().detach().numpy()
    # predicted type labels
    pred_labels = np.argmax(out['segmentation'][entry], axis=-1)#.cpu().detach().numpy()

    scores = scipy.special.softmax(points[:, 3:5], axis=1)
    points = points[:,:3]


    # PPN predictions after masking
    mask = (~(mask == 0)).any(axis=1)

    scores = scores[mask]
    maskinds = np.where(mask)[0]
    keep = scores[:,1] > score_threshold

    # NMS filter
    keep2 = nms_numpy(points[mask][keep], scores[keep,1], nms_score_threshold, window_size)

    maskinds = maskinds[keep][keep2]
    points = points[maskinds]
    labels = pred_labels[maskinds]

    data_in = data#.cpu().detach().numpy()
    voxels = data_in[:,:3]
    ppn_pts = voxels[maskinds] + 0.5 + points
    batch = data_in[maskinds,3]
    label = pred_labels[maskinds]

    # TODO: only return single point in voxel per batch per label
    ppn_pts, batch, label = group_points(ppn_pts, batch, label)


    # Output should be in [x,y,z,bid,label] format
    pts_out = np.column_stack((ppn_pts, batch, label))

    # return indices of points in input, offsets
    return pts_out
