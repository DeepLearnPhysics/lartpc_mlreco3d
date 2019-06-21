from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import scipy
from mlreco.utils.dbscan import dbscan_types
import torch


def contains(meta, point, point_type="3d"):
    if point_type == '3d':
        return point.x() >= meta.min_x() and point.y() >= meta.min_y() \
            and point.z() >= meta.min_z() and point.x() <= meta.max_x() \
            and point.y() <= meta.max_y() and point.z() <= meta.max_z()
    else:
        return point.x() >= meta.min_x() and point.x() <= meta.max_x() \
            and point.y() >= meta.min_y() and point.y() <= meta.max_y()


def get_ppn_info(particle_v, meta, point_type="3d", min_voxel_count=5, min_energy_deposit=0.05):
    """
    Gets particle information for training ppn
    """
    if point_type not in ["3d", "xy", "yz", "zx"]:
        raise Exception("Point type not supported in PPN I/O.")
    # from larcv import larcv
    gt_positions = []
    for particle in particle_v:
        pdg_code = particle.pdg_code()
        prc = particle.creation_process()
        # Skip particle under some conditions
        if (particle.energy_deposit() < min_energy_deposit or particle.num_voxels() < min_voxel_count):
            continue
        if pdg_code > 1000000000:  # skipping nucleus trackid
            continue
        if pdg_code == 11 or pdg_code == 22 or pdg_code == -11:  # Shower
            if not contains(meta, particle.first_step(), point_type=point_type):
                continue
            # # Skipping delta ray
            # if (particle.parent_pdg_code() == 13 and prc == "muIoni") or prc == "hIoni":
            #     continue

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

        # TODO deal with different 2d projections
        # Register start point
        x = particle.first_step().x()
        y = particle.first_step().y()
        z = particle.first_step().z()
        if point_type == '3d':
            x = (x - meta.min_x()) / meta.size_voxel_x()
            y = (y - meta.min_y()) / meta.size_voxel_y()
            z = (z - meta.min_z()) / meta.size_voxel_z()
            gt_positions.append([x, y, z, gt_type])
        else:
            x = (x - meta.min_x()) / meta.pixel_width()
            y = (y - meta.min_y()) / meta.pixel_height()
            gt_positions.append([x, y, gt_type])

        # Register end point (for tracks only)
        if gt_type == 0 or gt_type == 1:
            x = particle.last_step().x()
            y = particle.last_step().y()
            z = particle.last_step().z()
            if point_type == '3d':
                x = (x - meta.min_x()) / meta.size_voxel_x()
                y = (y - meta.min_y()) / meta.size_voxel_y()
                z = (z - meta.min_z()) / meta.size_voxel_z()
                gt_positions.append([x, y, z, gt_type])
            else:
                x = (x - meta.min_x()) / meta.pixel_width()
                y = (y - meta.min_y()) / meta.pixel_height()
                gt_positions.append([x, y, gt_type])

    return np.array(gt_positions)



def nms_numpy(im_proposals, im_scores, threshold, size):
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


def uresnet_ppn_point_selector(data, out, nms_score_threshold=0.8, window_size=4, score_threshold=0.9, **kwargs):
    """
    input: 
        data - 5-types sparse tensor
        out - ppn output
    output:
        [x,y,z,bid,label] of ppn-predicted points
    """
    # analysis_keys:
    #  segmentation: 3
    #  points: 0
    #  mask: 5
    #  ppn1: 1
    #  ppn2: 2
    # FIXME assumes 3D for now
    points = out[0][0].cpu().detach().numpy()
    ppn1 = out[1][0].cpu().detach().numpy()
    ppn2 = out[2][0].cpu().detach().numpy()
    mask = out[5][0].cpu().detach().numpy()
    # predicted type labels
    pred_labels = torch.argmax(out[3][0],-1).cpu().detach().numpy()

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
    
    data_in = data.cpu().detach().numpy()
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
