import numpy as np
import scipy
import torch

from mlreco.utils import local_cdist
from mlreco.utils.dbscan import dbscan_types, dbscan_points

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


def get_ppn_info(particle_v, meta, point_type="3d", min_voxel_count=5, min_energy_deposit=0, use_particle_shape=True):
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
        Array of points of shape (N, 11) where 11 = x,y,z + point type + pdg
        code + energy deposit + num voxels + energy_init + particle index +
        start (0) or end (1) point tagging

    Notes
    -----
    We skip some particles under specific conditions (e.g. low energy deposit,
    low voxel count, nucleus track, etc.)
    For now in 2D we assume a specific 2d projection (plane).
    """
    if point_type not in ["3d", "xy", "yz", "zx"]:
        raise Exception("Point type not supported in PPN I/O.")
    from larcv import larcv
    gt_positions = []
    for part_index, particle in enumerate(particle_v):
        pdg_code = abs(particle.pdg_code())
        prc = particle.creation_process()
        # Skip particle under some conditions
        if particle.energy_deposit() < min_energy_deposit or particle.num_voxels() < min_voxel_count:
            # print('[a] skipping',part_index,'/',len(particle_v), 'with pdg ', pdg_code, ' num voxels ', particle.num_voxels(), ' energy deposit ', particle.energy_deposit(), ' energy init ', particle.energy_init())
            # print(' created at ',
            #     (particle.position().x() - meta.min_x()) / meta.size_voxel_x(),
            #     (particle.position().y() - meta.min_y()) / meta.size_voxel_y(),
            #     (particle.position().z() - meta.min_z()) / meta.size_voxel_z())
            continue
        if pdg_code > 1000000000:  # skipping nucleus trackid
            #print('[b] skipping',part_index,'/',len(particle_v))
            continue
        if pdg_code == 11 or pdg_code == 22:  # Shower
            if not contains(meta, particle.first_step(), point_type=point_type):
                #print('[c] skipping particle id',particle.id(),'as its start is not contained in the box...')
                #print(particle.dump())
                #print(meta.dump())

                continue
            # Skipping delta ray
            #if particle.parent_pdg_code() == 13 and particle.creation_process() == "muIoni":
            #    continue

        # Determine point type
        if not use_particle_shape:
            gt_type = -1
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
            if gt_type == -1: # FIXME unknown point type ??
                #print('[d] skipping',part_index,'/',len(particle_v))
                continue
        else:
            from larcv import larcv
            gt_type = particle.shape()
            if particle.shape() in [larcv.kShapeLEScatter, larcv.kShapeUnknown]:
                #print('[e] skipping',part_index,'/',len(particle_v))
                continue

        #if pass_particle(gt_type,particle.first_step(),particle.last_step(),particle.energy_deposit(),particle.num_voxels()):
        #    continue

        # TODO deal with different 2d projections
        record = [pdg_code,
                  particle.energy_deposit(),
                  particle.num_voxels(),
                  particle.energy_init(),
                  part_index]
        assert(part_index == particle.id())

        # Register start point
        x = particle.first_step().x()
        y = particle.first_step().y()
        z = particle.first_step().z()
        if point_type == '3d':
            x = (x - meta.min_x()) / meta.size_voxel_x()
            y = (y - meta.min_y()) / meta.size_voxel_y()
            z = (z - meta.min_z()) / meta.size_voxel_z()
            gt_positions.append([x, y, z, gt_type] + record + [0])
            #if str(pdg_code) in ["22", "11"]:
            #    print(pdg_code, [x, y, z, gt_type] + record)
        else:
            x = (x - meta.min_x()) / meta.pixel_width()
            y = (y - meta.min_y()) / meta.pixel_height()
            gt_positions.append([x, y, gt_type] + record + [0])

        # Register end point (for tracks only)
        track_types = [0,1]
        if use_particle_shape:
            track_types = [larcv.kShapeTrack]
        if gt_type in track_types:
            x = particle.last_step().x()
            y = particle.last_step().y()
            z = particle.last_step().z()
            if point_type == '3d':
                x = (x - meta.min_x()) / meta.size_voxel_x()
                y = (y - meta.min_y()) / meta.size_voxel_y()
                z = (z - meta.min_z()) / meta.size_voxel_z()
                gt_positions.append([x, y, z, gt_type] + record + [1])
                #if str(pdg_code) == "13":
                #    print(pdg_code, [x, y, z, gt_type] + record)
            else:
                x = (x - meta.min_x()) / meta.pixel_width()
                y = (y - meta.min_y()) / meta.pixel_height()
                gt_positions.append([x, y, gt_type] + record + [1])


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


def uresnet_ppn_type_point_selector(data, out, score_threshold=0.5, type_score_threshold=0.5,
                                    type_threshold=1.999, entry=0, score_pool='max', enforce_type=True,
                                    batch_col=0, coords_col=(1, 4), type_col=(3,8), score_col=(8,10),
                                    selection=None, num_classes=5, apply_deghosting=True, **kwargs):
    """
    Postprocessing of PPN points.

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
    """
    event_data = data#.cpu().detach().numpy()
    points = out['points'][0]#[entry]#.cpu().detach().numpy()
    ppn_coords = out['ppn_coords']
    # If 'points' is specified in `concat_result`,
    # then it won't be unwrapped.
    if len(points) == len(ppn_coords[-1]):
        pass
        # print(entry, np.unique(ppn_coords[-1][:, 0], return_counts=True))
        #points = points[ppn_coords[-1][:, 0] == entry, :]
    else: # in case it has been unwrapped (possible in no-ghost scenario)
        points = out['points'][entry]

    enable_classify_endpoints = 'classify_endpoints' in out
    print("ENABLE CLASSIFY ENDPOINTS = ", enable_classify_endpoints)
    if enable_classify_endpoints:
        classify_endpoints = out['classify_endpoints'][0]
        print(classify_endpoints)

    mask_ppn = out['mask_ppn'][-1]
    # predicted type labels
    # uresnet_predictions = torch.argmax(out['segmentation'][0], -1).cpu().detach().numpy()
    uresnet_predictions = np.argmax(out['segmentation'][entry], -1)

    if 'ghost' in out and apply_deghosting:
        mask_ghost = np.argmax(out['ghost'][entry], axis=1) == 0
        event_data = event_data[mask_ghost]
        #points = points[mask_ghost]
        #if enable_classify_endpoints:
        #    classify_endpoints = classify_endpoints[mask_ghost]
        #mask_ppn = mask_ppn[mask_ghost]
        uresnet_predictions = uresnet_predictions[mask_ghost]
        #scores = scores[mask_ghost]

    scores = scipy.special.softmax(points[:, score_col[0]:score_col[1]], axis=1)
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
    batch_ids  = event_data[:, batch_col]
    for b in np.unique(batch_ids):
        final_points = []
        final_scores = []
        final_types = []
        final_softmax = []
        final_endpoints = []
        batch_index = batch_ids == b
        batch_index2 = ppn_coords[-1][:, 0] == b
        # print(batch_index.shape, batch_index2.shape, mask_ppn.shape, scores.shape)
        mask = ((~(mask_ppn[batch_index2] == 0)).any(axis=1)) & (scores[batch_index2][:, 1] > score_threshold)
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

        ppn_type_predictions = np.argmax(scipy.special.softmax(points[batch_index2][mask][:, type_col[0]:type_col[1]], axis=1), axis=1)
        ppn_type_softmax = scipy.special.softmax(points[batch_index2][mask][:, type_col[0]:type_col[1]], axis=1)
        if enable_classify_endpoints:
            ppn_classify_endpoints = scipy.special.softmax(classify_endpoints[batch_index2][mask], axis=1)
        if enforce_type:
            for c in range(num_classes):
                uresnet_points = uresnet_predictions[batch_index][mask] == c
                ppn_points = ppn_type_softmax[:, c] > type_score_threshold #ppn_type_predictions == c
                if np.count_nonzero(ppn_points) > 0 and np.count_nonzero(uresnet_points) > 0:
                    d = scipy.spatial.distance.cdist(points[batch_index2][mask][ppn_points][:, :3] + event_data[batch_index][mask][ppn_points][:, coords_col[0]:coords_col[1]] + 0.5, event_data[batch_index][mask][uresnet_points][:, coords_col[0]:coords_col[1]])
                    ppn_mask = (d < type_threshold).any(axis=1)
                    final_points.append(points[batch_index2][mask][ppn_points][ppn_mask][:, :3] + 0.5 + event_data[batch_index][mask][ppn_points][ppn_mask][:, coords_col[0]:coords_col[1]])
                    final_scores.append(scores[batch_index2][mask][ppn_points][ppn_mask])
                    final_types.append(ppn_type_predictions[ppn_points][ppn_mask])
                    final_softmax.append(ppn_type_softmax[ppn_points][ppn_mask])
                    if enable_classify_endpoints:
                        final_endpoints.append(ppn_classify_endpoints[ppn_points][ppn_mask])
        else:
            final_points = [points[batch_index2][mask][:, :3] + 0.5 + event_data[batch_index][mask][:, coords_col[0]:coords_col[1]]]
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


def get_track_endpoints_geo(data, f, points_tensor=None, use_numpy=False):
    """
    Compute endpoints of a track-like cluster f
    based on PPN point predictions (coordinates
    and scores) and geometry (voxels farthest
    apart from each other in the cluster).

    If points_tensor is left unspecified, the endpoints will
    be purely based on geometry.

    Input:
    - data is the input data tensor, which can be indexed by f.
    - points_tensor is the output of PPN 'points' (optional)
    - f is a list of voxel indices for voxels that belong to the track.

    Output:
    - array of shape (2, 3) (2 endpoints, 3 coordinates each)
    """
    if use_numpy:
        import scipy
        cdist = scipy.spatial.distance.cdist
        argmax = np.argmax
        sigmoid = scipy.special.expit
        cat = lambda x: np.stack(x, axis=0)
    else:
        cdist = local_cdist
        argmax = torch.argmax
        sigmoid = torch.sigmoid
        cat = torch.cat

    dist_mat = cdist(data[f,1:4], data[f,1:4])
    idx = argmax(dist_mat)
    idxs = int(idx)//len(f), int(idx)%len(f)
    correction0, correction1 = 0.0, 0.0
    if points_tensor is not None:
        scores = sigmoid(points_tensor[f, -1])
        correction0 = points_tensor[f][idxs[0], :3] + \
                      0.5 if scores[idxs[0]] > 0.5 else 0.0
        correction1 = points_tensor[f][idxs[1], :3] + \
                      0.5 if scores[idxs[1]] > 0.5 else 0.0
    end_points =  cat([data[f[idxs[0]],1:4] + correction0,
                        data[f[idxs[1]],1:4] + correction1])
    return end_points
