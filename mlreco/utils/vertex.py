import numpy as np
import scipy
from mlreco.utils.ppn import uresnet_ppn_type_point_selector


def predict_vertex(inter_idx, data_idx, input_data, res,
                    coords_col=(1, 4), primary_label=1,
                    shower_label=0, track_label=1, endpoint_label=0,
                    attaching_threshold=2, inter_threshold=10):
    """
    Heuristic to find the vertex by looking at
    - predicted primary particles within predicted interaction
    - predicted PPN points for these primary particles

    For now, very simple: taking the barycenter of potential candidates.
    """
    clusts = res['inter_particles'][data_idx]
    inter_mask = res['inter_group_pred'][data_idx] == inter_idx
    interaction = clusts[inter_mask]

    # Identify predicted primary particles within the interaction
    primary_particles = np.argmax(res['node_pred_vtx'][data_idx][inter_mask][:, 3:], axis=1) == primary_label
    ppn_candidates, c_candidates = [], []
    ppn = uresnet_ppn_type_point_selector(input_data[data_idx], res,
                                        entry=data_idx,
                                        score_threshold=0.5,
                                        type_threshold=2)
                                        #selection=c)

    # if ppn.shape[0] == 0:
    #     continue
    ppn_voxels = ppn[:, coords_col[0]:coords_col[1]]
    ppn_score = ppn[:, 5]
    #ppn_occupancy = ppn[:, 6]
    ppn_type = ppn[:, 7:12]
    ppn_endpoints = np.argmax(ppn[:, 13:15], axis=1)

    no_delta = ppn_type[:, 3] < 0.5
    ppn = ppn[no_delta]
    ppn_voxels = ppn_voxels[no_delta]
    ppn_score = ppn_score[no_delta]
    ppn_type = ppn_type[no_delta]
    ppn_endpoints = ppn_endpoints[no_delta]
    #ppn = ppn[(ppn_type[:, track_label] > 0.5) | (ppn_type[:, shower_label] > 0.5)]

    all_voxels = input_data[data_idx]
    if 'ghost' in res:
        mask_ghost = np.argmax(res['ghost'][data_idx], axis=1) == 0
        all_voxels = input_data[data_idx][mask_ghost]

    # Look at PPN predictions for each primary particle
    for c_idx, c in enumerate(clusts[inter_mask][primary_particles]):
        c_seg = res['particles_seg'][data_idx][inter_mask][primary_particles][c_idx]
        if c_seg == shower_label:
            # TODO select primary fragment
            shower_primaries = np.argmax(res['shower_node_pred'][data_idx], axis=1) == 0
            shower_primary = None
            for p in res['shower_fragments'][data_idx][shower_primaries]:
                if len(np.intersect1d(c, p)):
                    shower_primary = p
                    break
            if shower_primary is not None:
                c = shower_primary

        # If it is not a shower or track particle, ignore
        if c_seg not in [track_label, shower_label]:
            continue

        d = scipy.spatial.distance.cdist(ppn[:, coords_col[0]:coords_col[1]], all_voxels[c, coords_col[0]:coords_col[1]])
        ppn_candidates.append(ppn[d.min(axis=1) < attaching_threshold])
        # PPN post-processing
        # while only running on the voxels of current primary particle
        # ppn = uresnet_ppn_type_point_selector(input_data[data_idx], res,
        #                                     entry=data_idx,
        #                                     score_threshold=0.5,
        #                                     type_threshold=2,
        #                                     selection=c)
        #
        # if ppn.shape[0] == 0:
        #     continue
        # ppn_voxels = ppn[:, coords_col[0]:coords_col[1]]
        # ppn_score = ppn[:, 5]
        # #ppn_occupancy = ppn[:, 6]
        # ppn_type = ppn[:, 7:12]
        # ppn_endpoints = np.argmax(ppn[:, 13:15], axis=1)
        #
        # no_delta = ppn_type[:, 3] < 0.5
        # ppn = ppn[no_delta]
        # ppn_voxels = ppn_voxels[no_delta]
        # ppn_score = ppn_score[no_delta]
        # ppn_type = ppn_type[no_delta]
        # ppn_endpoints = ppn_endpoints[no_delta]
        #
        # # Pick the PPN points predicted as start point for tracks
        # # Pick any PPN point for showers (primary?)
        # # ppn_candidates.append(ppn[((ppn_type[:, track_label] > 0.5) & (ppn_endpoints == endpoint_label)) | (ppn_type[:, shower_label] > 0.5)])
        # ppn_candidates.append(ppn[((ppn_type[:, track_label] > 0.5) ) | (ppn_type[:, shower_label] > 0.5)])
        c_candidates.append(c)

    if len(ppn_candidates) > 1:
        # print('now', len(ppn_candidates))
        ppn_candidates2 = []
        # For each primary particle, select the PPN predicted point
        # that is closest to other primary particles in the interaction.
        for p_idx, points in enumerate(ppn_candidates):
            #print(points[:, :3])
            if len(points) == 0:
                continue

            d = scipy.spatial.distance.cdist(points[:, coords_col[0]:coords_col[1]], all_voxels[np.hstack([c for idx, c in enumerate(c_candidates) if idx != p_idx])][:, coords_col[0]:coords_col[1]])
            #print(p_idx, d.min(axis=1))
            if d.min() < inter_threshold:
                ppn_candidates2.append(points[np.where(d.min(axis=1) < inter_threshold)[0]])
        ppn_candidates = ppn_candidates2
        #print('now again', len(ppn_candidates))
        #print([x[:, 1:4] for x in ppn_candidates])

    vtx_std, vtx_candidate = [-1, -1, -1], [-1, -1, -1]

    # Take barycenter
    if len(ppn_candidates):
        ppn_candidates = np.concatenate(ppn_candidates, axis=0)
        #print("ppn_candidates", ppn_candidates[:, :4])

        # Refine with dbscan to eliminate ppn candidates that are
        # far away (e.g. middle of a track)
        # ppn_candidates_group = DBSCAN(eps=7, min_samples=1).fit(ppn_candidates[:, :3]).labels_
        # groups, counts = np.unique(ppn_candidates_group, return_counts=True)
        # #print(counts)
        # ppn_candidates = ppn_candidates[ppn_candidates_group == groups[np.argmax(counts)]]

        #print("Selecting %d / %d points after dbscan" % (len(ppn_candidates), len(ppn_candidates_group)))
        # Now take barycenter
        vtx_candidate = np.mean(ppn_candidates[:,coords_col[0]:coords_col[1]], axis=0)
        vtx_std = np.std(ppn_candidates[:, coords_col[0]:coords_col[1]], axis=0)

    return ppn_candidates, c_candidates, vtx_candidate, vtx_std


def get_vertex(kinematics, cluster_label, data_idx, inter_idx,
                vtx_col=9, primary_label=1):
    """
    Getting true vertex for interaction identified by inter_idx

    Look at kinematics label, selecting only primary particles
    within this interaction, and get vertex which occurs the most.

    Note: can there ever be >1 vertex using this method?
    """
    inter_mask = cluster_label[data_idx][:, 7] == inter_idx
    primary_mask = kinematics[data_idx][:, vtx_col+3] == primary_label
    #print(inter_idx, inter_mask.sum(), (inter_mask & primary_mask).sum(), cluster_label[data_idx].shape, kinematics[data_idx].shape)
    mask = inter_mask if (inter_mask & primary_mask).sum() == 0 else inter_mask & primary_mask
    #print(kinematics[data_idx][mask].shape, kinematics[data_idx][inter_mask].shape)
    vtx, counts = np.unique(kinematics[data_idx][mask][:, [vtx_col, vtx_col+1, vtx_col+2]], axis=0, return_counts=True)
    vtx = vtx[np.argmax(counts)]
    return vtx
