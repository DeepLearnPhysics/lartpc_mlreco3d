import numpy as np
import scipy
from mlreco.utils.ppn import uresnet_ppn_type_point_selector
from mlreco.utils.ppn import get_track_endpoints_geo
from sklearn.decomposition import PCA
from mlreco.utils.gnn.evaluation import primary_assignment
from mlreco.utils.groups import type_labels

def find_closest_points_of_approach(point1, direction1, point2, direction2):
    """
    See also
    https://math.stackexchange.com/a/1993990/391047
    """
    print(point1, point2)
    common_normal = np.cross(direction1, direction2)
    A = np.stack([direction1, - direction2, common_normal], axis=1)
    b = point2 - point1
    print(A.shape, b.shape)
    x = np.linalg.solve(A, b)
    return point1 + x[0] * direction1, point2 + x[1] * direction2

def get_ppn_points_per_particles(input_data, res,
                                primary_particles, primary_particles_seg,
                                data_idx=0, coords_col=(1, 4),
                                attaching_threshold=2,
                                track_label=1,
                                shower_label=0,
                                unwrapped=False,
                                apply_deghosting=True,
                                return_distances=False,
                                min_voxel_count=20,
                                min_track_count=None):
    """
    Get predicted PPN points

    Returns:
    - list of N arrays of shape (M_i,f) of M_i PPN candidate points, f corresponds to the number
    of feature in the output of uresnet_ppn_type_point_selector.
    - array of N lists of voxel indices, corresponding to the particles
    whose predicted semantic is track or shower.
    N is the number of particles which are either track or shower (predicted).
    """
    clusts = res['inter_particles'][data_idx]
    ppn_candidates, c_candidates, distances, c_indices = [], [], [], []
    ppn = uresnet_ppn_type_point_selector(input_data[data_idx], res,
                                        entry=data_idx,
                                        score_threshold=0.5,
                                        type_threshold=2,
                                        unwrapped=unwrapped,
                                        apply_deghosting=apply_deghosting)
                                        #selection=c)

    if ppn.shape[0] == 0:
        return np.empty((0, 3)), np.empty((0,)), np.empty((0, 3)), np.empty((0, 3))

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
    ppn = ppn[(ppn_type[:, track_label] > 0.5) | (ppn_type[:, shower_label] > 0.5)]
    #print('\n', ppn_voxels, '\n')
    all_voxels = input_data[data_idx]

    all_primaries = primary_assignment(res['shower_node_pred'][data_idx], res['shower_group_pred'][data_idx])

    if 'ghost' in res and apply_deghosting:
        mask_ghost = np.argmax(res['ghost'][data_idx], axis=1) == 0
        all_voxels = input_data[data_idx][mask_ghost]

    # Look at PPN predictions for each primary particle
    preliminary_candidates = []
    for c_idx, c in enumerate(primary_particles):
        if len(c) < min_voxel_count: continue
        c_seg = primary_particles_seg[c_idx]
        if c_seg == shower_label:
            # TODO select primary fragment
            #shower_primaries = np.argmax(res['shower_node_pred'][data_idx], axis=1) == 0
            shower_primary = []

            # Several shower fragments can be part of the predicted primary
            for p in res['shower_fragments'][data_idx][all_primaries]:
                if len(np.intersect1d(c, p)):
                    shower_primary.append(p)

            if len(shower_primary):
                #print('Selecting shower primary', len(c), len(np.hstack(shower_primary)))
                c = np.hstack(shower_primary)

        # If it is not a shower or track particle, ignore
        if c_seg not in [track_label, shower_label]:
            continue

        preliminary_candidates.append((c_idx, c_seg, c))

    # If we have >= 2 tracks, just ignore shower particles
    # to find the vertex
    if min_track_count is not None:
        assert type(min_track_count) == int
        track_count = np.sum([int(c_seg == track_label) for _, c_seg, _ in preliminary_candidates])
        if track_count >= min_track_count:
            preliminary_candidates = [(c_idx, c_seg, c) for c_idx, c_seg, c in preliminary_candidates if c_seg == track_label]

    for c_idx, c_seg, c in preliminary_candidates:
        c_candidates.append(c)
        c_indices.append(c_idx)

        # if ppn.shape[0] == 0:
        #     distances.append(-1)
        #     ppn_candidates.append(np.empty(0, 6))
        #     continue
        d = scipy.spatial.distance.cdist(ppn[:, coords_col[0]:coords_col[1]], all_voxels[c, coords_col[0]:coords_col[1]])
        distances.append(d.min(axis=1))
        #print('distances', len(c), d.min(axis=1), np.isclose(all_voxels[c, coords_col[0]:coords_col[1]], [337,121,745]).all(axis=1).any(), '\n')
        #print('Particle ', c_seg, np.any(d.min(axis=1) < attaching_threshold), d.min(axis=1))
        #ppn_candidates.append(ppn[np.unravel_index(np.argmin(d), d.shape)[0]]) #< attaching_threshold])
        good_ppn_predictions = d.min(axis=1) < attaching_threshold

        # If it's a track also use geometry
        if c_seg == track_label:
            end_points = get_track_endpoints_geo(all_voxels, c, use_numpy=True)
            print(end_points.shape)
            end_points = np.concatenate([end_points, ppn[good_ppn_predictions, coords_col[0]:coords_col[1]]], axis=0)
        else:
            end_points = ppn[good_ppn_predictions, coords_col[0]:coords_col[1]]
        ppn_candidates.append(end_points)
        print("getting ppn", c_idx, c_seg, "found points = ", len(end_points), d.min(axis=1).min())

    c_indices = np.array(c_indices)
    if return_distances:
        return ppn_candidates, c_candidates, c_indices, distances
    else:
        return ppn_candidates, c_candidates, c_indices


def predict_vertex(inter_idx, data_idx, input_data, res,
                    coords_col=(1, 4), primary_label=1,
                    shower_label=0, track_label=1, endpoint_label=0,
                    attaching_threshold=2, inter_threshold=10, unwrapped=False,
                    apply_deghosting=True, return_distances=False,
                    other_primaries_threshold=30,
                    other_primaries_gamma_threshold=-1,
                    fraction_bad_primaries=0.6,
                    pca_radius=21,
                    min_track_count=None,
                    min_voxel_count=20):
    """
    Heuristic to find the vertex by looking at
    - predicted primary particles within predicted interaction
    - predicted PPN points for these primary particles

    For now, very simple: taking the barycenter of potential candidates.
    """
    if other_primaries_gamma_threshold < 0:
        other_primaries_gamma_threshold = other_primaries_threshold

    pca = PCA(n_components=3)
    clusts = res['inter_particles'][data_idx]
    if inter_idx not in set(res['inter_group_pred'][data_idx]):
        raise ValueError("Interaction ID: {} does not exist for data entry : {}.\n"\
            " Available Interactions: {}".format(inter_idx, data_idx, str(np.unique(res['inter_group_pred'][data_idx]))))
    inter_mask = res['inter_group_pred'][data_idx] == inter_idx
    interaction = clusts[inter_mask]

    # Identify predicted primary particles within the interaction
    primary_particles = np.argmax(res['node_pred_vtx'][data_idx][inter_mask][:, 3:], axis=1) == primary_label

    # Identify PID among primary particles
    pid = np.argmax(res['node_pred_type'][data_idx][inter_mask][primary_particles], axis=1)
    photon_label = type_labels[22]

    out = get_ppn_points_per_particles(input_data, res,
                                    clusts[inter_mask][primary_particles],
                                    res['particles_seg'][data_idx][inter_mask][primary_particles],
                                    data_idx=data_idx,
                                    attaching_threshold=attaching_threshold,
                                    track_label=track_label,
                                    shower_label=shower_label,
                                    coords_col=coords_col,
                                    unwrapped=unwrapped,
                                    apply_deghosting=apply_deghosting,
                                    return_distances=return_distances,
                                    min_track_count=min_track_count,
                                    min_voxel_count=min_voxel_count)
    if return_distances:
        ppn_candidates, c_candidates, c_indices, distances = out
    else:
        ppn_candidates, c_candidates, c_indices = out

    all_voxels = input_data[data_idx]
    if 'ghost' in res and apply_deghosting:
        mask_ghost = np.argmax(res['ghost'][data_idx], axis=1) == 0
        all_voxels = input_data[data_idx][mask_ghost]

    ppn_candidates2 = []
    directions = []
    distances_others, distances_primaries = [], []
    if len(ppn_candidates) > 1:
        #print('\t Candidates are', [x[:, 1:4] for x in ppn_candidates])
        # For each primary particle, select the PPN predicted point
        # that is closest to other primary particles in the interaction.
        for p_idx, points in enumerate(ppn_candidates):
            # No PPN point associated with this primary particle
            print('Looking at ', c_indices[p_idx], len(c_candidates[p_idx]))

            if len(points) == 0:
                print('no points')
                continue

            current_pid = pid[c_indices[p_idx]]
            # compute distance to primaries that are not the current particle
            other_primaries_coordinates = all_voxels[np.hstack([c for idx, c in enumerate(c_candidates) if idx != p_idx])][:, coords_col[0]:coords_col[1]]
            #d = scipy.spatial.distance.cdist(points[:, coords_col[0]:coords_col[1]], other_primaries_coordinates)
            d = scipy.spatial.distance.cdist(all_voxels[c_candidates[p_idx], coords_col[0]:coords_col[1]], other_primaries_coordinates)
            #print(p_idx, d.min(axis=1))
            #distances_others.extend(d.min(axis=1))
            distances_others.append(d.min())

            distance_to_other_primaries = []
            points_to_other_primaries = []
            # Ignore photons if
            # - at least 3 primary particles involved
            # - at least 2 non-photon primary
            # ignore_photon = primary_particles.sum() > 2 and (pid[c_indices[c_indices != p_idx]] == type_labels[22]).sum() > 0
            # use_gamma_threshold = (primary_particles.sum() == 2 and (pid == type_labels[22]).sum() >= 1)
            use_gamma_threshold = (pid[c_indices] != type_labels[22]).sum() <= 1
            for c_idx, c2 in enumerate(c_candidates):
                if c_idx == p_idx: continue
                print(c_idx, p_idx, use_gamma_threshold, pid[c_indices[c_idx]] == type_labels[22])
                # Ignore photons
                # if no_photon_count > 0 and pid[c_indices[c_idx]] == type_labels[22]: continue
                if ~use_gamma_threshold and pid[c_indices[c_idx]] == type_labels[22]: continue
                d2 = scipy.spatial.distance.cdist(all_voxels[c_candidates[p_idx], coords_col[0]:coords_col[1]], all_voxels[c2, coords_col[0]:coords_col[1]])
                distance_to_other_primaries.append(d2.min())
                d3 = scipy.spatial.distance.cdist(points, all_voxels[c2, coords_col[0]:coords_col[1]])
                points_to_other_primaries.append((d3.min(axis=1) - d2.min())[:, None])
            distance_to_other_primaries = np.array(distance_to_other_primaries)
            points_to_other_primaries = np.concatenate(points_to_other_primaries, axis=1)
            #print(points_to_other_primaries.shape, len(points))
            print(points_to_other_primaries)
            if len(distance_to_other_primaries) == 0: continue

            #d2 = scipy.spatial.distance.cdist(all_voxels[c_candidates[p_idx], coords_col[0]:coords_col[1]], other_primaries_coordinates)
            #distances_primaries.append(d2.min())
            #if d.min() < inter_threshold:
            #    ppn_candidates2.append(points[np.where(d.min(axis=1) < inter_threshold)[0]])
            candidate_distances = np.mean(np.abs(points_to_other_primaries), axis=1)
            best_candidate_distance = candidate_distances.min()
            distances_primaries.append(best_candidate_distance)

            # print(p_idx, len(c_candidates[p_idx]), best_candidate_distance, np.sum(np.abs(points_to_other_primaries), axis=1).argmin(), distance_to_other_primaries)
            # print(p_idx, points[:, 1:4])

            #
            # Apply T_B threshold
            #
            use_gamma_threshold = (current_pid == type_labels[22]) or use_gamma_threshold
            if use_gamma_threshold and (other_primaries_gamma_threshold > -1) and (distance_to_other_primaries.min() >= other_primaries_gamma_threshold):
                print('Skipping photon')
                continue
            elif (~use_gamma_threshold or other_primaries_gamma_threshold == -1) and (other_primaries_threshold > -1) and (distance_to_other_primaries.min() >= other_primaries_threshold):
               print("Skipping", p_idx, (distance_to_other_primaries >= other_primaries_threshold).sum(), len(distance_to_other_primaries), distance_to_other_primaries)
               continue

            print("best candidate distance = ", best_candidate_distance)

            if best_candidate_distance < inter_threshold:
                # FIXME pick one or all of the points below threshold ?
                ppn_candidates2.append(points[candidate_distances.argmin()][None, :])
                d = scipy.spatial.distance.cdist(all_voxels[c_candidates[p_idx], coords_col[0]:coords_col[1]], [points[candidate_distances.argmin()]])
                X = all_voxels[c_candidates[p_idx], coords_col[0]:coords_col[1]][d.reshape((-1,)) < pca_radius]
                directions.append(pca.fit(X).components_[0][None, :])
                print("candidate ", ppn_candidates2[-1], np.sum(np.abs(points_to_other_primaries), axis=1).min())

        #ppn_candidates = ppn_candidates2
        #print('now again', len(ppn_candidates))
        #print([x[:, 1:4] for x in ppn_candidates])

    vtx_std, vtx_candidate = [-1, -1, -1], [-1, -1, -1]

    # Take barycenter
    # Here ppn_candidates can be [array([], shape=(0, 15), dtype=float64)]
    # mpvmpr test set event (200-300)
    if len(ppn_candidates2):
        ppn_candidates2 = np.concatenate(ppn_candidates2, axis=0)
        directions = np.concatenate(directions, axis=0)
        print("ppn_candidates", ppn_candidates2[:, :4], ppn_candidates2.shape)

        if len(ppn_candidates2) > 1:
            closest_points = []
            for p1_idx in range(0, len(ppn_candidates2)):
                for p2_idx in range(p1_idx+1, len(ppn_candidates2)):
                    closest_points.extend(find_closest_points_of_approach(ppn_candidates2[p1_idx],
                                                    directions[p1_idx],
                                                    ppn_candidates2[p2_idx],
                                                    directions[p2_idx]))
            closest_points = np.stack(closest_points)
        else:
            closest_points = ppn_candidates2
        print('closest points', closest_points)
        # Refine with dbscan to eliminate ppn candidates that are
        # far away (e.g. middle of a track)
        # ppn_candidates_group = DBSCAN(eps=7, min_samples=1).fit(ppn_candidates[:, :3]).labels_
        # groups, counts = np.unique(ppn_candidates_group, return_counts=True)
        # #print(counts)
        # ppn_candidates = ppn_candidates[ppn_candidates_group == groups[np.argmax(counts)]]

        #print("Selecting %d / %d points after dbscan" % (len(ppn_candidates), len(ppn_candidates_group)))
        # Now take barycenter

        # This part here was giving a divide by zero RuntimeWarning. Best to avoid if possible

        vtx_candidate = np.mean(closest_points, axis=0)
        vtx_std = np.std(closest_points, axis=0)

    if return_distances:
        return ppn_candidates2, c_candidates, vtx_candidate, vtx_std, ppn_candidates, distances, distances_others, distances_primaries
    else:
        return ppn_candidates2, c_candidates, vtx_candidate, vtx_std, ppn_candidates


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
    mask = inter_mask if (inter_mask & primary_mask).sum() == 0 else inter_mask & primary_mask
    vtx, counts = np.unique(kinematics[data_idx][mask][:, [vtx_col, vtx_col+1, vtx_col+2]], axis=0, return_counts=True)
    vtx = vtx[np.argmax(counts)]
    return vtx
