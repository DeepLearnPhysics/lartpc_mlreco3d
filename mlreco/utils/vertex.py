#
# Functions related to vertex finding heuristic.
#
import numpy as np
import scipy
from mlreco.utils.ppn import uresnet_ppn_type_point_selector
from mlreco.utils.ppn import get_track_endpoints_geo
from sklearn.decomposition import PCA
from mlreco.utils.gnn.evaluation import primary_assignment
from mlreco.utils.groups import type_labels
from analysis.algorithms.calorimetry import compute_particle_direction


def find_closest_points_of_approach(point1, direction1, point2, direction2):
    """
    Given two lines in 3D space, find the two points that are closest to
    each other on these lines.

    See also https://math.stackexchange.com/a/1993990/391047.

    Parameters
    ==========
    point1: np.ndarray
        Point belonging to first line. Shape (3,)
    direction1: np.ndarray
        Direction defining the first line with `point1`. Shape (3,)
    point2: np.ndarray
        Point belonging to second line. Shape (3,)
    direction2: np.ndarray
        Direction defining the second line with `point2`. Shape (3,)

    Output
    ======
    tuple of np.ndarray
        Two points of approach, tuple with shape ((3,), (3,))
    """
    #print(point1, point2)
    common_normal = np.cross(direction1, direction2)
    A = np.stack([direction1, - direction2, common_normal], axis=1)
    b = point2 - point1
    #print(A.shape, b.shape)
    x = np.linalg.solve(A, b)
    return point1 + x[0] * direction1, point2 + x[1] * direction2


def get_ppn_points_per_particles(input_data, res,
                                primary_particles, primary_particles_seg,
                                data_idx=0, coords_col=(1, 4),
                                attaching_threshold=10,
                                track_label=1,
                                shower_label=0,
                                unwrapped=False,
                                apply_deghosting=True,
                                return_distances=False,
                                min_voxel_count=10,
                                min_track_count=2):
    """
    Get predicted PPN points

    Parameters
    ----------
    input_data: dict
    res: dict
    primary_particles: list
    primary_particles_seg: list
    data_idx: int, default 0
    coords_col: tuple of int, default (1, 4)
    attaching_threshold: float, default 10
        Distance (in voxels) to associate PPN candidates with particles.
    track_label: int, default 1
        Semantic label for track-like particles.
    shower_label: int, default 0
        Semantic label for shower-like particles.
    unwrapped: bool, default False
    apply_deghosting: bool, default True
        Whether PPN post-processing should consider the output already deghosted or not.
    return_distancesL bool, default False
        For tuning studies.
    min_voxel_count: int, default 10
        Any particle with predicted deghosted voxel count below this threshold
        will be ignored.
    min_track_count: int, default 2
        Will ignore shower particles as soon as we have >= min_track_count
        track-like particles in the interaction. This can be set to `None`
        to disable this behavior completely.
    Output
    ------
    - list of N arrays of shape (M_i,f) of M_i PPN candidate points, f corresponds to the number
    of feature in the output of uresnet_ppn_type_point_selector.
    - array of N lists of voxel indices, corresponding to the particles
    whose predicted semantic is track or shower.
    N is the number of particles which are either track or shower (predicted).
    """
    clusts = res['inter_particles'][data_idx]
    ppn_candidates, c_candidates, distances, c_indices = [], [], [], []

    #
    # 1. Run PPN post-processing first to extract PPN candidates
    #
    ppn = uresnet_ppn_type_point_selector(input_data[data_idx], res,
                                        entry=data_idx,
                                        score_threshold=0.5,
                                        type_threshold=2,
                                        unwrapped=unwrapped,
                                        apply_deghosting=apply_deghosting)

    if ppn.shape[0] == 0:
        if return_distances:
            return np.empty((0, 3)), np.empty((0,)), np.empty((0, 3)), np.empty((0, 3))
        else:
            return np.empty((0, 3)), np.empty((0,)), np.empty((0, 3))

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

    #
    # 2. Find shower primary predictions
    #
    if 'shower_node_pred' in res:
        all_primaries = primary_assignment(res['shower_node_pred'][data_idx], res['shower_group_pred'][data_idx])
    else:
        all_primaries = []

    if 'ghost' in res and apply_deghosting:
        mask_ghost = np.argmax(res['ghost'][data_idx], axis=1) == 0
        all_voxels = input_data[data_idx][mask_ghost]

    #
    # 3. Identify predicted particles that we want to keep to
    # find the vertex ("preliminary candidates").
    #
    preliminary_candidates = []
    for c_idx, c in enumerate(primary_particles):
        if len(c) < min_voxel_count: continue
        c_seg = primary_particles_seg[c_idx]
        if c_seg == shower_label:
            shower_primary = []
            # Several shower fragments can be part of the predicted primary
            for p in res['shower_fragments'][data_idx][all_primaries]:
                if len(np.intersect1d(c, p)):
                    shower_primary.append(p)

            if len(shower_primary):
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

    #
    # 4. Look at PPN predictions for each primary particle
    # Keep them when distance is < attaching_threshold.
    # This outputs a list of "ppn_candidates" that we will
    # use to compute the vertex. `ppn_candidates` has the
    # same length as `preliminary_candidates` and lists
    # points for each candidate particle of interest.
    #
    for c_idx, c_seg, c in preliminary_candidates:
        c_candidates.append(c)
        c_indices.append(c_idx)

        d = scipy.spatial.distance.cdist(ppn[:, coords_col[0]:coords_col[1]], all_voxels[c, coords_col[0]:coords_col[1]])
        distances.append(d.min(axis=1))
        good_ppn_predictions = d.min(axis=1) < attaching_threshold

        # If it's a track also use geometry, not just PPN candidates
        if c_seg == track_label:
            end_points = get_track_endpoints_geo(all_voxels, c, use_numpy=True)
            end_points = np.concatenate([end_points, ppn[good_ppn_predictions, coords_col[0]:coords_col[1]]], axis=0)
        else:
            end_points = ppn[good_ppn_predictions, coords_col[0]:coords_col[1]]
        ppn_candidates.append(end_points)

    c_indices = np.array(c_indices)
    if return_distances:
        return ppn_candidates, c_candidates, c_indices, distances
    else:
        return ppn_candidates, c_candidates, c_indices


def predict_vertex(inter_idx, data_idx, input_data, res,
                    coords_col=(1, 4),
                    primary_label=1,
                    shower_label=0,
                    track_label=1,
                    attaching_threshold=10,
                    inter_threshold=20,
                    unwrapped=False,
                    apply_deghosting=True,
                    return_distances=False,
                    other_primaries_threshold=10,
                    other_primaries_gamma_threshold=100,
                    pca_radius=28,
                    min_track_count=2,
                    min_voxel_count=10):
    """
    Heuristic to find the vertex by looking at
    - predicted primary particles within predicted interaction
    - predicted PPN points for these primary particles

    For now, very simple: taking the barycenter of potential candidates.

    Parameters
    ----------
    inter_idx: int
        Predicted interaction index.
    data_idx: int
        Batch entry index.
    input_data: dict
        Input dictionary.
    res: dict
        Output dictionary.
    coords_col: tuple, default (1, 4)
    primary_label: int, default 1
        In GNN predictions, integer tagging predicted primary particles.
    shower_label: int, default 0
        Semantic label for shower-like particles.
    track_label: int, default 1
        Semantic label for track-like particles.
    attaching_threshold: float, default 10
        See `get_ppn_points_per_particles`.
    inter_threshold: float, default 20
        PPN candidates need to minimize difference between distance
        to closest primary and distance of current primary particle
        voxels to closest primary particle.
    unwrapped: bool, default False
        Whether `input_data` and `res` are already unwrapped or not.
    apply_deghosting: bool, default True
        Whether to apply deghosting.
    return_distances: bool, default False
        For tuning studies.
    other_primaries_threshold: float, default 10
        Primaries too far from the other primaries will be ignored.
    other_primaries_gamma_threshold: float, default 100
        Same as previous but for photon-like particles exclusively ($T_B$).
        Can be -1, then the same threshold as all other primaries will be used.
    pca_radius: float, default 28
    min_track_count: int, default 2
        See `get_ppn_points_per_particles`.
    min_voxel_count: int, default 10
        See `get_ppn_points_per_particles`.

    Output
    ------
    np.ndarray
        vtx_candidate with shape (3,)
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

    #
    # Get PPN candidates for vertex, listed per primary particle
    #
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
        if 'input_rescaled' not in res:
            mask_ghost = np.argmax(res['ghost'][data_idx], axis=1) == 0
            all_voxels = input_data[data_idx][mask_ghost]
        else:
            all_voxels = res['input_rescaled'][data_idx]

    # Handle the case where only a single primary is available
    if len(ppn_candidates) == 1:
        particle_seg = res['particles_seg'][data_idx][inter_mask][primary_particles][c_indices[0]]
        end_points = res['particle_node_features'][data_idx][inter_mask][primary_particles][c_indices[0], -9:-3].reshape(-1,3)
        if particle_seg != 1:
            # If there's a single shower object, pick the shower start point
            return end_points[0]
        else:
            # If there's a single track, pick the end point with the lowest local charge density
            voxels = all_voxels[c_candidates[0], coords_col[0]:coords_col[1]]
            dist_mat = scipy.spatial.distance.cdist(end_points, voxels)
            mask = dist_mat < 5
            charges = all_voxels[c_candidates[0],4]
            locald = [np.sum(charges[mask[0]]), np.sum(charges[mask[1]])]
            return end_points[np.argmin(locald)]

    # Handle all other cases
    ppn_candidates2 = []
    directions = []
    distances_others, distances_primaries = [], []
    if len(ppn_candidates) > 1:
        # For each primary particle, select the PPN predicted point
        # that is closest to other primary particles in the interaction.
        for p_idx, points in enumerate(ppn_candidates):
            # No PPN point associated with this primary particle
            if len(points) == 0:
                continue

            current_pid = pid[c_indices[p_idx]]

            # For tuning purpose only
            # compute distance from current particle to primaries that are not the current particle
            if return_distances:
                other_primaries_coordinates = all_voxels[np.hstack([c for idx, c in enumerate(c_candidates) if idx != p_idx])][:, coords_col[0]:coords_col[1]]
                d = scipy.spatial.distance.cdist(all_voxels[c_candidates[p_idx], coords_col[0]:coords_col[1]], other_primaries_coordinates)
                distances_others.append(d.min())

            distance_to_other_primaries = []
            points_to_other_primaries = []
            # Ignore photons if
            # - at least 3 primary particles involved
            # - at least 2 non-photon primary
            use_gamma_threshold = (pid[c_indices] != type_labels[22]).sum() <= 1
            for c_idx, c2 in enumerate(c_candidates):
                if c_idx == p_idx: continue
                # Ignore photons
                # if no_photon_count > 0 and pid[c_indices[c_idx]] == type_labels[22]: continue
                if ~use_gamma_threshold and pid[c_indices[c_idx]] == type_labels[22]: continue
                d2 = scipy.spatial.distance.cdist(all_voxels[c_candidates[p_idx], coords_col[0]:coords_col[1]], all_voxels[c2, coords_col[0]:coords_col[1]])
                distance_to_other_primaries.append(d2.min())
                d3 = scipy.spatial.distance.cdist(points, all_voxels[c2, coords_col[0]:coords_col[1]])
                points_to_other_primaries.append((d3.min(axis=1) - d2.min())[:, None])
            distance_to_other_primaries = np.array(distance_to_other_primaries)
            points_to_other_primaries = np.concatenate(points_to_other_primaries, axis=1)

            if len(distance_to_other_primaries) == 0: continue

            # Select points that minimize
            # - distance from the point to other primaries
            # - distance from current primary to other primaries
            # (this is a heuristic to select points closest to vertex)
            candidate_distances = np.mean(np.abs(points_to_other_primaries), axis=1)
            best_candidate_distance = candidate_distances.min()
            distances_primaries.append(best_candidate_distance)

            #
            # Apply T_B threshold
            #
            use_gamma_threshold = (current_pid == type_labels[22]) or use_gamma_threshold
            if use_gamma_threshold and (other_primaries_gamma_threshold > -1) and (distance_to_other_primaries.min() >= other_primaries_gamma_threshold):
                #print('Skipping photon')
                continue
            elif (~use_gamma_threshold or other_primaries_gamma_threshold == -1) and (other_primaries_threshold > -1) and (distance_to_other_primaries.min() >= other_primaries_threshold):
               #print("Skipping", p_idx, (distance_to_other_primaries >= other_primaries_threshold).sum(), len(distance_to_other_primaries), distance_to_other_primaries)
               continue

            if best_candidate_distance < inter_threshold:
                # Look at all of the points below threshold, pick first one for which we can make a direction
                all_mask = candidate_distances < inter_threshold
                best_idx = np.where(all_mask)[0][candidate_distances[all_mask].argsort()]
                for b in best_idx:
                    d = scipy.spatial.distance.cdist(all_voxels[c_candidates[p_idx], coords_col[0]:coords_col[1]], [points[b]])
                    X = all_voxels[c_candidates[p_idx], coords_col[0]:coords_col[1]][d.reshape((-1,)) < pca_radius]
                    # We need at least 3 (2?) distinct voxels to compute a PCA
                    if X.shape[0] >= 3:
                        ppn_candidates2.append(points[b][None, :])
                        directions.append(pca.fit(X).components_[0][None, :])
                        break

    # Initialize final output
    vtx_std, vtx_candidate = np.array([-1, -1, -1]), np.array([-1, -1, -1])

    # If we have many candidates, we will consider closest points of approach
    # When we are happy with our candidates we take a barycenter.
    if len(ppn_candidates2):
        ppn_candidates2 = np.concatenate(ppn_candidates2, axis=0)
        directions = np.concatenate(directions, axis=0)
        if len(ppn_candidates2) > 1:
            closest_points = []
            # We want to find the closest points of approach between any pair
            # of (vertex candidate, direction of associated primary particle)
            for p1_idx in range(0, len(ppn_candidates2)):
                for p2_idx in range(p1_idx+1, len(ppn_candidates2)):
                    closest_points.extend(find_closest_points_of_approach(ppn_candidates2[p1_idx],
                                                    directions[p1_idx],
                                                    ppn_candidates2[p2_idx],
                                                    directions[p2_idx]))
            closest_points = np.stack(closest_points)
        else:
            closest_points = ppn_candidates2

        # Refine with dbscan to eliminate ppn candidates that are
        # far away (e.g. middle of a track)
        # ppn_candidates_group = DBSCAN(eps=7, min_samples=1).fit(ppn_candidates[:, :3]).labels_
        # groups, counts = np.unique(ppn_candidates_group, return_counts=True)
        # #print(counts)
        # ppn_candidates = ppn_candidates[ppn_candidates_group == groups[np.argmax(counts)]]
        #print("Selecting %d / %d points after dbscan" % (len(ppn_candidates), len(ppn_candidates_group)))
        # This part here was giving a divide by zero RuntimeWarning. Best to avoid if possible

        # Now take barycenter
        vtx_candidate = np.mean(closest_points, axis=0)
        vtx_std = np.std(closest_points, axis=0)

    if return_distances:
        return ppn_candidates2, c_candidates, vtx_candidate, vtx_std, ppn_candidates, distances, distances_others, distances_primaries
    else:
        return vtx_candidate


def get_vertex(kinematics, cluster_label, data_idx, inter_idx,
                vtx_col=9, primary_label=1):
    """
    Getting true vertex for interaction identified by inter_idx

    Look at kinematics label, selecting only primary particles
    within this interaction, and get vertex which occurs the most.

    Parameters
    ----------
    kinematics: list of np.ndarray
        Kinematics labels.
    cluster_label: list of np.ndarray
        Cluster labels.
    data_idx: int
        Which entry we are looking at (labels).
    inter_idx: int
        The true interaction id for which we want the vertex.
    vtx_col: int, default 9
        First column of vertex coordinates in the kinematics labels.
        Coordinates columns go from vtx_col to vtx_col+2.
    primary_label: int, default 1
        What integer tags primary particles in kinematics labels
        ("primary particles" ~ particles coming out of the vertex).

    Output
    ------
    np.ndarray
        True vertex coordinates. Shape (3,)
    """
    inter_mask = cluster_label[data_idx][:, 7] == inter_idx
    primary_mask = kinematics[data_idx][:, vtx_col+3] == primary_label
    mask = inter_mask if (inter_mask & primary_mask).sum() == 0 else inter_mask & primary_mask
    vtx, counts = np.unique(kinematics[data_idx][mask][:, [vtx_col, vtx_col+1, vtx_col+2]], axis=0, return_counts=True)
    vtx = vtx[np.argmax(counts)]
    return vtx
