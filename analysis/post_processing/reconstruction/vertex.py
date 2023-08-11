import numpy as np
import numba as nb

from mlreco.utils.globals import SHOWR_SHP, TRACK_SHP
from mlreco.utils import numba_local as nbl

from analysis.post_processing import post_processing


@post_processing(data_capture=[],
                 result_capture=['interactions'],
                 result_capture_optional=['truth_interactions'])
def reconstruct_vertex(data_dict, result_dict,
                       include_semantics = [SHOWR_SHP, TRACK_SHP],
                       use_primaries = True,
                       update_primaries = False,
                       anchor_vertex = True,
                       touching_threshold = 2.0,
                       angle_threshold = 0.3,
                       run_mode = 'reco'):
    '''
    Post-processor which reconstructs one vertex for each
    interaction in the provided list. It modifies the input list
    of interactions in place.

    Parameters
    ----------
    interactions : List[Interaction]
        List of reconstructed particle interactions
    truth_interactions : List[TruthInteractions], optional
        List of true interactions
    include_semantics : List[int]
        List of semantic classes to consider for vertex reconstruction
    use_primaries : bool, default True
        If true, only considers primary particles to reconstruct the vertex
    update_primaries : bool, default False
        Use the reconstructed vertex to update primaries
    anchor_vertex : bool, default True
        If true, anchor the candidate vertex to particle objects,
        with the expection of interactions only composed of showers
    touching_threshold : float, default 2 cm
        Maximum distance for two track points to be considered touching
    angle_threshold : float, default 0.3 radians
        Maximum angle between the vertex-to-start-point vector and a
        shower direction to consider that a shower originated from the vertex
    run_mode : str
        One of `reco`, `truth`, `both` to tell which interaction types to
        apply this algorithm to.
    '''
    # Loop over interactions
    if run_mode not in ['reco', 'truth', 'both']:
        raise ValueError('`run_mode` must be either `reco`, `truth` or `both`')

    if run_mode in ['reco', 'both']:
        for ia in result_dict['interactions']:
            reconstruct_vertex_single(ia, include_semantics, use_primaries, update_primaries,
                    anchor_vertex, touching_threshold, angle_threshold)

    if run_mode in ['truth', 'both']:
        assert 'truth_interactions' in result_dict,\
                'Need truth interaction to apply vertex reconstruction to them'
        for ia in result_dict['truth_interactions']:
            reconstruct_vertex_single(ia, include_semantics, use_primaries, False,
                    anchor_vertex, touching_threshold, angle_threshold)

    return {}


def reconstruct_vertex_single(interaction,
                              include_semantics,
                              use_primaries,
                              update_primaries,
                              anchor_vertex,
                              touching_threshold,
                              angle_threshold):

    '''
    Post-processor which reconstructs one vertex for each
    interaction in the provided list. It modifies the input list
    of interactions in place.

    Parameters
    ----------
    interaction : List[Interaction, TruthInteraction]
        Reconstructed/truth interaction object
    include_semantics : List[int]
        List of semantic classes to consider for vertex reconstruction
    use_primaries : bool
        If true, only considers primary particles to reconstruct the vertex
    update_primaries : bool
        Use the reconstructed vertex to update primaries
    anchor_vertex : bool
        If true, anchor the candidate vertex to particle objects,
        with the expection of interactions only composed of showers
    touching_threshold : float
        Maximum distance for two track points to be considered touching
    angle_threshold : float
        Maximum angle between the vertex-to-start-point vector and a
        shower direction to consider that a shower originated from the vertex
    '''
    # Selected the set of particles to use as a basis for vertex prediction
    if use_primaries:
        particles = [p for p in interaction.particles \
            if p.is_primary and (p.semantic_type in include_semantics)]
    if not use_primaries or not len(particles):
        particles = [p for p in interaction.particles \
            if p.semantic_type in include_semantics]
    if not len(particles):
        return

    # Reconstruct the vertex for this interaction
    vtx, vtx_mode = reconstruct_vertex_dispatch(particles, anchor_vertex, touching_threshold, return_mode=True)
    interaction.vertex = vtx
    interaction.vertex_mode = vtx_mode

    # If requested, update primaries on the basis of the predicted vertex
    if update_primaries:
        for p in interaction.particles:
            if p.semantic_type not in [SHOWR_SHP, TRACK_SHP]:
                p.is_primary = False
            elif np.linalg.norm(p.start_point - interaction.vertex) < touching_threshold:
                p.is_primary = True
            elif p.semantic_type == SHOWR_SHP and np.dot(p.start_point, interaction.vertex) < angle_threshold:
                p.is_primary = True


def reconstruct_vertex_dispatch(particles, anchor_vertex, touching_threshold,
                                return_mode=False):
    '''
    Reconstruct the vertex of an individual interaction.

    Parameters
    ----------
    particles : List[Particle]
        List of candidate particles to find the vertex from
    anchor_vertex : bool
        If true, anchor the candidate vertex to particle objects,
        with the expection of interactions only composed of showers.
    touching_threshold : float
        Maximum distance for two track points to be considered touching
    '''
    # Collapse particle objects to a set of start, end points and directions
    start_points = np.vstack([p.start_point for p in particles]).astype(np.float32)
    end_points   = np.vstack([p.end_point for p in particles]).astype(np.float32)
    directions   = np.vstack([p.start_dir for p in particles]).astype(np.float32)
    semantics    = np.array([p.semantic_type for p in particles], dtype=np.int32)

    # If there is only one particle: choose the start point
    if len(particles) == 1:
        if return_mode:
            return start_points[0], 'single_start'
        return start_points[0]

    # If there is only one track and N showers: pick the track end-point
    # which minimizes the cosine distance between the normalized
    # vertex-to-point vectors and the shower direction estimates.
    track_mask = semantics == TRACK_SHP
    track_ids  = np.where(track_mask)[0]
    if anchor_vertex and len(track_ids) == 1:
        candidates = np.vstack([start_points[track_mask], end_points[track_mask]])
        losses     = angular_loss(candidates, start_points[~track_mask], directions[~track_mask])
        if return_mode:
            return candidates[np.argmin(losses)], '1trackNshower'
        return candidates[np.argmin(losses)]

    # If there are >=2 tracks, try multiple things
    if anchor_vertex and len(track_ids) > 1:
        # Step 1: if there is a unique point where >=1 track meet, pick it
        # as the vertex (no need for direction predictions here)
        vertices = get_track_confluence(start_points[track_mask], end_points[track_mask], touching_threshold)
        if len(vertices) == 1:
            if return_mode:
                return vertices[0], 'multiTrack1'
            return vertices[0]

        # Step 2: if there is a unique *start* point where >=1 track start,
        # pick it as the vertex.
        vertices = get_track_confluence(start_points[track_mask], touching_threshold=touching_threshold)
        if len(vertices) == 1:
            if return_mode:
                return vertices[0], 'multiTrack2'
            return vertices[0]

        # Step 3: if all else fails on track groups, simply pick the longest
        # track, take its starting point and hope for the best...
        track_lengths = np.linalg.norm(end_points[track_mask]-start_points[track_mask], axis=-1)
        if return_mode:
            return start_points[track_mask][np.argmax(track_lengths)], 'multiTrack3'
        return start_points[track_mask][np.argmax(track_lengths)]

    # If there is only showers (or the vertex is not anchored): find the point which
    # minimizes the sum of distances from the vertex w.r.t. to all direction lines.
    # TODO: Find the point which minimizes the angular difference between the
    # vertex-to-point vector and the shower direction estimates (much better).
    if return_mode:
        return get_pseudovertex(start_points, directions), 'pseudovtx'
    return get_pseudovertex(start_points, directions)


@nb.njit(cache=True)
def angular_loss(candidates: nb.float32[:,:],
                 points: nb.float32[:,:],
                 directions: nb.float32[:,:],
                 use_cos: bool = True) -> nb.float32:
    '''
    Computes the angular/cosine distance between vectors that
    join candidate points to the start points of particles and their
    respective direction estimates. Values are normalized between
    0 (perfect fit) and 1 (complete disagreement).

    Parameters
    ----------
    candidates : np.ndarray
        (C, 3) Vertex coordinates
    points : np.ndarray
        (P, 3) Particle start points
    directions : np.ndarray
        (P, 3) Particle directions
    use_cos : bool
        Whether or not to use the cosine as a metric

    Returns
    -------
    np.ndarray
        (C) Loss for each of the candidates
    '''
    n_c = len(candidates)
    losses = np.empty(n_c, dtype=np.float32)
    for i, c in enumerate(candidates):
        loss = 0.
        for p, d in zip(points, directions):
            v  = p - c
            v /= np.linalg.norm(v)
            if use_cos:
                loss += (1. - np.sum(v*d))/2/n_c
            else:
                loss += np.arccos(np.sum(v*d))/np.pi/n_c

        losses[i] = loss

    return losses


@nb.njit(cache=True)
def get_track_confluence(start_points: nb.float32[:,:],
                         end_points: nb.float32[:,:] = np.empty((0,3), dtype=np.float32),
                         touching_threshold: nb.float32 = 5.0) -> nb.types.List(nb.float32[:]):
    '''
    Find the points where multiple tracks touch.

    Parameters
    ----------
    start_points : np.ndarray
        (P, 3) Particle start points
    end_points : np.ndarray, optional
        (P, 3) Particle end points
    touching_threshold : float, default 5 cm
        Maximum distance for two track points to be considered touching

    Returns
    -------
    List[np.ndarray]
        List of vertices that correspond to the confluence points
    '''
    # Create a track-to-track distance matrix
    n_tracks = len(start_points)
    dist_mat = np.zeros((n_tracks, n_tracks), dtype=np.float32)
    end_mat  = np.zeros((n_tracks, n_tracks), dtype=np.int32)
    if not len(end_points):
        for i, si in enumerate(start_points):
            for j, sj in enumerate(start_points):
                if j > i:
                    dist_mat[i,j] = np.linalg.norm(sj - si)
                if j < i:
                    dist_mat[i,j] = dist_mat[j,i]
    else:
        for i, (si, ei) in enumerate(zip(start_points, end_points)):
            pointsi = np.vstack((si, ei))
            for j, (sj, ej) in enumerate(zip(start_points, end_points)):
                if j > i:
                    pointsj = np.vstack((sj, ej))
                    submat = nbl.cdist(pointsi, pointsj)
                    mini, minj = np.argmin(submat)//2, np.argmin(submat)%2
                    dist_mat[i,j] = submat[mini, minj]
                    end_mat[i,j], end_mat[j,i] = mini, minj
                if j < i:
                    dist_mat[i,j] = dist_mat[j,i]

    # Convert distance matrix to an adjacency matrix, compute
    # the square graphic to find the number of walks
    adj_mat  = (dist_mat < touching_threshold).astype(np.float32) # @ does not like integers
    walk_mat = adj_mat @ adj_mat
    for i in range(n_tracks):
        walk_mat[i,i] = np.max(walk_mat[i][np.arange(n_tracks) != i])

    # Find cycles to build track groups and confluence points (vertices)
    leftover  = np.ones(n_tracks, dtype=np.bool_)
    max_walks = nbl.max(walk_mat, axis=1)
    vertices  = nb.typed.List.empty_list(np.empty(0, dtype=np.float32))
    while np.any(leftover):
        # Find the longest available cycle (must be at least 2 tracks)
        left_ids = np.where(leftover)[0]
        max_id   = left_ids[np.argmax(max_walks[leftover])]
        max_walk = max_walks[max_id]
        if max_walk < 2:
            break

        # Form the track group that make up the cycle
        leftover[walk_mat[max_id] == max_walk] = False
        group  = np.where(walk_mat[max_id] == max_walk)[0]

        # Take the barycenter of the touching track ends as the vertex
        if not len(end_points):
            vertices.append(nbl.mean(start_points[group], axis=0))
        else:
            vertex = np.zeros(3, dtype=np.float32)
            for i, t in enumerate(group):
                end_id = np.argmax(np.bincount(end_mat[t][group][np.arange(len(group)) != i]))
                vertex += start_points[t]/len(group) if not end_id else end_points[t]/len(group)
            vertices.append(vertex)

    return vertices


@nb.njit(cache=True)
def get_pseudovertex(points: nb.float32[:,:],
                     directions:  nb.float32[:,:],
                     dim: int = 3) -> nb.float32[:]:
    '''
    Finds the vertex which minimizes the total distance
    from itself to all the lines defined by the start points
    of particles and their directions.

    Parameters
    ----------
    points : np.ndarray
        (P, 3) Particle start points
    directions : np.ndarray
        (P, 3) Particle directions
    dim : int
        Number of dimensions
    '''
    assert len(points),\
            'Cannot reconstruct pseudovertex without points'

    if len(points) == 1:
        return points[0]

    pseudovtx = np.zeros((dim, ), dtype=np.float32)
    S = np.zeros((dim, dim), dtype=np.float32)
    C = np.zeros((dim, ), dtype=np.float32)

    for p, d in zip(points, directions):
        S += (np.outer(d, d) - np.eye(dim, dtype=np.float32))
        C += (np.outer(d, d) - np.eye(dim, dtype=np.float32)) @ np.ascontiguousarray(p)

    pseudovtx = np.linalg.pinv(S) @ C

    return pseudovtx
