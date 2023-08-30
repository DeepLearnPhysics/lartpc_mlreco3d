import numpy as np
import numba as nb

from . import numba_local as nbl
from .globals import TRACK_SHP, INTER_COL, PGRP_COL, VTX_COLS


def get_vertex(start_points,
               end_points,
               directions,
               semantics,
               anchor_vertex=True,
               touching_threshold=2.0,
               return_mode=False):
    '''
    Reconstruct the vertex of an individual interaction.

    Parameters
    ----------
    start_points : np.ndarray
        (P, 3) Particle start points
    end_points : np.ndarray
        (P, 3) Particle end points
    directions : np.ndarray
        (P, 3) Particle directions
    semantics : np.ndarray
        (P) : particle semantic type
    anchor_vertex : bool, default True
        If true, anchor the candidate vertex to particle objects,
        with the expection of interactions only composed of showers.
    touching_threshold : float, default 2.0
        Maximum distance for two particle points to be considered touching
    '''
    # If there is no particle: return default values
    if not len(start_points):
        if return_mode:
            return np.full(3, -np.inf), 'no_particle'
        return np.full(3, -np.inf)

    # If there is only one particle: choose the start point
    if len(start_points) == 1:
        if return_mode:
            return start_points[0], 'single_start'
        return start_points[0]

    # If this more than one particle, anchor the vertex to a particle point
    track_mask = semantics == TRACK_SHP
    track_ids  = np.where(track_mask)[0]
    if anchor_vertex:
        # If there is a unique point where >=2 particles meet, pick it. Include
        # track start and end points, to not rely on direction predictions
        vertices = get_confluence_points(start_points, end_points, touching_threshold)
        if len(vertices) == 1:
            if return_mode:
                return vertices[0], 'confluence_nodir'
            return vertices[0]

        # If there is more than one option, restrict track end points to the
        # predicted start points (relies on direction prediction), check again
        if len(vertices) > 1 and len(track_ids):
            vertices = get_confluence_points(start_points,
                    touching_threshold=touching_threshold)
            if len(vertices) == 1:
                if return_mode:
                    return vertices[0], 'confluence_dir'
                return vertices[0]

        # If there are no obvious confluence points, but there is a single
        # track and N showers, pick the track end-point which minimizes the
        # cosine distance between the normalized vertex-to-point vectors
        # and the shower direction estimates.
        if len(track_ids) == 1:
            candidates = np.vstack([start_points[track_mask], end_points[track_mask]])
            losses     = angular_loss(candidates, start_points[~track_mask], directions[~track_mask])
            if return_mode:
                return candidates[np.argmin(losses)], 'confluence_track_showers'
            return candidates[np.argmin(losses)]

        # If all else fails on track groups, simply pick the longest
        # track, take its starting point and hope for the best...
        if len(track_ids):
            track_lengths = np.linalg.norm(end_points[track_mask]-start_points[track_mask], axis=-1)
            if return_mode:
                return start_points[track_mask][np.argmax(track_lengths)], 'track_length'
            return start_points[track_mask][np.argmax(track_lengths)]

    # If there is only showers (or the vertex is not anchored): find the point which
    # minimizes the sum of distances from the vertex w.r.t. to all direction lines.
    # TODO: Find the point which minimizes the angular difference between the
    # vertex-to-point vector and the shower direction estimates (much better).
    if return_mode:
        return get_pseudovertex(start_points, directions), 'pseudo_vertex'
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
def get_confluence_points(start_points: nb.float32[:,:],
                          end_points: nb.float32[:,:] = None,
                          touching_threshold: nb.float32 = 2.0) -> nb.types.List(nb.float32[:]):
    '''
    Find the points where multiple particles touch.

    Parameters
    ----------
    start_points : np.ndarray
        (P, 3) Particle start points
    end_points : np.ndarray, optional
        (P, 3) Particle end points
    touching_threshold : float, default 2.0
        Maximum distance for two particle points to be considered touching

    Returns
    -------
    List[np.ndarray]
        List of vertices that correspond to the confluence points
    '''
    # Create a particle-to-particle distance matrix
    n_part   = len(start_points)
    dist_mat = np.zeros((n_part, n_part), dtype=start_points.dtype)
    end_mat  = np.zeros((n_part, n_part), dtype=np.int32)
    if end_points is None:
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
    for i in range(n_part):
        walk_mat[i,i] = np.max(walk_mat[i][np.arange(n_part) != i])

    # Find cycles to build particle groups and confluence points (vertices)
    leftover  = np.ones(n_part, dtype=np.bool_)
    max_walks = nbl.max(walk_mat, axis=1)
    vertices  = nb.typed.List.empty_list(np.empty(0, dtype=start_points.dtype))
    while np.any(leftover):
        # Find the longest available cycle (must be at least 2 particles)
        left_ids = np.where(leftover)[0]
        max_id   = left_ids[np.argmax(max_walks[leftover])]
        max_walk = max_walks[max_id]
        if max_walk < 2:
            break

        # Form the particle group that make up the cycle
        leftover[walk_mat[max_id] == max_walk] = False
        group  = np.where(walk_mat[max_id] == max_walk)[0]

        # Take the barycenter of the touching particle ends as the vertex
        if end_points is None:
            vertices.append(nbl.mean(start_points[group], axis=0))
        else:
            vertex = np.zeros(3, dtype=start_points.dtype)
            for i, t in enumerate(group):
                end_id = np.argmax(np.bincount(end_mat[t][group][np.arange(len(group)) != i]))
                vertex += start_points[t]/len(group) if not end_id else end_points[t]/len(group)
            vertices.append(vertex)

    return vertices


@nb.njit(cache=True)
def get_pseudovertex(start_points: nb.float32[:,:],
                     directions:  nb.float32[:,:],
                     dim: int = 3) -> nb.float32[:]:
    '''
    Finds the vertex which minimizes the total distance
    from itself to all the lines defined by the start points
    of particles and their directions.

    Parameters
    ----------
    start_points : np.ndarray
        (P, 3) Particle start points
    directions : np.ndarray
        (P, 3) Particle directions
    dim : int
        Number of dimensions
    '''
    assert len(start_points),\
            'Cannot reconstruct pseudovertex without points'

    if len(start_points) == 1:
        return start_points[0]

    pseudovtx = np.zeros((dim, ), dtype=start_points.dtype)
    S = np.zeros((dim, dim), dtype=start_points.dtype)
    C = np.zeros((dim, ), dtype=start_points.dtype)

    for p, d in zip(start_points, directions):
        S += (np.outer(d, d) - np.eye(dim, dtype=start_points.dtype))
        C += (np.outer(d, d) - np.eye(dim, dtype=start_points.dtype)) @ np.ascontiguousarray(p)

    pseudovtx = np.linalg.pinv(S) @ C

    return pseudovtx


def get_truth_vertex(cluster_label,
                     data_idx,
                     inter_idx,
                     primary_label=1):
    """
    Getting true vertex for interaction identified by inter_idx

    Look at cluster labels, selecting only primary particles
    within this interaction, and get vertex which occurs the most.

    Parameters
    ----------
    cluster_label: list of np.ndarray
        Cluster labels.
    data_idx: int
        Which entry we are looking at (labels).
    inter_idx: int
        The true interaction id for which we want the vertex.
    primary_label: int, default 1
        What integer tags primary particles in kinematics labels
        ("primary particles" ~ particles coming out of the vertex).

    Output
    ------
    np.ndarray
        True vertex coordinates. Shape (3,)
    """
    inter_mask = cluster_label[data_idx][:, INTER_COL] == inter_idx
    primary_mask = cluster_label[data_idx][:, PGRP_COL] == primary_label
    mask = inter_mask if (inter_mask & primary_mask).sum() == 0 else inter_mask & primary_mask
    vtx, counts = np.unique(cluster_label[data_idx][mask][:, [VTX_COLS[0], VTX_COLS[1], VTX_COLS[2]]], axis=0, return_counts=True)
    vtx = vtx[np.argmax(counts)]
    return vtx
