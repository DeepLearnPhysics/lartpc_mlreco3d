import sys
from itertools import combinations

import numpy as np
import numba as nb
from scipy.spatial.distance import cdist

from mlreco.utils.gnn.cluster import cluster_direction
from analysis.post_processing import post_processing
from mlreco.utils.globals import COORD_COLS


@post_processing(data_capture=[],
                 result_capture=['interactions'])
def reconstruct_vertex(data_dict, result_dict,
                       include_semantics=[0,1],
                       use_primaries=True,
                       r1=5.0,
                       r2=10.0):
    
    for ia in result_dict['interactions']:
        
        candidates = []
        
        if use_primaries:
            particles = [p for p in ia.particles \
                if p.is_primary and (p.semantic_type in include_semantics)]
        else:
            particles = [p for p in ia.particles \
                if p.semantic_type in include_semantics]
            
        cand_1 = get_adjacent_startpoint_candidate(particles, r1)
        cand_2 = get_track_shower_candidate(particles, r2=r2)
        cand_3 = get_pseudovertex_candidate(particles, dim=3)
        
        if len(cand_1) > 0:
            candidates.append(cand_1)
        if len(cand_2) > 0:
            candidates.append(cand_2)
        if len(cand_3) > 0:
            candidates.append(cand_3)
            
        if len(candidates) > 0:
            candidates = np.vstack(candidates)
            vertex = np.mean(candidates, axis=0)
            ia.vertex = vertex
            
    return {}

@nb.njit(cache=True)
def point_to_line_distance_(p1, p2, v2):
    dist = np.sqrt(np.sum(np.cross(v2, (p2 - p1))**2)+1e-8)
    return dist

@nb.njit(cache=True)
def point_to_line_distance(P1, P2, V2):
    dist = np.zeros((P1.shape[0], P2.shape[0]))
    for i, p1 in enumerate(P1):
        for j, p2 in enumerate(P2):
            d = point_to_line_distance_(p1, p2, V2[j])
            dist[i, j] = d
    return dist

def get_adjacent_startpoint_candidate(particles,
                                      r1=5.0):
    candidates = []
    startpoints = []
    for p in particles:
        startpoints.append(p.start_point)
    if len(startpoints) == 0:
        return np.array(candidates)
    startpoints = np.vstack(startpoints)
    dist = cdist(startpoints, startpoints)
    dist += -np.eye(dist.shape[0])
    idx, idy = np.where((dist < r1) & (dist > 0))
    
    # Keep track of duplicate pairs
    duplicates = []
    # Append barycenter of two touching points within radius r1 to candidates
    for ix, iy in zip(idx, idy):
        center = (startpoints[ix] + startpoints[iy]) / 2.0
        if not((ix, iy) in duplicates or (iy, ix) in duplicates):
            candidates.append(center)
            duplicates.append((ix, iy))
            
    candidates = np.array(candidates)
    return candidates

def get_track_shower_candidate(particles,
                               r2=5.0):
    candidates, track_starts = [], []
    shower_starts, shower_dirs = [], []
    
    for p in particles:
        if p.semantic_type == 0 and len(p.points) > 0:
            shower_starts.append(p.start_point)
            shower_dirs.append(p.start_dir)
        if p.semantic_type == 1:
            track_starts.append(p.start_point)
            
    if len(shower_starts) == 0 or len(track_starts) == 0:
        return np.array(candidates)
    
    shower_dirs = np.vstack(shower_dirs)
    shower_starts = np.vstack(shower_starts)
    track_starts = np.vstack(track_starts)
    
    dist = point_to_line_distance(track_starts, shower_starts, shower_dirs)
    idx, idy = np.where(dist < r2)
    for ix, iy in zip(idx, idy):
        candidates.append(track_starts[ix])
        
    candidates = np.array(candidates)
    return candidates

def get_pseudovertex_candidate(particles, dim=3):
    
    if len(particles) < 2:
        return np.array([])
    
    pseudovtx = np.zeros((dim, ))
    S = np.zeros((dim, dim))
    C = np.zeros((dim, ))

    assert len(particles) >= 2
        
    for p in particles:
        startpt = p.start_point
        vec = p.start_dir
        w = 1.0
        S += w * (np.outer(vec, vec) - np.eye(dim))
        C += w * (np.outer(vec, vec) - np.eye(dim)) @ startpt

    pseudovtx = np.linalg.pinv(S) @ C
    return pseudovtx



# ---------------------------DEPRECATED--------------------------------

@post_processing(data_capture=[],
                 result_capture=['particle_clusts',
                                 'particle_seg',
                                 'particle_start_points',
                                 'particle_group_pred',
                                 'particle_node_pred_vtx',
                                 'input_rescaled',
                                 'interactions'],
                 result_capture_optional=['particle_dirs'])
def reconstruct_vertex_deprecated(data_dict, result_dict,
                       mode='all',
                       include_semantics=[0,1],
                       use_primaries=True,
                       r1=5.0,
                       r2=10.0):
    """Post processing for reconstructing interaction vertex.
    
    """

    particles              = result_dict['particle_clusts']
    particle_group_pred    = result_dict['particle_group_pred']
    primary_ids            = np.argmax(result_dict['particle_node_pred_vtx'], axis=1)
    particle_seg           = result_dict['particle_seg']
    input_coords           = result_dict['input_rescaled'][:, COORD_COLS]
    startpoints            = result_dict['particle_start_points'][:, COORD_COLS]

    # Optional
    particle_dirs          = result_dict.get('particle_dirs', None)

    assert len(primary_ids) == len(particles)

    if particle_dirs is not None:
        assert len(particle_dirs) == len(particles)
    
    vertices = []
    interaction_ids = []
    # Loop over interactions:
    for ia in np.unique(particle_group_pred):
        interaction_ids.append(ia)
        # Default bogus value for no vertex
        candidates = []
        vertex = np.array([-sys.maxsize, -sys.maxsize, -sys.maxsize])

        int_mask = particle_group_pred == ia
        particles_int = []
        startpoints_int = []
        particle_seg_int = []
        primaries_int = []

        dirs_int = None
        if particle_dirs is not None:
            dirs_int = [p for i, p in enumerate(particle_dirs[int_mask]) \
                         if particle_seg[int_mask][i] in include_semantics]

        for i, primary_id in enumerate(primary_ids[int_mask]):
            if particle_seg[int_mask][i] not in include_semantics:
                continue
            if not use_primaries or primary_id == 1:
                particles_int.append(particles[int_mask][i])
                particle_seg_int.append(particle_seg[int_mask][i])
                primaries_int.append(primary_id)
                startpoints_int.append(startpoints[int_mask][i])
                if particle_dirs is not None:
                    dirs_int.append(particle_dirs[int_mask][i])
        
        if len(startpoints_int) > 0:
            startpoints_int = np.vstack(startpoints_int)
            if len(startpoints_int) == 1:
                vertex = startpoints_int.squeeze()
            else:
                # Gather vertex candidates from each algorithm
                vertices_1 = get_centroid_adj_pairs(startpoints_int, r1=r1)
                vertices_2 = get_track_shower_poca(startpoints_int,
                                                particles_int,
                                                particle_seg_int,
                                                input_coords,
                                                r2=r2,
                                                particle_dirs=dirs_int)
                if len(particles_int) >= 2:
                    pseudovertex = compute_pseudovertex(particles_int, 
                                                        startpoints_int, 
                                                        input_coords, 
                                                        dim=3, 
                                                        particle_dirs=dirs_int)
                else:
                    pseudovertex = np.array([])

                if vertices_1.shape[0] > 0:
                    candidates.append(vertices_1)
                if vertices_2.shape[0] > 0:
                    candidates.append(vertices_2)
                if len(candidates) > 0:
                    candidates = np.vstack(candidates)
                    vertex = np.mean(candidates, axis=0)
        vertices.append(vertex)

    if len(vertices) > 0:
        vertices = np.vstack(vertices)
    else:
        msg = "Vertex reconstructor saw an image with no interactions, "\
        "maybe there's an image with no voxels?"
        raise RuntimeWarning(msg)
        vertices = np.array([])

    interaction_ids = np.array(interaction_ids).reshape(-1, 1)

    vertices = {key: val for key, val in zip(interaction_ids.squeeze(), vertices)}

    for i, ia in enumerate(result_dict['interactions']):
        ia.vertex = vertices[ia.id]

    return {}


def get_centroid_adj_pairs(particle_start_points, 
                           r1=5.0):
    '''
    From N x 3 array of N particle startpoint coordinates, find
    two points which touch each other within r1, and return the
    barycenter of such pairs. 
    '''
    candidates = []

    startpoints = []
    for i, pts in enumerate(particle_start_points):
        startpoints.append(pts)
    if len(startpoints) == 0:
        return np.array(candidates)
    startpoints = np.vstack(startpoints)
    dist = cdist(startpoints, startpoints)
    dist += -np.eye(dist.shape[0])
    idx, idy = np.where( (dist < r1) & (dist > 0))
    # Keep track of duplicate pairs
    duplicates = []
    # Append barycenter of two touching points within radius r1 to candidates
    for ix, iy in zip(idx, idy):
        center = (startpoints[ix] + startpoints[iy]) / 2.0
        if not((ix, iy) in duplicates or (iy, ix) in duplicates):
            candidates.append(center)
            duplicates.append((ix, iy))
    candidates = np.array(candidates)
    return candidates


def get_track_shower_poca(particle_start_points,
                          particle_clusts,
                          particle_seg, 
                          input_coords,
                          r2=5.0,
                          particle_dirs=None):
    '''
    From list of particles, find startpoints of track particles that lie
    within r2 distance away from the closest line defined by a shower
    direction vector. 
    '''
    
    candidates = []
    
    track_starts = []
    shower_starts, shower_dirs = [], []
    for i, mask in enumerate(particle_clusts):
        pts = input_coords[mask]
        if particle_seg[i] == 0 and len(pts) > 0:
            if particle_dirs is not None:
                vec = particle_dirs[i]
            else:
                vec = cluster_direction(pts, 
                        particle_start_points[i], 
                        optimize=True)
            shower_dirs.append(vec)
            shower_starts.append(
                particle_start_points[i])
        if particle_seg[i] == 1:
            track_starts.append(
                particle_start_points[i])

    shower_dirs   = np.array(shower_dirs)
    shower_starts = np.array(shower_starts)
    track_starts  = np.array(track_starts)

    assert len(shower_dirs) == len(shower_starts)

    if len(shower_dirs) == 0 or len(track_starts) == 0:
        return np.array(candidates)

    dist = point_to_line_distance(track_starts, shower_starts, shower_dirs)
    idx, idy = np.where(dist < r2)
    for ix, iy in zip(idx, idy):
        candidates.append(track_starts[ix])
        
    candidates = np.array(candidates)
    return candidates


def compute_pseudovertex(particle_clusts,
                         particle_start_points,
                         input_coords, 
                         dim=3,
                         particle_dirs=None):
    """
    Given a set of particles, compute the vertex by the following method:

    1) Estimate the direction of each particle
    2) Using infinite lines defined by the direction and the startpoint of
    each particle, compute the point of closest approach. 
    3) Solve the least squares optimization problem. 

    The least squares problem in this case has an analytic solution
    which could be solved by matrix pseudoinversion. 
    """
    pseudovtx = np.zeros((dim, ))
    S = np.zeros((dim, dim))
    C = np.zeros((dim, ))

    assert len(particle_clusts) >= 2
        
    for i, mask in enumerate(particle_clusts):
        pts = input_coords[mask]
        startpt = particle_start_points[i]
        if particle_dirs is not None:
            vec = particle_dirs[i]
        else:
            vec = cluster_direction(pts, startpt, optimize=True)
        w = 1.0
        S += w * (np.outer(vec, vec) - np.eye(dim))
        C += w * (np.outer(vec, vec) - np.eye(dim)) @ startpt

    pseudovtx = np.linalg.pinv(S) @ C
    return pseudovtx


def prune_vertex_candidates(candidates, pseudovtx, r=30):
    dist = np.linalg.norm(candidates - pseudovtx.reshape(1, -1), axis=1)
    pruned = candidates[dist < r]
    return pruned