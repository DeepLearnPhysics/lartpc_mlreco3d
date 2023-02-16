import numpy as np
import numba as nb
from scipy.spatial.distance import cdist
from analysis.algorithms.calorimetry import compute_particle_direction
from mlreco.utils.utils import func_timer
from analysis.classes.Interaction import Interaction


@nb.njit(cache=True)
def point_to_line_distance_(p1, p2, v2):
    dist = np.linalg.norm(np.cross(v2, (p2 - p1)))
    return dist


@nb.njit(cache=True)
def point_to_line_distance(P1, P2, V2):
    dist = np.zeros((P1.shape[0], P2.shape[0]))
    for i, p1 in enumerate(P1):
        for j, p2 in enumerate(P2):
            d = point_to_line_distance_(p1, p2, V2[j])
            dist[i, j] = d
    return dist


def get_centroid_adj_pairs(particles, r1=5.0, return_annot=False):
    '''
    From N x 3 array of N particle startpoint coordinates, find
    two points which touch each other within r1, and return the
    barycenter of such pairs. 
    '''
    candidates = []
    vp_startpoints = np.vstack([p.startpoint for p in particles])
    vp_labels = np.array([p.id for p in particles])
    dist = cdist(vp_startpoints, vp_startpoints)
    dist += -np.eye(dist.shape[0])
    idx, idy = np.where( (dist < r1) & (dist > 0))
    # Keep track of duplicate pairs
    duplicates = []
    # Append barycenter of two touching points within radius r1 to candidates
    for ix, iy in zip(idx, idy):
        center = (vp_startpoints[ix] + vp_startpoints[iy]) / 2.0
        if not((ix, iy) in duplicates or (iy, ix) in duplicates):
            if return_annot:
                candidates.append((center, str((vp_labels[ix], vp_labels[iy]))))
            else:
                candidates.append(center)
            duplicates.append((ix, iy))
    return candidates


def get_track_shower_poca(particles, return_annot=False, start_segment_radius=10, r2=5.0):
    '''
    From list of particles, find startpoints of track particles that lie
    within r2 distance away from the closest line defined by a shower
    direction vector. 
    '''
    
    candidates = []
    
    track_ids, shower_ids = np.array([p.id for p in particles if p.semantic_type == 1]), []
    track_starts = np.array([p.startpoint for p in particles if p.semantic_type == 1])
    shower_starts, shower_dirs = [], []
    for p in particles:
        vec = compute_particle_direction(p, start_segment_radius=start_segment_radius)
        if p.semantic_type == 0 and (vec != -1).all():
            shower_dirs.append(vec)
            shower_starts.append(p.startpoint)
            shower_ids.append(p.id)

    assert len(shower_starts) == len(shower_dirs)
    assert len(shower_dirs) == len(shower_ids)

    shower_dirs = np.array(shower_dirs)
    shower_starts = np.array(shower_starts)
    shower_ids = np.array(shower_ids)

    if len(track_ids) == 0 or len(shower_ids) == 0:
        return []

    dist = point_to_line_distance(track_starts, shower_starts, shower_dirs)
    idx, idy = np.where(dist < r2)
    for ix, iy in zip(idx, idy):
        if return_annot:
            candidates.append((track_starts[ix], str((track_ids[ix], shower_ids[iy]))))
        else:
            candidates.append(track_starts[ix])
        
    return candidates


def compute_vertex_matrix_inversion(particles, 
                                    dim=3, 
                                    use_primaries=True, 
                                    weight=False, 
                                    var_sigma=0.05):
    """
    Given a set of particles, compute the vertex by the following method:

    1) Estimate the direction of each particle
    2) Using infinite lines defined by the direction and the startpoint of
    each particle, compute the point of closest approach. 
    3) Solve the least squares optimization problem. 

    The least squares problem in this case has an analytic solution
    which could be solved by matrix pseudoinversion. 

    Obviously, we require at least two particles. 

    Parameters
    ----------
    particles: List of Particle
    dim: dimension of image (2D, 3D)
    use_primaries: option to only consider primaries in defining lines
    weight: if True, the function will use the information from PCA's 
    percentage of explained variance to weigh each contribution to the cost. 
    This is to avoid ill defined directions to affect the solution.

    Returns
    -------
    np.ndarray
        Shape (3,)
    """
    pseudovtx = np.zeros((dim, ))

    if use_primaries:
        particles = [p for p in particles if (p.is_primary and p.startpoint is not None)]
        
    if len(particles) < 2:
        return np.array([-1, -1, -1])
        
    S = np.zeros((dim, dim))
    C = np.zeros((dim, ))
        
    for p in particles:
        vec, var = compute_particle_direction(p, return_explained_variance=True)
        w = 1.0
        if weight:
            w = np.exp(-(var[0] - 1)**2 / (2.0 * var_sigma)**2)
        S += w * (np.outer(vec, vec) - np.eye(dim))
        C += w * (np.outer(vec, vec) - np.eye(dim)) @ p.startpoint
    # print(S, C)
    pseudovtx = np.linalg.pinv(S) @ C
    return pseudovtx


def compute_vertex_candidates(particles, 
                              use_primaries=True, 
                              valid_semantic_types=[0,1], 
                              r1=5.0, 
                              r2=5.0,
                              return_annot=False):
    
    candidates = []
    
    # Exclude unwanted particles
    valid_particles = []
    for p in particles:
        check = p.is_primary or (not use_primaries)
        if check and (p.semantic_type in valid_semantic_types):
            valid_particles.append(p)
            
    if len(valid_particles) == 0:
        return [], None
    elif len(valid_particles) == 1:
        startpoint = p.startpoint if p.startpoint is not None else -np.ones(3)
        return [startpoint], None
    else:
        # 1. Select two startpoints within dist r1
        candidates.extend(get_centroid_adj_pairs(valid_particles, 
                                                 r1=r1, 
                                                 return_annot=return_annot))
        # 2. Select a track start point which is close
        #    to a line defined by shower direction
        candidates.extend(get_track_shower_poca(valid_particles, 
                                                r2=r2, 
                                                return_annot=return_annot))
        # 3. Select POCA of all primary tracks and showers
        pseudovtx = compute_vertex_matrix_inversion(valid_particles, 
                                                  dim=3, 
                                                  use_primaries=True, 
                                                  weight=True)
        # if not (pseudovtx < 0).all():
        #     candidates.append(pseudovtx)
            
        return candidates, pseudovtx


def prune_vertex_candidates(candidates, pseudovtx, r=30):
    dist = np.linalg.norm(candidates - pseudovtx.reshape(1, -1), axis=1)
    pruned = candidates[dist < r]
    return pruned


def estimate_vertex(particles, 
                    use_primaries=True, 
                    r_adj=10, 
                    r_poca=10, 
                    r_pvtx=30,
                    prune_candidates=False, 
                    return_candidate_count=False,
                    mode='all'):

    # Exclude unwanted particles
    valid_particles = []
    for p in particles:
        check = p.is_primary or (not use_primaries)
        if check and (p.semantic_type in [0,1]):
            valid_particles.append(p)
            
    if len(valid_particles) == 0:
        candidates = []
    elif len(valid_particles) == 1:
        startpoint = p.startpoint if p.startpoint is not None else -np.ones(3)
        candidates = [startpoint]
    else:
        if mode == 'adj':
            candidates = get_centroid_adj_pairs(valid_particles, r1=r_adj)
        elif mode == 'track_shower_pair':
            candidates = get_track_shower_poca(valid_particles, r2=r_poca)
        elif mode == 'all':
            candidates, pseudovtx = compute_vertex_candidates(valid_particles, 
                                                            use_primaries=True, 
                                                            r1=r_adj, 
                                                            r2=r_poca)
        else:
            raise ValueError("Mode {} for vertex selection not supported!".format(mode))

    out = np.array([-1, -1, -1])

    if len(candidates) == 0:
        out = np.array([-1, -1, -1])
    elif len(candidates) == 1:
        out = candidates[0]
    else:
        candidates = np.vstack(candidates)
        if mode == 'all' and prune_candidates:
            pruned = prune_vertex_candidates(candidates, pseudovtx, r=r_pvtx)
        else:
            pruned = candidates
        if pruned.shape[0] > 0:
            out = pruned.mean(axis=0)
        else:
            out = candidates.mean(axis=0)
    
    if return_candidate_count:
        return out, len(candidates)
    else:
        return out

def correct_primary_with_vertex(ia, r_adj=10, r_bt=10, start_segment_radius=10):
    assert type(ia) is Interaction
    if ia.vertex is not None and (ia.vertex > 0).all():
        for p in ia.particles:
            if p.semantic_type == 1:
                dist = np.linalg.norm(p.startpoint - ia.vertex)
                print(p.id, p.is_primary, p.semantic_type, dist)
                if dist < r_adj:
                    p.is_primary = True
                else:
                    p.is_primary = False
            if p.semantic_type == 0:
                vec = compute_particle_direction(p, start_segment_radius=start_segment_radius)
                dist = point_to_line_distance_(ia.vertex, p.startpoint, vec)
                if np.linalg.norm(p.startpoint - ia.vertex) < r_adj:
                    p.is_primary = True
                elif dist < r_bt:
                    p.is_primary = True
                else:
                    p.is_primary = False