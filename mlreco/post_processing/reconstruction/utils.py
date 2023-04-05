import numpy as np
import numba as nb
from scipy.spatial.distance import cdist

from mlreco.utils.gnn.cluster import cluster_direction
from mlreco.post_processing import post_processing
from mlreco.utils.globals import COORD_COLS


@post_processing(data_capture=[],
                 result_capture=['particle_clusts',
                                 'input_rescaled',
                                 'particle_start_points'])
def reconstruct_direction(data_dict, result_dict,
                          max_dist=-1, optimize=True):
    """Post processing for reconstructing particle direction.
    
    """
    startpts  = result_dict['particle_start_points'][:, COORD_COLS[0]:COORD_COLS[-1]+1]
    coords    = result_dict['input_rescaled'][:, COORD_COLS[0]:COORD_COLS[-1]+1]
    particles = result_dict['particle_clusts']

    update_dict = {}

    particle_dirs = []
    for i, mask in enumerate(particles):
        pts = coords[mask]
        vec = cluster_direction(pts, startpts[i], 
                                max_dist=max_dist, optimize=optimize)
        particle_dirs.append(vec)
    if len(particle_dirs) > 0:
        particle_dirs = np.vstack(particle_dirs)
        update_dict['particle_dirs'] = particle_dirs
    else:
        update_dict['particle_dirs'] = np.array(particle_dirs)
    return update_dict


@nb.njit
def closest_distance_two_lines(a0, u0, a1, u1):
    '''
    a0, u0: point (a0) and unit vector (u0) defining line 1
    a1, u1: point (a1) and unit vector (u1) defining line 2
    '''
    cross = np.cross(u0, u1)
    # if the cross product is zero, the lines are parallel
    if np.linalg.norm(cross) == 0:
        # use any point on line A and project it onto line B
        t = np.dot(a1 - a0, u1)
        a = a1 + t * u1 # projected point
        return np.linalg.norm(a0 - a)
    else:
        # use the formula from https://en.wikipedia.org/wiki/Skew_lines#Distance
        t = np.dot(np.cross(a1 - a0, u1), cross) / np.linalg.norm(cross)**2
        # closest point on line A to line B
        p = a0 + t * u0
        # closest point on line B to line A
        q = p - cross * np.dot(p - a1, cross) / np.linalg.norm(cross)**2
        return np.linalg.norm(p - q) # distance between p and q