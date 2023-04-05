import numpy as np
import numba as nb
from collections import OrderedDict
from turtle import up
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

from analysis.classes import Particle
from analysis.classes import TruthParticle
from analysis.algorithms.point_matching import get_track_endpoints_max_dist
# from lartpc_mlreco3d.analysis.algorithms.arxiv.calorimetry import compute_track_dedx, get_particle_direction


def attach_prefix(update_dict, prefix):
    if prefix is None:
        return update_dict
    out = OrderedDict({})

    for key, val in update_dict.items():
        new_key = "{}_".format(prefix) + str(key)
        out[new_key] = val

    return out


def get_mparticles_from_minteractions(int_matches):
    '''
    Given list of Tuple[(Truth)Interaction, (Truth)Interaction], 
    return list of particle matches Tuple[TruthParticle, Particle]. 

    If no match, (Truth)Particle is replaced with None.
    '''

    matched_particles, match_counts = [], []

    for m in int_matches:
        ia1, ia2 = m[0], m[1]
        num_parts_1, num_parts_2 = -1, -1
        if m[0] is not None:
            num_parts_1 = len(m[0].particles)
        if m[1] is not None:
            num_parts_2 = len(m[1].particles)
        if num_parts_1 <= num_parts_2:
            ia1, ia2 = m[0], m[1]
        else:
            ia1, ia2 = m[1], m[0]
            
        for p in ia2.particles:
            if len(p.match) == 0:
                if type(p) is Particle:
                    matched_particles.append((None, p))
                    match_counts.append(-1)
                else:
                    matched_particles.append((p, None))
                    match_counts.append(-1)
            for match_id in p.match:
                if type(p) is Particle:
                    matched_particles.append((ia1[match_id], p))
                else:
                    matched_particles.append((p, ia1[match_id]))
                match_counts.append(p._match_counts[match_id])
    return matched_particles, np.array(match_counts)

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