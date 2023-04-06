import numpy as np
import numba as nb
from scipy.spatial.distance import cdist

from mlreco.utils.gnn.cluster import cluster_direction
from analysis.post_processing import post_processing
from mlreco.utils.globals import COORD_COLS


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