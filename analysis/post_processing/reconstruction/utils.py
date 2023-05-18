import numpy as np
import numba as nb
from scipy.spatial.distance import cdist

from mlreco.utils.gnn.cluster import cluster_direction
from analysis.post_processing import post_processing
from mlreco.utils.globals import COORD_COLS


def _estimate_point_endpointness(query_point : np.ndarray, 
                                point_cloud : np.ndarray, 
                                r : float = 10.0):
    """Estimate endpointness score for a single point.
    
    The endpointness score of a point is computed by the following method:
    1) For a given point, compute the unit vector between that point and
    all other points within a radius r.
    2) Compute the "averaged direction vector" by averaging (and normalizing)
    the result from 1)
    3) Compute the average cosine score between the averaged unit vector
    and the result from 1)
    
    NOTE: the cosine score is defined as (1 + cos) / 2 and lies between [0,1].

    Parameters
    ----------
    query_point : np.ndarray
        Point to compute the endpointness score
    point_cloud : np.ndarray
        Point cloud that contains the query point.
    r : float, optional
        The query radius, by default 10.0

    Returns
    -------
    score : float
        The cosine accuracy endpointness score of the point.
    """
    displacement = point_cloud - query_point
    dist = np.linalg.norm(displacement, axis=1)
    mask = np.logical_and(dist < r, dist > 1e-8)
    if not mask.any():
        return 0.0
        # raise ValueError(f"Query point {str(query_point)} has no adjacent pixels within radius = {r}!")
    vec = displacement[mask] / dist[mask].reshape(-1, 1)
    v_avg = vec.sum(axis=0)
    v_avg = v_avg / np.linalg.norm(v_avg)
    cosine_accuracy = (1 + np.dot(vec, v_avg)) / 2.0
    score = cosine_accuracy.mean()
    return score

def estimate_endpointness(X : np.ndarray, 
                          Y : np.ndarray, 
                          r : float = 10.0):
    """Estimate the endpointness score for all points in X, with respect to
    points in Y.

    Parameters
    ----------
    X : np.ndarray
        (N x D) array of query points
    Y : np.ndarray
        (M X D) point cloud containing the query points X.
    r : float, optional
        Query radius, by default 10.0

    Returns
    -------
    out : np.ndarray
        (N, ) array of endpointness scores for points in X.
    """
    out = np.zeros(X.shape[0])
    for ix, vx in enumerate(X):
        out[ix] = _estimate_point_endpointness(vx, Y, r=r)
    return out

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