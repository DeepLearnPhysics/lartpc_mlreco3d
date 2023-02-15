from typing import List
import numpy as np
import pandas as pd

from scipy.spatial.distance import cdist
from scipy.special import expit
from ..classes.particle import Particle

def match_points_to_particles(ppn_points : np.ndarray,
                              particles : List[Particle],
                              semantic_type=None, ppn_distance_threshold=2):
    """Function for matching ppn points to particles.

    For each particle, match ppn_points that have hausdorff distance
    less than <threshold> and inplace update particle.ppn_candidates

    If semantic_type is set to a class integer value,
    points will be matched to particles with the same
    predicted semantic type.

    Parameters
    ----------
    ppn_points : (N x 4 np.array)
        PPN point array with (coords, point_type)
    particles : list of <Particle> objects
        List of particles for which to match ppn points.
    semantic_type: int
        If set to an integer, only match ppn points with prescribed
        semantic type
    ppn_distance_threshold: int or float
        Maximum distance required to assign ppn point to particle.

    Returns
    -------
        None (operation is in-place)
    """
    if semantic_type is not None:
        ppn_points_type = ppn_points[ppn_points[:, 5] == semantic_type]
    else:
        ppn_points_type = ppn_points
        # TODO: Fix semantic type ppn selection

    ppn_coords = ppn_points_type[:, :3]
    for particle in particles:
        dist = cdist(ppn_coords, particle.points)
        matches = ppn_points_type[dist.min(axis=1) < ppn_distance_threshold]
        particle.ppn_candidates = matches

# Deprecated
def get_track_endpoints(particle : Particle, verbose=False):
    """Function for getting endpoints of tracks (DEPRECATED)

    Using ppn_candiates attached to <Particle>, get two
    endpoints of tracks by farthest distance from the track's
    spatial centroid.

    Parameters
    ----------
    particle : <Particle> object
        Track particle for which to obtain endpoint coordinates
    verbose : bool
        If set to True, output print message indicating whether
        particle has no or only one PPN candidate.

    Returns
    -------
    endpoints : (2, 3) np.array
        Xyz coordinates of two endpoints predicted or manually found
        by network.
    """
    if verbose:
        print("Found {} PPN candidate points for particle {}".format(
            particle.ppn_candidates.shape[0], particle.id))
    if particle.semantic_type != 1:
        raise AttributeError(
            "Particle {} has type {}, can only give"\
            " endpoints to tracks!".format(particle.id,
                                           particle.semantic_type))
    if particle.ppn_candidates.shape[0] == 0:
        if verbose:
            print("Particle {} has no PPN candidates!"\
                " Running brute-force endpoint finder...".format(particle.id))
        startpoint, endpoint = get_track_endpoints_max_dist(particle)
    elif particle.ppn_candidates.shape[0] == 1:
        if verbose:
            print("Particle {} has only one PPN candidate!"\
                " Running brute-force endpoint finder...".format(particle.id))
        startpoint, endpoint = get_track_endpoints_max_dist(particle)
    else:
        centroid = particle.points.mean(axis=0)
        ppn_coordinates = particle.ppn_candidates[:, :3]
        dist = cdist(centroid.reshape(1, -1), ppn_coordinates).squeeze()
        endpt_inds = dist.argsort()[-2:]
        endpoints = particle.ppn_candidates[endpt_inds]
        particle.endpoints = endpoints
    assert endpoints.shape[0] == 2
    return endpoints


def get_track_endpoints_max_dist(particle):
    """Helper function for getting track endpoints.

    Computes track endpoints without ppn predictions by
    selecting the farthest two points from the coordinate centroid.

    Parameters
    ----------
    particle : <Particle> object

    Returns
    -------
    endpoints : (2, 3) np.array
        Xyz coordinates of two endpoints predicted or manually found
        by network.
    """
    coords = particle.points
    dist = cdist(coords, coords)
    pts = particle.points[np.where(dist == dist.max())[0]]
    return pts[0], pts[1]


# Deprecated
def get_shower_startpoint(particle : Particle, verbose=False):
    """Function for getting startpoint of EM showers. (DEPRECATED)

    Using ppn_candiates attached to <Particle>, get one
    startpoint of shower by nearest hausdorff distance.

    Parameters
    ----------
    particle : <Particle> object
        Track particle for which to obtain endpoint coordinates
    verbose : bool
        If set to True, output print message indicating whether
        particle has no or only one PPN candidate.

    Returns
    -------

    endpoints : (2, 3) np.array
        Xyz coordinates of two endpoints predicted or manually found
        by network.
    """
    if particle.semantic_type != 0:
        raise AttributeError(
            "Particle {} has type {}, can only give"\
            " startpoints to shower fragments!".format(
                particle.id, particle.semantic_type))
    if verbose:
        print("Found {} PPN candidate points for particle {}".format(
            particle.ppn_candidates.shape[0], particle.id))
    if particle.ppn_candidates.shape[0] == 0:
        if verbose:
            print("Particle {} has no PPN candidates!".format(particle.id))
        startpoint = -np.ones(3)
    else:
        centroid = particle.points.mean(axis=0)
        ppn_coordinates = particle.ppn_candidates[:, :3]
        dist = np.linalg.norm((ppn_coordinates - centroid), axis=1)
        index = dist.argsort()[0]
        startpoint = ppn_coordinates[index]
    particle.startpoint = startpoint
    assert sum(startpoint.shape) == 3
    return startpoint
