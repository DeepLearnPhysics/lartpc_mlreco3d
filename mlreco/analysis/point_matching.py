from typing import List
import numpy as np
import pandas as pd

from scipy.spatial.distance import cdist
from .particle import Particle

def match_points_to_particles(ppn_points : np.ndarray, 
                              particles : List[Particle], 
                              semantic_type=None, ppn_distance_threshold=2):
    '''
    For each particle, match ppn_points that have hausdorff distance
    less than <threshold> and inplace update particle.ppn_candidates

    If semantic_type is set to a class integer value, 
    points will be matched to particles with the same 
    predicted semantic type. 
    
    Inputs:
        - ppn_points (N x 4 np.array): (coords, point_type) array 
        - particles: list of Particles

    '''
    if semantic_type is not None:
        ppn_points_type = ppn_points[ppn_points[:, -1] == semantic_type]
    else:
        ppn_points_type = ppn_points
        # TODO: Fix semantic type ppn selection
    
    ppn_coords = ppn_points_type[:, :3]
    for particle in particles:
        dist = cdist(ppn_coords, particle.points)
        matches = ppn_points_type[dist.min(axis=1) < ppn_distance_threshold]
        particle.ppn_candidates = matches
        
        
def get_track_endpoints(particle : Particle, verbose=False):
    '''
    Using ppn_candiates attached to <Particle>, get two
    endpoints of tracks by farthest distance from the track's
    spatial centroid. 
    '''
    if verbose:
        print("Found {} PPN candidate points for particle {}".format(
            particle.ppn_candidates.shape[0], particle.id))
    if particle.semantic_type != 1:
        raise AttributeError(
            "Particle {} has type {}, can only give"\
            " endpoints to tracks!".format(particle.id, 
                                           particle.semantic_type))
    if particle.ppn_candidates.shape[0] == 0:
        print("Particle {} has no PPN candidates!"\
            " Running brute-force endpoint finder...".format(particle.id))
        endpoints = get_track_endpoints_centroid(particle)
    elif particle.ppn_candidates.shape[0] == 1:
        print("Particle {} has only one PPN candidate!"\
            " Running brute-force endpoint finder...".format(particle.id))
        endpoints = get_track_endpoints_centroid(particle)
    else:
        centroid = particle.points.mean(axis=0)
        ppn_coordinates = particle.ppn_candidates[:, :3]
        dist = cdist(centroid.reshape(1, -1), ppn_coordinates).squeeze()
        endpt_inds = dist.argsort()[-2:]
        endpoints = particle.ppn_candidates[endpt_inds]
        particle.endpoints = endpoints
    assert endpoints.shape[0] == 2
    return endpoints



def get_track_endpoints_centroid(particle):
    '''
    Computes track endpoints without ppn predictions by
    selecting the farthest two points from the coordinate centroid. 
    '''
    coords = particle.points
    centroid = coords.mean(axis=0)
    dist = cdist(coords, centroid.reshape(1, -1))
    inds = dist.squeeze().argsort()[-2:]
    endpoints = coords[inds]
    particle.endpoints = endpoints
    return endpoints


def get_shower_startpoint(particle : Particle, verbose=False):
    '''
    Using ppn_candiates attached to <Particle>, get one
    startpoint of shower by nearest hausdorff distance. 
    '''
    if particle.semantic_type != 0:
        raise AttributeError(
            "Particle {} has type {}, can only give"\
            " startpoints to shower fragments!".format(
                particle.id, particle.semantic_type))
    if verbose:
        print("Found {} PPN candidate points for particle {}".format(
            particle.ppn_candidates.shape[0], particle.id))
    if particle.ppn_candidates.shape[0] == 0:
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

