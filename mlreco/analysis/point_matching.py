from typing import List
import numpy as np
import pandas as pd

from scipy.spatial.distance import cdist
from .particle import Particle

def match_points_to_particles(ppn_points : pd.DataFrame, 
                              particles : List[Particle], 
                              semantic_type=None, threshold=2):
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
        ppn_points_type = ppn_points.query(
            "'PPN Point Type' == {}".format(semantic_type))
    else:
        ppn_points_type = ppn_points
    
    ppn_coords = ppn_points_type[['x', 'y', 'z']].to_numpy()
    for particle in particles:
        dist = cdist(ppn_coords, particle.points)
        matches = ppn_points_type[dist.min(axis=1) < threshold]
        particle.ppn_candidates = matches
        
        
def get_track_endpoints(particle : Particle, verbose=False):
    '''
    Using ppn_candiates attached to <Particle>, get two
    endpoints of tracks by farthest distance from the track's
    spatial centroid. 
    '''
    if particle.semantic_type != 1:
        raise AttributeError(
            "Particle {} has type {}, can only give"\
            " endpoints to tracks!".format(particle.id, 
                                           particle.semantic_type))
    if particle.ppn_candidates.shape[0] == 0:
        raise AttributeError(
            "Particle {} has no PPN candidates!".format(particle.id))
    if verbose:
        print("Found {} PPN candidate points for particle {}".format(
            particle.ppn_candidates.shape[0], particle.id))
    centroid = particle.points.mean(axis=0)
    ppn_coordinates = particle.ppn_candidates[['x', 'y', 'z']].to_numpy()
    dist = cdist(centroid.reshape(1, -1), 
                 particle.ppn_candidates[['x', 'y', 'z']].to_numpy()).squeeze()
    endpt_inds = dist.argsort()[-2:]
    endpoints = ppn_coordinates[endpt_inds]
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
    if particle.ppn_candidates.shape[0] == 0:
        raise AttributeError(
            "Particle {} has no PPN candidates!".format(particle.id))
    if verbose:
        print("Found {} PPN candidate points for particle {}".format(
            particle.ppn_candidates.shape[0], particle.id))
    centroid = particle.points.mean(axis=0)
    ppn_coordinates = particle.ppn_candidates[['x', 'y', 'z']].to_numpy()
    dist = cdist(centroid.reshape(1, -1), 
                 particle.ppn_candidates[['x', 'y', 'z']].to_numpy()).squeeze()
    index = dist.argsort()[0]
    startpoint = ppn_coordinates[index]
    particle.startpoint = startpoint
    return startpoint

