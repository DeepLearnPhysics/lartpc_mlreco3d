import numpy as np
from typing import List
from scipy.spatial.distance import cdist

from analysis.post_processing import post_processing
from mlreco.utils.globals import *
from mlreco.utils.ppn import uresnet_ppn_type_point_selector
from analysis.classes import Particle

PPN_COORD_COLS  = (0,1,2)
PPN_LOGITS_COLS = (3,4,5,6,7)
PPN_SCORE_COL   = (8,9)

@post_processing(data_capture=[], result_capture=['input_rescaled',
                                                  'particles',
                                                  'ppn_classify_endpoints',
                                                  'ppn_output_coords',
                                                  'ppn_points',
                                                  'ppn_coords',
                                                  'ppn_masks',
                                                  'segmentation'])
def assign_ppn_candidates(data_dict, result_dict):
    """Select ppn candidates and assign them to each particle instance.

    Parameters
    ----------
    data_dict : dict
        Data dictionary (contains one image-worth of data)
    result_dict : dict
        Result dictionary (contains one image-worth of full chain outputs)

    Returns
    -------
    None
        Operation is in-place on Particles. 
    """
    
    result = {}
    for key, val in result_dict.items():
        result[key] = [val]
    
    ppn = uresnet_ppn_type_point_selector(result['input_rescaled'][0],
                                          result, entry=0, 
                                          apply_deghosting=False)
    
    ppn_voxels = ppn[:, 1:4]
    ppn_score = ppn[:, 5]
    ppn_type = ppn[:, 12]
    if 'ppn_classify_endpoints' in result:
        ppn_endpoint = ppn[:, 13:]
        assert ppn_endpoint.shape[1] == 2

    ppn_candidates = []
    for i, pred_point in enumerate(ppn_voxels):
        pred_point_type, pred_point_score = ppn_type[i], ppn_score[i]
        x, y, z = ppn_voxels[i][0], ppn_voxels[i][1], ppn_voxels[i][2]
        if 'ppn_classify_endpoints' in result:
            ppn_candidates.append(np.array([x, y, z, 
                                            pred_point_score, 
                                            pred_point_type, 
                                            ppn_endpoint[i][0],
                                            ppn_endpoint[i][1]]))
        else:
            ppn_candidates.append(np.array([x, y, z, 
                                            pred_point_score, 
                                            pred_point_type]))

    if len(ppn_candidates):
        ppn_candidates = np.vstack(ppn_candidates)
    else:
        enable_classify_endpoints = 'ppn_classify_endpoints' in result
        ppn_candidates = np.empty((0, 5 if not enable_classify_endpoints else 7), 
                                  dtype=np.float32)
        
    match_points_to_particles(ppn_candidates, result_dict['particles'])
    
    return {}
    
    
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
        particle.ppn_candidates = matches.reshape(-1, 7)