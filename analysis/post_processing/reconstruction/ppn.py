import numpy as np

from scipy.spatial.distance import cdist

from mlreco.utils.globals import COORD_COLS, PPN_SHAPE_COL
from mlreco.utils.ppn import get_ppn_predictions

from analysis.post_processing import post_processing

@post_processing(data_capture=['input_data'],
                 result_capture=['input_rescaled',
                                 'segmentation',
                                 'ppn_points',
                                 'ppn_coords',
                                 'ppn_masks',
                                 'ppn_classify_endpoints'],
                 result_capture_optional=['input_rescaled'])
def get_ppn_candidates(data_dict, result_dict, **kwargs):
    '''
    Run the PPN post-processing function to produce PPN candidate
    points from the raw PPN output.

    Parameters
    ----------
    data_dict : dict
        Data dictionary
    result_dict : dict
        Result dictionary
    **kwargs : dict, optional
        Keyword arguments to pass to the `get_ppn_predictions` function

    Returns
    -------
    dict
        Update result dictionary containing 'ppn_candidates' key
    '''
    # Pick the input data to be used
    if 'input_rescaled' not in result_dict.keys():
        input_data = data_dict['input_data']
    else:
        input_data = result_dict['input_rescaled']
    
    result_nest = {}
    for key, val in result_dict.items():
        result_nest[key] = [val]

    # Get the PPN candidates
    ppn_candidates = get_ppn_predictions(input_data,
            result_nest, apply_deghosting=False)
    result_dict['ppn_candidates'] = ppn_candidates
        
    return result_dict
    
    
@post_processing(data_capture=['input_data'],
                 result_capture=['input_rescaled',
                                 'segmentation',
                                 'ppn_points',
                                 'ppn_coords',
                                 'ppn_masks',
                                 'ppn_classify_endpoints',
                                 'particles'],
                 result_capture_optional=['input_rescaled',
                                          'ppn_candidates'])
def assign_ppn_candidates(data_dict, result_dict,
                          restrict_semantic_type = False,
                          ppn_distance_threshold = 2,
                          **kwargs):
    '''
    Function for matching ppn points to particles.

    For each particle, match ppn_points that have hausdorff distance
    less than <threshold> and inplace update particle.ppn_candidates

    If semantic_type is set to a class integer value,
    points will be matched to particles with the same
    predicted semantic type.

    Parameters
    ----------
    data_dict : dict
        Data dictionary
    result_dict : dict
        Result dictionary
    restrict_semantic_type : bool, default False
        If `True`, only associate PPN candidates with compatible shape
    ppn_distance_threshold : float, default 2
        Maximum distance required to assign ppn point to particle
    **kwargs : dict, optional
        Keyword arguments to pass to the `get_ppn_predictions` function
    '''
    # If not yet done, produce the PPN candidates from the raw PPN output
    if 'ppn_candidates' not in result_dict:
        get_ppn_candidates(data_dict, result_dict, **kwargs)
    ppn_candidates = result_dict['ppn_candidates']

    # Now fill the PPN candidates attribute of each reconstucted particle
    valid_mask = np.arange(len(ppn_candidates))
    for p in result_dict['particles']:
        if restrict_semantic_type:
            valid_mask = np.where(ppn_candidates[:, PPN_SHAPE_COL] == p.shape)

        ppn_points = ppn_candidates[valid_mask][:, COORD_COLS]
        dists = np.min(cdist(ppn_points, p.points), axis=1)
        matches = ppn_candidates[valid_mask][dists < ppn_distance_threshold]
        p.ppn_candidates = matches

    return result_dict
