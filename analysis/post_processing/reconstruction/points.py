import numpy as np
import numba as nb

from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

from mlreco.utils.globals import COORD_COLS, TRACK_SHP
from mlreco.utils.tracking import check_track_orientation
from mlreco.utils.ppn import get_ppn_predictions, check_track_orientation_ppn

from analysis.post_processing import post_processing


@post_processing(data_capture=[], 
                 result_capture=['particles'],
                 result_capture_optional=['ppn_candidates'])
def assign_particle_extrema(data_dict, result_dict,
                            method='local',
                            **kwargs):
    '''
    Assigns track start point and end point.
    
    Parameters
    ----------
    data_dict : dict
        Input data dictionary
    result_dict : dict
        Chain output dictionary
    method : algorithm to correct track startpoint/endpoint misplacement.
        The following modes are available:
        - local: computes local energy deposition density only at
        the extrema and chooses the higher one as the endpoint.
        - gradient: computes local energy deposition density throughout the
        track, computes the overall slope (linear fit) of the energy density
        variation to estimate the direction.
        - ppn: uses ppn candidate predictions (classify_endpoints) to assign
        start and endpoints.
    kwargs : dict
        Extra arguments to pass to the `check_track_orientation` or the
        `check_track_orientation_ppn' functions
    '''
    for p in result_dict['particles']:
        if p.semantic_type == TRACK_SHP:
            # Check if the end points need to be flipped
            if method in ['local', 'gradient']:
                flip = not check_track_orientation(p.points, p.depositions,
                        p.start_point, p.end_point, method, **kwargs)
            elif method == 'ppn':
                assert 'ppn_candidates' in result_dict, \
                        'Must run the get_ppn_predictions post-processor '\
                        'before using PPN predictions to assign track extrema'
                flip = not check_track_orientation_ppn(p.start_point,
                        p.end_point, result_dict['ppn_candidates'])
            else:
                raise ValueError(f'Point assignment method not recognized: {method}')

            # If needed, flip en end points
            if flip:
                start_point, end_point = p.end_point, p.start_point
                p.start_point = start_point
                p.end_point   = end_point
    
    return {}
