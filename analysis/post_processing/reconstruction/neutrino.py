from pprint import pprint

import numpy as np

from analysis.post_processing import post_processing
from mlreco.utils.globals import *

@post_processing(data_capture=[], 
                 result_capture=['interactions'],
                 result_capture_optional=['truth_interactions'])
def nu_calorimetric_energy(data_dict,
                           result_dict,
                           conversion_factor=1.):
    """
    
    """
    
    # for ia in result_dict['interactions']:
    #     if 
            
    return {}