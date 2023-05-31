import numpy as np

from analysis.post_processing import post_processing
from mlreco.utils.globals import *


@post_processing(data_capture=[], result_capture=['particles', 'interactions'])
def adjust_pid_and_primary_labels(data_dict,
                                  result_dict,
                                  pid_thresholds=None,
                                  primary_threshold=0.5):

    particles = result_dict['particles']
    interactions = result_dict['interactions']
    for p in particles:
        # Process PIDs
        if (p.pid_scores[0] == 0) and (p.pid_scores[4] >= 0.85) and p.pid != 4:
            p.pid = 4
        elif (p.pid_scores[0] == 0) and (p.pid_scores[4] < 0.85) and (p.pid_scores[2] >= 0.1) and p.pid != 2:
            p.pid = 2
        elif (p.pid_scores[0] == 0) and (p.pid_scores[4] < 0.85) and (p.pid_scores[2] < 0.1) and p.pid != 3:
            p.pid = 3
        # Process Primaries
        if p.primary_scores[1] > primary_threshold and not p.is_primary:
            p.is_primary = True
            
    for ia in interactions:
        ia._update_particle_info()
        
    return {}
