import numpy as np
from copy import deepcopy
from scipy.spatial.distance import cdist

from analysis.post_processing import post_processing
from mlreco.utils.globals import *


@post_processing(data_capture=['input_data'], result_capture=['input_rescaled',
                                                              'particle_clusts',
                                                              'particle_start_points',
                                                              'particle_end_points'])
def order_end_points(data_dict,
                     result_dict,
                     method='local_dedx',
                     neighborhood_radius=5):

    assert method == 'local_dedx', 'Only method currently supported'

    input_data   = data_dict['input_data'] if 'input_rescaled' not in result_dict else result_dict['input_rescaled']
    particles    = result_dict['particle_clusts']
    start_points = result_dict['particle_start_points']
    end_points   = result_dict['particle_end_points']

    start_dedxs, end_dedxs = np.empty(len(particles)), np.empty(len(particles))
    for i, p in enumerate(particles):
        dist_mat = cdist(start_points[i, COORD_COLS][None,:], input_data[p][:, COORD_COLS]).flatten()
        de = np.sum(input_data[p][dist_mat < neighborhood_radius, VALUE_COL])
        start_dedxs[i] = de/neighborhood_radius

        dist_mat = cdist(end_points[i, COORD_COLS][None,:], input_data[p][:, COORD_COLS]).flatten()
        de = np.sum(input_data[p][dist_mat < neighborhood_radius, VALUE_COL])
        end_dedxs[i] = de/neighborhood_radius

    switch_mask = start_dedxs > end_dedxs
    temp_start_points = deepcopy(start_points)
    start_points[switch_mask] = end_points[switch_mask]
    end_points[switch_mask] = temp_start_points[switch_mask]

    update_dict = {
        'particle_start_points': start_points,
        'particle_end_points': end_points
    }
            
    return update_dict
