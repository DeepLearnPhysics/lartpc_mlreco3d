import numpy as np

from mlreco.utils.gnn.cluster import get_cluster_directions, cluster_direction
from analysis.post_processing import post_processing
from mlreco.utils.globals import *


@post_processing(data_capture=['input_data'], result_capture=['input_rescaled',
                                                              'particle_clusts',
                                                              'particle_start_points',
                                                              'particle_end_points',
                                                              'particles'])
def particle_direction(data_dict,
                       result_dict,
                       neighborhood_radius=5,
                       optimize=True):

    if 'input_rescaled' not in result_dict:
        input_data = data_dict['input_data']
    else:
        input_data = result_dict['input_rescaled']
    particles      = result_dict['particle_clusts']
    start_points   = result_dict['particle_start_points']
    end_points     = result_dict['particle_end_points']

    update_dict = {
        'particle_start_directions': get_cluster_directions(input_data[:,COORD_COLS],
                                                            start_points[:,COORD_COLS], 
                                                            particles,
                                                            neighborhood_radius, 
                                                            optimize),
        'particle_end_directions':   get_cluster_directions(input_data[:,COORD_COLS],
                                                            end_points[:,COORD_COLS], 
                                                            particles,
                                                            neighborhood_radius, 
                                                            optimize)
    }
    
    for i, p in enumerate(result_dict['particles']):
        p.start_dir = update_dict['particle_start_directions'][i]
        p.end_dir   = update_dict['particle_end_directions'][i]
            
    return update_dict

@post_processing(data_capture=['input_data'], result_capture=['truth_particles'])
def particle_direction_truth(data_dict,
                             result_dict,
                             neighborhood_radius=5,
                             optimize=True):
    for p in result_dict['truth_particles']:
        p.start_dir = cluster_direction(p.points, p.start_point, 
                                        neighborhood_radius, 
                                        optimize=optimize)
    return {}

@post_processing(data_capture=['meta'], 
                 result_capture=['particles', 'interactions'],
                 result_capture_optional=['truth_particles', 'truth_interactions'])
def fiducial_cut(data_dict, result_dict, margin=0, spatial_units='cm'):
    """_summary_

    Parameters
    ----------
    data_dict : _type_
        _description_
    result_dict : _type_
        _description_
    margin : int, optional
        _description_, by default 5
    spatial_units : str, optional
        _description_, by default 'cm'
    """
    particles = result_dict['particles']
    interactions = result_dict['interactions']
    
    for p in particles:
        p.is_contained = check_containment_cm(p, margin=margin)
        
    for ia in interactions:
        ia.is_contained = check_containment_cm(ia, margin=margin)
        
    if 'truth_particles' in result_dict:
        for p in result_dict['truth_particles']:
            p.is_contained = check_containment_cm(p, margin=margin)
            
    if 'truth_interactions' in result_dict:
        for ia in result_dict['truth_interactions']:
            ia.is_contained = check_containment_cm(ia, margin=margin)
            
    return {}
            
            
# ------------------------Helper Functions----------------------------

FIDUCIAL_VOLUME = {
    'x1_min': -358.49,
    'x2_min': 61.94,
    'x1_max': -61.94,
    'x2_max': 358.49,
    'y_min': -181.86,
    'y_max': 134.96,
    'z_min': -894.95,
    'z_max': 894.95
}

def check_containment_cm(obj, margin=0):
    x1 = (obj.points[:, 0] > FIDUCIAL_VOLUME['x1_min'] + margin) \
       & (obj.points[:, 0] < FIDUCIAL_VOLUME['x1_max'] - margin)
    x2 = (obj.points[:, 0] > FIDUCIAL_VOLUME['x2_min'] + margin) \
       & (obj.points[:, 0] < FIDUCIAL_VOLUME['x2_max'] - margin)
    y  = (obj.points[:, 1] > FIDUCIAL_VOLUME['y_min'] + margin) \
       & (obj.points[:, 1] < FIDUCIAL_VOLUME['y_max'] - margin)
    z  = (obj.points[:, 2] > FIDUCIAL_VOLUME['z_min'] + margin) \
       & (obj.points[:, 2] < FIDUCIAL_VOLUME['z_max'] - margin)
    x  = x1 | x2
    x = x.all()
    y = y.all()
    z = z.all()
    return (x and y and z)