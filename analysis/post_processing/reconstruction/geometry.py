import numpy as np

from mlreco.utils.gnn.cluster import get_cluster_directions, cluster_direction
from analysis.post_processing import post_processing
from mlreco.utils.globals import *

import networkx as nx
from collections import Counter


@post_processing(data_capture=['input_data'], result_capture=['input_rescaled',
                                                              'particle_clusts',
                                                              'particle_start_points',
                                                              'particle_end_points',
                                                              'particles'])
def particle_direction(data_dict,
                       result_dict,
                       neighborhood_radius=5,
                       optimize=True):
    """Estimate the direction of a particle using the startpoint.
    This modifies the <start_dir> attribute of each Particle object in-place.

    Parameters
    ----------
    data_dict : dict
        Input data dictionary
    result_dict : dict
        Chain output dictionary
    neighborhood_radius : int, optional
        The radius of the neighborhood around the startpoint, 
        used to compute the direction vector , by default 5
    optimize : bool, optional
        Option to use the optimizing algorithm, by default True

    Returns
    -------
    update_dict: dict
        Dictionary containing start and end directions for all particles in the image. 
    """

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


@post_processing(data_capture=[],
                 result_capture=['truth_particles'])
def count_children(data_dict, result_dict, mode='semantic_type'):
    """Post-processor for counting the number of children of a given particle,
    using the particle hierarchy information from parse_particle_graph.

    Parameters
    ----------
    data_dict : dict
        Input data dictionary
    result_dict : dict
        Chain output dictionary
    mode : str, optional
        Attribute name to categorize children, by default 'semantic_type'.
        This will count each child particle for different semantic types
        separately. 

    Returns
    -------
    None
        (Operation is in-place)
    """
    
    G = nx.DiGraph()
    edges = []
    
    for p in result_dict['truth_particles']:
        G.add_node(p.id, attr=getattr(p, mode))
    for p in result_dict['truth_particles']:
        parent = p.asis.parent_id()
        if parent in G:
            edges.append((parent, p.id))
    G.add_edges_from(edges)
    
    for p in result_dict['truth_particles']:
        successors = list(G.successors(p.id))
        counter = Counter([G.nodes[succ]['attr'] for succ in successors])
        for key, val in counter.items():
            p._children_counts[key] = val
            
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