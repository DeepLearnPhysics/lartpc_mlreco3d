import numpy as np
import networkx as nx

from collections import Counter

from mlreco.utils.globals import *
from mlreco.utils.gnn.cluster import get_cluster_directions, cluster_direction
from mlreco.utils.geometry import Geometry

from analysis.post_processing import post_processing



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
    if not len(particles):
        return data_dict

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

@post_processing(data_capture=['input_data'], result_capture=['input_rescaled',
                                                              'particle_clusts',
                                                              'particle_start_points',
                                                              'particle_end_points',
                                                              'particles'])
def reconstruct_directions(data_dict,
                           result_dict,
                           neighborhood_radius=5,
                           optimize=True):
    if 'input_rescaled' not in result_dict:
        input_data = data_dict['input_data']
    else:
        input_data = result_dict['input_rescaled']

    if not len(result_dict['particles']):
        return {}

    particles = np.array([p.index for p in result_dict['particles']])
    start_points = np.vstack([p.start_point for p in result_dict['particles']])
    end_points = np.vstack([p.end_point for p in result_dict['particles']])

    update_dict = {
        'particle_start_directions': get_cluster_directions(input_data[:,COORD_COLS],
                                                            start_points, 
                                                            particles,
                                                            neighborhood_radius, 
                                                            optimize),
        'particle_end_directions':   get_cluster_directions(input_data[:,COORD_COLS],
                                                            end_points, 
                                                            particles,
                                                            neighborhood_radius, 
                                                            optimize)
    }
    
    for i, p in enumerate(result_dict['particles']):
        p.start_dir = update_dict['particle_start_directions'][i]
        p.end_dir   = update_dict['particle_end_directions'][i]
            
    return {}


@post_processing(data_capture=['graph'],
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
    graph = data_dict['graph']
    particles = result_dict['truth_particles']
    
    for p in particles:
        G.add_node(p.id, attr=getattr(p, mode))
    for p in result_dict['truth_particles']:
        parent = p.parent_id
        if parent in G and int(parent) != int(p.id):
            edges.append((parent, p.id))
    G.add_edges_from(edges)
    
    for p in particles:
        successors = list(G.successors(p.id))
        counter = Counter()
        counter.update([G.nodes[succ]['attr'] for succ in successors])
        for key, val in counter.items():
            p.children_counts[key] = val
    return {}


@post_processing(data_capture=['meta'], 
                 result_capture=['particles', 'interactions'],
                 result_capture_optional=['truth_particles', 'truth_interactions'])
def check_containement(data_dict, result_dict,
                       margin=5,
                       detector='icarus',
                       boundary_file=None,
                       mode='module'):
    """
    Check whether a particle comes within some distance of the boundaries
    of the detector and assign the `is_contained` attribute accordingly.

    Parameters
    ----------
    data_dict : dict
        Input data dictionary
    result_dict : dict
        Chain output dictionary
    margin : float, default 5 cm
        Minimum distance from a detector wall to be considered contained
    detector : str, default 'icarus'
        Detector to get the geometry from
    boundary_file : str, optional
        Path to a detector boundary file. Supersedes `detector` if set
    mode : str, default 'module'
        Containement criterion (one of 'global', 'module', 'tpc'):
        - If 'global', makes sure is is contained within the outermost walls
        - If 'module', makes sure it is contained within a single module
        - If 'tpc', makes sure it is contained within a single tpc
    """
    # Initialize the geometry
    geo = Geometry(detector, boundary_file)

    # Check containment
    for k in ['particles', 'interactions', 'truth_particles', 'truth_interactions']:
        if k in result_dict:
            for p in result_dict[k]:
                # Make sure the particle coordinates are expressed in centimeters
                if p.units != 'cm':
                    raise ValueError('Particle coordinates must be expressed in cm '
                            'to use the range-based kinetice energy reconstruction')

                p.is_contained = geo.check_containment(p.points, margin, mode)

    return {}
