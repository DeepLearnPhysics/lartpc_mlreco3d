import numpy as np

from mlreco.utils.globals import *
from mlreco.utils.gnn.cluster import cluster_direction
from mlreco.utils.geometry import Geometry

from analysis.classes import TruthParticle
from analysis.post_processing import post_processing


@post_processing(data_capture=[],
                 result_capture=['particles'],
                 result_capture_optional=['truth_particles'])
def reconstruct_directions(data_dict,
                           result_dict,
                           neighborhood_radius=5,
                           optimize=True,
                           truth_point_mode='points',
                           run_mode='both'):
    '''
    Reconstruct the direction of particles w.r.t. to their end points.

    Parameters
    ----------
    data_dict : dict
        Input data dictionary
    result_dict : dict
        Chain output dictionary
    neighborhood_radius : float, default 5
        Max distance between start voxel and other voxels
    optimize : bool, default True
        Optimizes the number of points involved in the estimate
    truth_point_mode : str, default 'points'
        Point attribute to use to compute the direction of true particles
    run_mode : str, default 'both'
        Which output to run on (one of 'both', 'reco' or 'truth')
    '''
    # List objects for which to reconstruct direcions
    key_list = []
    if run_mode in ['reco', 'both']:
        key_list += ['particles']
    if run_mode in ['truth', 'both']:
        key_list += ['truth_particles']

    # Loop over particle objects
    for k in key_list:
        for p in result_dict[k]:
            # Make sure the particle coordinates are expressed in cm
            if p.units != 'cm':
                raise ValueError('Particle coordinates must be expressed in cm '
                        'to use the range-based kinetic energy reconstruction, currently in {}'.format(p.units))

            # Get point coordinates
            if not isinstance(p, TruthParticle):
                coordinates = p.points
            else:
                coordinates = getattr(p, truth_point_mode)
            if not len(coordinates):
                continue

            # Reconstruct directions from either end of the particle
            p.start_dir = cluster_direction(coordinates, p.start_point,
                    neighborhood_radius, optimize)
            p.end_dir   = cluster_direction(coordinates, p.end_point,
                    neighborhood_radius, optimize)

    return {}


@post_processing(data_capture=[],
                 result_capture=['particles', 'interactions'],
                 result_capture_optional=['truth_particles', 'truth_interactions'])
def check_containement(data_dict, result_dict,
                       margin=5,
                       detector='icarus',
                       boundary_file=None,
                       mode='module',
                       truth_point_mode='points',
                       run_mode='both'):
    '''
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
        - If 'detector', makes sure is is contained within the outermost walls
        - If 'module', makes sure it is contained within a single module
        - If 'tpc', makes sure it is contained within a single tpc
    truth_point_mode : str, default 'points'
        Point attribute to use to check containment of true particles
    run_mode : str, default 'both'
        Which output to run on (one of 'both', 'reco' or 'truth')
    '''
    # Initialize the geometry
    if boundary_file is not None:
        geo = Geometry(boundary_file)
    else:
        geo = Geometry(detector)

    # List objects for which to check containement
    key_list = []
    if run_mode in ['reco', 'both']:
        key_list += ['particles', 'interactions']
    if run_mode in ['truth', 'both']:
        key_list += ['truth_particles', 'truth_interactions']

    # Loop over particle objects
    for k in key_list:
        for p in result_dict[k]:
            # Make sure the particle coordinates are expressed in cm
            if p.units != 'cm':
                raise ValueError('Particle coordinates must be expressed in cm '
                        'to use the range-based kinetice energy reconstruction')

            # Get point coordinates
            if not isinstance(p, TruthParticle):
                coordinates = p.points
            else:
                coordinates = getattr(p, truth_point_mode)
            if not len(coordinates):
                continue

            # Check containment
            p.is_contained = geo.check_containment(p.points, margin, mode)

    return {}
