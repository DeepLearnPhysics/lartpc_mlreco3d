import numpy as np

from mlreco.utils.globals import *
from mlreco.utils.gnn.cluster import cluster_direction
from mlreco.utils.geometry import Geometry

from analysis.classes import TruthParticle, TruthInteraction
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
                        'to reconstruct directions, currently in {}'.format(p.units))

            # Get point coordinates
            if not isinstance(p, TruthParticle):
                coords = p.points
            else:
                coords = getattr(p, truth_point_mode)
            if not len(coords):
                continue

            # Reconstruct directions from either end of the particle
            p.start_dir = cluster_direction(coords, p.start_point,
                    neighborhood_radius, optimize)
            p.end_dir   = cluster_direction(coords, p.end_point,
                    neighborhood_radius, optimize)

    return {}


@post_processing(data_capture=[],
                 result_capture=['particles', 'interactions'],
                 result_capture_optional=['truth_particles', 'truth_interactions'])
def check_containement(data_dict, result_dict,
                       margin,
                       use_source=False,
                       detector=None,
                       boundary_file=None,
                       source_file=None,
                       mode='module',
                       truth_point_mode='points',
                       run_mode='both'):
    '''
    Check whether a particle or interaction comes within some distance
    of the boundaries of the detector and assign the `is_contained`
    attribute accordingly.

    If `use_source` is True, the cut will be based on the source of the point
    cloud, i.e. if a point cloud was produced by TPCs i and j, it must be
    contained within the volume bound by the set of TPCs i and j.

    Parameters
    ----------
    data_dict : dict
        Input data dictionary
    result_dict : dict
        Chain output dictionary
    margin : Union[float, List[float], np.array]
        Minimum distance from a detector wall to be considered contained:
        - If float: distance buffer is shared between all 6 walls
        - If [x,y,z]: distance is shared between pairs of falls facing
          each other and perpendicular to a shared axis
        - If [[x_low,x_up], [y_low,y_up], [z_low,z_up]]: distance is specified
          individually of each wall.
    use_source : bool, default False
        If True, use the point sources to define a containment volume
    detector : str, optional
        Detector to get the geometry from
    boundary_file : str, optional
        Path to a detector boundary file. Supersedes `detector` if set
    source_file : str, optional
        Path to a detector source file. Supersedes `detector` if set
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
    # Define boundary and source files
    boundaries = boundary_file if boundary_file is not None else detector
    assert boundaries is not None, \
            'Must provide detector name or boundary file to check containment'

    if not use_source:
        sources = None
    else:
        sources = source_file if source_file is not None else detector

    # Initialize the geometry
    geo = Geometry(boundaries, sources)

    # List objects for which to check containement
    key_list = []
    if run_mode in ['reco', 'both']:
        key_list += ['particles', 'interactions']
    if run_mode in ['truth', 'both']:
        key_list += ['truth_particles', 'truth_interactions']

    # Loop over partcile/interaction objects
    for k in key_list:
        for p in result_dict[k]:
            # Make sure the particle/interaction coordinates are expressed in cm
            if p.units != 'cm':
                raise ValueError('Particle coordinates must be expressed in cm '
                        'to check containement, currently in {}'.format(p.units))

            # Get point coordinates
            if not isinstance(p, TruthParticle) \
                    and not isinstance(p, TruthInteraction):
                coords = p.points
            else:
                coords = getattr(p, truth_point_mode)
            if not len(coords):
                continue

            # Check containment
            sources = p.sources if use_source and len(p.sources) else None
            p.is_contained = geo.check_containment(coords, margin, sources, mode)

    return {}


@post_processing(data_capture=[],
                 result_capture=['interactions'],
                 result_capture_optional=['truth_interactions'])
def check_fiducial(data_dict, result_dict,
                   margin,
                   detector=None,
                   boundary_file=None,
                   mode='module',
                   truth_vertex_mode='truth_vertex',
                   run_mode='both'):
    '''
    Check whether an interaction vertex is within some fiducial volume defined
    as margin distances from each of the detector walls.

    Parameters
    ----------
    data_dict : dict
        Input data dictionary
    result_dict : dict
        Chain output dictionary
    margin : Union[float, List[float], np.array]
        Minimum distance from a detector wall to be considered contained:
        - If float: distance buffer is shared between all 6 walls
        - If [x,y,z]: distance is shared between pairs of falls facing
          each other and perpendicular to a shared axis
        - If [[x_low,x_up], [y_low,y_up], [z_low,z_up]]: distance is specified
          individually of each wall.
    detector : str, default 'icarus'
        Detector to get the geometry from
    boundary_file : str, optional
        Path to a detector boundary file. Supersedes `detector` if set
    mode : str, default 'module'
        Containement criterion (one of 'global', 'module', 'tpc'):
        - If 'detector', makes sure is is contained within the outermost walls
        - If 'module', makes sure it is contained within a single module
        - If 'tpc', makes sure it is contained within a single tpc
    truth_vertex_mode : str, default 'truth_vertex'
        Vertex attribute to use to check containment of true interactions
    run_mode : str, default 'both'
        Which output to run on (one of 'both', 'reco' or 'truth')
    '''
    # Define boundary and source files
    boundaries = boundary_file if boundary_file is not None else detector
    assert boundaries is not None, \
            'Must provide detector name or boundary file to check containment'

    # Initialize the geometry
    geo = Geometry(boundaries)

    # List objects for which to check containement
    key_list = []
    if run_mode in ['reco', 'both']:
        key_list += ['interactions']
    if run_mode in ['truth', 'both']:
        key_list += ['truth_interactions']

    # Loop over interaction objects
    for k in key_list:
        for p in result_dict[k]:
            # Make sure the interaction coordinates are expressed in cm
            if p.units != 'cm':
                raise ValueError('Particle coordinates must be expressed in cm '
                        'to check fiducial, currently in {}'.format(p.units))

            # Get point coordinates
            if not isinstance(p, TruthInteraction):
                vertex = p.vertex
            else:
                vertex = getattr(p, truth_vertex_mode)
            vertex = vertex.reshape(-1,3)

            # Check containment
            p.is_fiducial = geo.check_containment(vertex, margin, mode=mode)

    return {}
