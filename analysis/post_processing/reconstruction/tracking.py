import numpy as np

from mlreco.utils.energy_loss import csda_table_spline
from mlreco.utils.tracking import get_track_length
from mlreco.utils.globals import *

from analysis.classes import TruthParticle
from analysis.post_processing import post_processing


@post_processing(data_capture=['meta'],
                 result_capture=['particles'],
                 result_capture_optional=['truth_particles'])
def reconstruct_csda_energy(data_dict, result_dict,
                            tracking_mode='bin_pca',
                            include_pids=[2,3,4,5],
                            truth_point_mode='points',
                            run_mode = 'both',
                            **kwargs):
    '''
    Reconstruct the kinetic energy of tracks based on their range in liquid
    argon using the continuous slowing down approximation (CSDA).

    Parameters
    ----------
    data_dict : dict
        Input data dictionary
    result_dict : dict
        Chain output dictionary
    tracking_mode : str, default 'step'
        Method used to compute the track length (one of 'displacement', 'step',
        'step_next', 'bin_pca' or 'spline')
    include_pids : list, default [2, 3, 4, 5]
        Particle species to compute the kinetic energy from
    truth_point_mode : str, default 'points'
        Point attribute to use to compute the track energy for true particles
    run_mode : str, default 'both'
        Which output to run on (one of 'both', 'reco' or 'truth')
    '''
    # Fetch the functions that map the range to a KE
    splines = {ptype: csda_table_spline(ptype) for ptype in include_pids}

    # List objects for which to reconstruct track KE
    if run_mode not in ['reco', 'truth', 'both']:
        raise ValueError('`run_mode` must be either `reco`, `truth` or `both`')

    key_list = []
    if run_mode in ['reco', 'both']:
        key_list += ['particles']
    if run_mode in ['truth', 'both']:
        key_list += ['truth_particles']

    # Loop over particle objects
    for k in key_list:
        for p in result_dict[k]:
            # Only run this algorithm on tracks that have a CSDA table
            if not ((p.semantic_type == 1) and (p.pid in include_pids)):
                continue
            
            # Make sure the particle coordinates are expressed in cm
            if p.units != 'cm':
                raise ValueError('Particle coordinates must be expressed in cm '
                        'to use the range-based kinetic energy reconstruction')

            # Get point coordinates
            if not isinstance(p, TruthParticle):
                coordinates = p.points
            else:
                # assert p.units == 'cm'
                coordinates = getattr(p, truth_point_mode)
            if not len(coordinates):
                continue

            # Compute the length of the track
            length = get_track_length(coordinates, point=p.start_point,
                    method=tracking_mode, **kwargs)

            # Store the length and the CSDA kinetic energy
            p.length  = length
            p.csda_ke = splines[p.pid](length).item() if length > 0. else 0.

    return {}
