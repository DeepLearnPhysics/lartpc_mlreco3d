import numpy as np

from analysis.post_processing import post_processing

from mlreco.utils.energy_loss import csda_table_spline
from mlreco.utils.tracking import get_track_length
from mlreco.utils.globals import *


@post_processing(data_capture=['meta'],
                 result_capture=['particles'],
                 result_capture_optional=['truth_particles'])
def reconstruct_track_energy(data_dict, result_dict,
                             tracking_mode='bin_pca',
                             segment_length=5.,
                             include_pids=[2,3,4],#,5],
                             truth_point_mode='points',
                             run_mode = 'both',
                             **kwargs):

    # Fetch the functions that map the range to a KE
    splines = {ptype: csda_table_spline(ptype) for ptype in include_pids}

    # Compute CSDA kinetic energy for reconstructed particles, if requested
    if run_mode in ['reco', 'both']:
        for p in result_dict['particles']:
            # Only run this algorithm on tracks that have a CSDA table
            if not ((p.semantic_type == 1) and (p.pid in include_pids)):
                continue
            
            # Make sure the particle coordinates are expressed in centimeters
            if p.units != 'cm':
                raise ValueError('Particle coordinates must be expressed in cm '
                        'to use the range-based kinetice energy reconstruction')

            # Compute the length of the track
            length = get_track_length(p.points, segment_length,
                    p.start_point, method=tracking_mode, **kwargs)

            # Store the length and the CSDA kinetic energy
            p.length = length
            p.csda_kinetic_energy = splines[p.pid](length)

    # Compute CSDA kinetic energy for true particles, if requested
    if run_mode in ['truth', 'both']:
        for p in result_dict['truth_particles']:
            # Only run this algorithm on tracks that have a CSDA table
            if not ((p.semantic_type == 1) and (p.pid in include_pids)):
                continue
            
            # Make sure the particle coordinates are expressed in centimeters
            if p.units != 'cm':
                raise ValueError('Particle coordinates must be expressed in cm '
                        'to use the range-based kinetice energy reconstruction')

            # Compute the length of the track
            coordinates = getattr(p, truth_point_mode)
            if not len(coordinates):
                continue
            length = get_track_length(coordinates, segment_length,
                    p.start_point, method=tracking_mode, **kwargs)

            # Store the length and the CSDA kinetic energy
            p.length = length
            p.csda_kinetic_energy = splines[p.pid](length)

    return {}
