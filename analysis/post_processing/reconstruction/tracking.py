import numpy as np

from mlreco.utils.energy_loss import csda_table_spline
from mlreco.utils.tracking import get_track_length
from mlreco.utils.globals import *

from analysis.classes import TruthParticle
from analysis.post_processing import PostProcessor


class CSDAEnergyProcessor(PostProcessor):
    '''
    Reconstruct the kinetic energy of tracks based on their range in liquid
    argon using the continuous slowing down approximation (CSDA).
    '''
    name = 'reconstruct_csda_energy'
    result_cap = ['particles']
    result_cap_optional = ['truth_particles']

    def __init__(self,
                 tracking_mode='bin_pca',
                 include_pids=[2,3,4,5],
                 truth_point_mode='points',
                 run_mode = 'both',
                 **kwargs):
        '''
        Store the necessary attributes to do CSDA range-based estimations

        Parameters
        ----------
        tracking_mode : str, default 'step'
            Method used to compute the track length (one of 'displacement',
            'step', 'step_next', 'bin_pca' or 'spline')
        include_pids : list, default [2, 3, 4, 5]
            Particle species to compute the kinetic energy from
        truth_point_mode : str, default 'points'
            Point attribute to use for true particles
        run_mode : str, default 'both'
            Which output to run on (one of 'both', 'reco' or 'truth')
        '''
        # Fetch the functions that map the range to a KE
        self.include_pids = include_pids
        self.splines = {ptype: csda_table_spline(ptype) \
                for ptype in include_pids}

        # Store the tracking parameters
        self.tracking_mode = tracking_mode
        self.kwargs = kwargs

        # List objects for which to reconstruct track KE
        if run_mode not in ['reco', 'truth', 'both']:
            raise ValueError('`run_mode` must be either `reco`, ' \
                    '`truth` or `both`')

        self.key_list = []
        if run_mode in ['reco', 'both']:
            self.key_list += ['particles']
        if run_mode in ['truth', 'both']:
            self.key_list += ['truth_particles']
        self.truth_point_mode = truth_point_mode

    def process(self, data_dict, result_dict):
        '''
        Reconstruct the CSDA KE estimates for each particle in one entry

        Parameters
        ----------
        data_dict : dict
            Input data dictionary
        result_dict : dict
            Chain output dictionary
        '''
        # Loop over particle objects
        for k in self.key_list:
            for p in result_dict[k]:
                # Only run this algorithm on tracks that have a CSDA table
                if not ((p.semantic_type == 1) \
                        and (p.pid in self.include_pids)):
                    continue
                
                # Make sure the particle coordinates are expressed in cm
                self.check_units(p)

                # Get point coordinates
                points = self.get_points(p)
                if not len(points):
                    continue

                # Compute the length of the track
                length = get_track_length(points, point=p.start_point,
                        method=self.tracking_mode, **self.kwargs)

                # Store the length and the CSDA kinetic energy
                p.length  = length
                p.csda_ke = self.splines[p.pid](length).item() \
                        if length > 0. else 0.

        return {}, {}
