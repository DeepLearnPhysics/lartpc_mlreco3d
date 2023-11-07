import numpy as np

from mlreco.utils.globals import TRACK_SHP
from mlreco.utils.energy_loss import csda_table_spline
from mlreco.utils.tracking import get_track_length

from analysis.post_processing import PostProcessor


class CSDAEnergyProcessor(PostProcessor):
    '''
    Reconstruct the kinetic energy of tracks based on their range in liquid
    argon using the continuous slowing down approximation (CSDA).
    '''
    name = 'reconstruct_csda_energy'
    result_cap = ['particles']
    result_cap_opt = ['truth_particles']

    def __init__(self,
                 tracking_mode='step_next',
                 include_pids=[2,3,4,5],
                 truth_point_mode='points',
                 run_mode = 'both',
                 **kwargs):
        '''
        Store the necessary attributes to do CSDA range-based estimations

        Parameters
        ----------
        tracking_mode : str, default 'step_next'
            Method used to compute the track length (one of 'displacement',
            'step', 'step_next', 'bin_pca' or 'spline')
        include_pids : list, default [2, 3, 4, 5]
            Particle species to compute the kinetic energy for
        **kwargs : dict, optional
            Additional arguments to pass to the tracking algorithm
        '''
        # Initialize the parent class
        super().__init__(run_mode, truth_point_mode)

        # Fetch the functions that map the range to a KE
        self.include_pids = include_pids
        self.splines = {ptype: csda_table_spline(ptype) \
                for ptype in include_pids}

        # Store the tracking parameters
        self.tracking_mode = tracking_mode
        self.tracking_kwargs = kwargs

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
        for k in self.part_keys:
            for p in result_dict[k]:
                # Only run this algorithm on tracks that have a CSDA table
                if not ((p.semantic_type == TRACK_SHP) \
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
                        method=self.tracking_mode, **self.tracking_kwargs)

                # Store the length and the CSDA kinetic energy
                p.length  = length
                p.csda_ke = self.splines[p.pid](length).item() \
                        if length > 0. else 0.

        return {}, {}
