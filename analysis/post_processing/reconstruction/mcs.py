import numpy as np

from mlreco.utils.globals import TRACK_SHP, PID_MASSES
from mlreco.utils.tracking import get_track_segments
from mlreco.utils.mcs import mcs_fit

from analysis.post_processing import PostProcessor


class MCSEnergyProcessor(PostProcessor):
    '''
    Reconstruct the kinetic energy of tracks based on their Multiple-Coulomb
    scattering (MCS) angles while passing through liquid argon.
    '''
    name = 'reconstruct_mcs_energy'
    result_cap = ['particles']
    result_cap_optional = ['truth_particles']

    def __init__(self,
                 tracking_mode='bin_pca',
                 segment_length=5,
                 include_pids=[2,3,4,5],
                 only_uncontained=False,
                 truth_point_mode='points',
                 run_mode = 'both',
                 **kwargs):
        '''
        Store the necessary attributes to do MCS-based estimations

        Parameters
        ----------
        tracking_mode : str, default 'bin_pca'
            Method used to compute the segment angles (one of 'step',
            'step_next' or 'bin_pca')
        segment_length : float, default 5 cm
            Segment length in the units that specify the coordinates
        include_pids : list, default [2, 3, 4, 5]
            Particle species to compute the kinetic energy for
        only_uncontained : bool, default False
            Only run the algorithm on particles that are marked as not contained
        **kwargs : dict, optiona
            Additional arguments to pass to the tracking algorithm
        '''
        # Initialize the parent class
        super().__init__(run_mode, truth_point_mode)

        # Store the general parameters
        self.include_pids = include_pids
        self.only_uncontained = only_uncontained

        # Store the tracking parameters
        assert tracking_mode in ['step', 'step_next', 'bin_pca'], \
                'The tracking algorithm must provide segment angles'
        self.tracking_mode = tracking_mode
        self.segment_length = segment_length
        self.tracking_kwargs = kwargs

    def process(self, data_dict, result_dict):
        '''
        Reconstruct the MCS KE estimates for each particle in one entry

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
                # Only run this algorithm on particle species that are needed
                if not ((p.semantic_type == TRACK_SHP) \
                        and (p.pid in self.include_pids)):
                    continue
                if self.only_uncontained and p.is_contained:
                    continue

                # Make sure the particle coordinates are expressed in cm
                self.check_units(p)

                # Get point coordinates
                points = self.get_points(p)
                if not len(points):
                    continue

                # Get the list of segment directions
                _, dirs, _ = get_track_segments(points, self.segment_length,
                        p.start_point, method=self.tracking_mode,
                        **self.tracking_kwargs)

                # Find the angles between successive segments
                costh = np.sum(dirs[:-1] * dirs[1:], axis = 1)
                theta = np.arccos(costh)
                if len(theta) < 1:
                    continue

                # Store the length and the MCS kinetic energy
                mass = PID_MASSES[p.pid]
                p.mcs_ke = mcs_fit(theta, mass, self.segment_length)

        return {}, {}
