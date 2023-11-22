import numpy as np

from mlreco.utils.geometry import Geometry


class GainCalibrator:
    '''
    Converts all charge depositions in ADC to a number of electrons. It can
    either use a flat converstion factor or one per TPC in the detector
    '''
    name = 'gain'

    def __init__(self,
                 gain,
                 detector = None,
                 boundary_file = None,
                 source_file = None):
        '''
        Initialize the recombination model and its constants.

        Parameters
        ----------
        gain : Union[list, float], optional
            Conversion factor from ADC to electrons (unique or per tpc)
        detector : str, optional
            Detector to get the geometry from
        boundary_file : str, optional
            Path to a detector boundary file. Supersedes `detector` if set
        source_file : str, optional
            Path to a detector source file. Supersedes `detector` if set
        '''
        # Initialize the geometry
        self.geo = Geometry(detector, boundary_file, source_file)

        # Initialize the gain values
        assert np.isscalar(gain) or len(gain) == self.geo.num_tpcs, \
                'Gain must be a single value or given per TPC'
        self.gain = gain

    def process(self, values, points=None, sources=None):
        '''
        Converts deposition values from ADC to 

        Parameters
        ----------
        values : np.ndarray
            (N) array of depositions in ADC
        points : np.ndarray, optional
            (N, 3) array of point coordinates associated with the charge
            depositions. Only needed if `sources` is not provided and if the
            gain is different in each TPC.
        sources : np.ndarray, optional
            (N) array of [cryo, tpc] specifying which TPC produced each hit. If
            not specified, uses the closest anode as a sensitive plane.

        Returns
        -------
        np.ndarray
            (N) array of depositions in number of electrons
        '''
        # If the gain is specified globally, use it as is
        if np.isscalar(self.gain):
            return values * self.gain

        # If there is no source and the gain is specified separately for each
        # TPC, assign TPC which is closest the each point
        if sources is None:
            assert points is not None, \
                    'If sources are not given, must provide points instead'
            tpc_indexes = self.geo.get_closest_tpc_indexes(points)

        # Loop over the unique TPCs, apply gain separately
        corr_values = np.empty(len(values), dtype=values.dtype)
        for t in range(self.geo.num_tpcs):
            # Get the set of points associated with this TPC
            module_id = t // self.geo.num_modules
            tpc_id = t % self.geo.num_modules
            if sources is not None:
                tpc_index = self.geo.get_tpc_index(sources, module_id, tpc_id)
            else:
                tpc_index = tpc_indexes[t]

            # Skip if there is no points in this TPC
            if not len(tpc_index):
                continue

            # Fetch the correct gain for this set of points
            corr_values[tpc_index] = self.gain[t] * values[tpc_index]

        return corr_values
