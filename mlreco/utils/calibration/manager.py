import numpy as np

from mlreco.utils.geometry import Geometry

from .factories import calibrator_factory


class CalibrationManager:
    '''
    Manager in charge of applying all calibration-related corrections onto
    a set of 3D space points and their associated measured charge depositions.
    '''
    def __init__(self, cfg, parent_path=''):
        '''
        Initialize the manager

        Parameters
        ----------
        cfg : dict
            Calibrator configurations
        parent_path : str, optional
            Path to the analysis tools configuration file
        '''
        # Initialize the geometry model shared across all modules
        assert 'geometry' in cfg, \
                'Must provide a geometry configuration to apply calibrations'
        self.geo = Geometry(**cfg.pop('geometry'))

        # Make sure the essential calibration modules are present
        assert 'gain' in cfg, \
                'Must provide an ADC to number electrons conversion factor'
        assert 'recombination' in cfg, \
                'Must provide a recombination model to convert from number ' \
                'of electrons to an energy deposition in MeV'

        # Add the modules to a processor list in decreasing order of priority
        self.modules   = {}
        self.profilers = {} # TODO: Replace this with StopWatchManager
        for k in cfg.keys():
            # If requested, profile the module (default True)
            profile = cfg[k].pop('profile') if 'profile' in cfg[k] else True
            if profile:
                self.profilers[k] = 0.

            # Add necessary geometry information
            if k != 'recombination':
                cfg[k]['num_tpcs'] = self.geo.num_tpcs
            else:
                cfg[k]['drift_dir'] = self.geo.drift_dirs[0, 0]

            # Append
            self.modules[k] = calibrator_factory(k, cfg[k], parent_path)

    def process(self, points, values, sources=None, run_id=None,
            dedx=None, track=None):
        '''
        Main calibration driver

        Parameters
        ----------
        points : np.ndarray, optional
            (N, 3) array of space point coordinates
        values : np.ndarray
            (N) array of depositions in ADC
        sources : np.ndarray, optional
            (N) array of [cryo, tpc] specifying which TPC produced each hit. If
            not specified, uses the closest TPC as calibration reference.
        run_id : int, optional
            ID of the run to get the calibration for. This is needed when using
            a database of corrections organized by run.
        dedx : float, optional
            If specified, use a flat value of dE/dx in MeV/cm to apply
            the recombination correction.
        track : bool, defaut `False`
            Whether the object is a track or not. If it is, the track gets
            segmented to evaluate local dE/dx and track angle.

        Returns
        -------
        np.ndarray
            (N) array of calibrated depositions in MeV
        '''
        # Reset the profilers
        for key in self.profilers:
            self.profilers[key] = 0.

        # Create a mask for each of the TPC volume in the detector
        if sources is not None:
            tpc_indexes = []
            for t in range(self.geo.num_tpcs):
                # Get the set of points associated with this TPC
                module_id = t // self.geo.num_modules
                tpc_id = t % self.geo.num_modules
                tpc_index = self.geo.get_tpc_index(sources, module_id, tpc_id)
                tpc_indexes.append(tpc_index)
        else:
            assert points is not None, \
                    'If sources are not given, must provide points instead'
            tpc_indexes = self.geo.get_closest_tpc_indexes(points)

        # Loop over the TPCs, apply the relevant calibration corrections
        new_values = np.copy(values)
        for t in range(self.geo.num_tpcs):
            # Restrict to the TPC of interest
            if not len(tpc_indexes[t]):
                continue
            tpc_points = points[tpc_indexes[t]]
            tpc_values = values[tpc_indexes[t]]

            # Apply the transparency correction
            if 'transparency' in self.modules:
                assert run_id is not None, \
                        'Must provide a run ID to get the transparency map'
                tpc_values = self.modules['transparency'].process(tpc_points,
                        tpc_values, t, run_id) # ADC

            # Apply the lifetime correction
            if 'lifetime' in self.modules:
                tpc_values = self.modules['lifetime'].process(tpc_points,
                        tpc_values, self.geo, t, run_id) # ADC

            # Apply the gain correction
            tpc_values = self.modules['gain'].process(tpc_values, t) # e-

            # Apply the recombination
            tpc_values = self.modules['recombination'].process(tpc_values,
                    tpc_points, dedx, track) # MeV

            # Append
            new_values[tpc_indexes[t]] = tpc_values

        return new_values
