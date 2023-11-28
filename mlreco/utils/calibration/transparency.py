import numpy as np

from .database import CalibrationDatabase

from mlreco.utils.geometry import Geometry


class TransparencyCalibrator:
    '''
    Applies a correction on the amount of charge observed in a space point
    based on its position in the plane of the sensitive wires/pixels (yz).
    '''
    name = 'transparency'

    def __init__(self,
                 transparency_db,
                 detector = None,
                 boundary_file = None,
                 value_key='scale'):
        '''
        Load the calibration maps

        Parameters
        ----------
        lifetime_db : str
            Path to a SQLite db file which maps [run, cryo, tpc] sets onto
            a specific transparency calibration map.
        detector : str, optional
            Detector to get the geometry from
        boundary_file : str, optional
            Path to a detector boundary file. Supersedes `detector` if set
        '''
        # Initialize the geometry
        self.geo = Geometry(detector, boundary_file)

        # Load the transparency database
        self.transparency = CalibrationDatabase(transparency_db,
                num_tpcs=self.geo.num_tpcs, db_type='map', value_key=value_key)

    def process(self, points, values, run_id, sources=None):
        '''
        Apply the transparency correction.

        Parameters
        ----------
        points : np.ndarray
            (N, 3) array of point coordinates
        values : np.ndarray
            (N) array of values associated with each point
        run_id : int
            If provided, used to get the appropriate transparency map
        sources : np.ndarray, optional
            (N) array of [cryo, tpc] specifying which TPC produced each hit. If
            not specified, uses the closest anode as a sensitive plane.

        Returns
        -------
        np.ndarray
            (N) array of corrected values
        '''
        # Get the appropriate transparency map for this run
        transparency = self.transparency[run_id]

        # If there is no source and the lifetime is specified separately for
        # each TPC, assign TPC which is closest the each point
        if sources is None:
            tpc_indexes = self.geo.get_closest_tpc_indexes(points)

        # Loop over the unique TPCs, correct individually
        new_values = np.empty(len(values), dtype=values.dtype)
        for t in range(self.geo.num_tpcs):
            # Get the set of points associated with this TPC
            module_id = t // self.geo.num_modules
            tpc_id = t % self.geo.num_modules
            if sources is not None:
                tpc_index = self.geo.get_tpc_index(sources, module_id, tpc_id)
            else:
                tpc_index = tpc_indexes[t]

            if not len(tpc_index):
                continue

            # Convert the drift distances to correction factors
            corrections = 1./transparency[t].query(points[tpc_index])
            new_values[tpc_index] = corrections * values

        # Scale and return
        return new_values


