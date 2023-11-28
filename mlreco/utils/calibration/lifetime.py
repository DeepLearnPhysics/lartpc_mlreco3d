import numpy as np

from .database import CalibrationDatabase

from mlreco.utils.geometry import Geometry


class LifetimeCalibrator:
    '''
    Applies a correction based on drift electron lifetime and the distance
    from the ionization point to the closest readout plane.
    '''
    name = 'lifetime'

    def __init__(self,
                 detector = None,
                 boundary_file = None,
                 source_file = None,
                 lifetime = None,
                 driftv = None,
                 lifetime_db = None,
                 driftv_db = None):
        '''
        Load the information needed to make a lifetime correction.

        Parameters
        ----------
        detector : str, optional
            Detector to get the geometry from
        boundary_file : str, optional
            Path to a detector boundary file. Supersedes `detector` if set
        source_file : str, optional
            Path to a detector source file. Supersedes `detector` if set
        lifetime : Union[float, list], optional
            Specifies the electron lifetime in microseconds. If `list`, it
            should map a tpc ID onto a specific value.
        driftv : Union[float, list], optional
            Specifies the electron drift velocity in cm/us. If `dict`, it
            should map a tpc ID onto a specific value.
        lifetime_db : str, optional
            Path to a SQLite db file which maps [run, cryo, tpc] sets onto
            a specific lifetime value in microseconds.
        driftv_db : str, optional
            Path to a SQLite db file which maps [run, cryo, tpc] sets onto
            a specific electron drift velocity value in cm/us.
        '''
        # Initialize the geometry
        self.geo = Geometry(detector, boundary_file, source_file)

        # Load the database, which maps run numbers onto a lifetime/drift v
        assert (lifetime is not None and driftv is not None) \
                ^ (lifetime_db is not None and driftv_db is not None), \
                'Must specify static values of the lifetime and drift ' \
                'velocity or paths to databases that provide it.'

        # If static values are specified, store them
        if lifetime:
            # Set the method
            self.use_db = False

            # Inititalize lifetime
            if np.issclar(lifetime):
                self.lifetime = np.full(self.geo.num_tpcs, lifetime)
            else:
                assert len(lifetime) == self.geo.num_tpcs, \
                        '`lifetime` list must provide one value per TPC'
                self.lifetime = lifetime

            # Initialize electron drift velocity
            if np.isscalar(driftv):
                self.driftv = np.full(self.geo.num_tpcs, driftv)
            else:
                assert len(driftv) == self.geo.num_tpcs, \
                        '`driftv` list must provide one value per TPC'
                self.drift = driftv

        # If databases are provided, load them in
        if lifetime_db:
            # Set the method
            self.use_db = True

            # Inialize lifetime database
            self.lifetime = CalibrationDatabase(lifetime_db, self.geo.num_tpcs)

            # Initialize electron drift velocity database
            self.driftv = CalibrationDatabase(driftv_db, self.geo.num_tpcs)

    def process(self, points, values, sources=None, run_id=None):
        '''
        Apply the lifetime correction.

        Parameters
        ----------
        points : np.ndarray
            (N, 3) array of point coordinates
        values : np.ndarray
            (N) array of values associated with each point
        sources : np.ndarray, optional
            (N) array of [cryo, tpc] specifying which TPC produced each hit. If
            not specified, uses the closest anode as a sensitive plane.
        run_id : int, optional
            If provided, used to get the appropriate lifetime/drift velocities

        Returns
        -------
        np.ndarray
            (N) array of corrected values
        '''
        # Get the corrections lifetimes/drift velocities
        lifetime = self.lifetime
        driftv = self.driftv
        if self.use_db:
            assert run_id is not None, \
                    'When using the database, must provide a run ID'
            lifetime = self.lifetime[run_id]
            driftv = self.driftv[run_id]

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

            # Compute the distance to the anode plane
            daxis, position = self.geo.anodes[module_id, tpc_id]
            drifts = np.abs(points[tpc_index, daxis] - position)

            # Clip down to the physical range of possible drift distances
            max_drift = self.geo.ranges[module_id, tpc_id][daxis]
            drifts = np.clip(drifts, 0., max_drift)

            # Convert the drift distances to correction factors
            corrections = np.exp(drifts/lifetime[t]/driftv[t])
            new_values[tpc_index] = corrections * values

        # Scale and return
        return new_values
