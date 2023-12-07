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
                 num_tpcs,
                 lifetime = None,
                 driftv = None,
                 lifetime_db = None,
                 driftv_db = None):
        '''
        Load the information needed to make a lifetime correction.

        Parameters
        ----------
        num_tpcs : int
            Number of TPCs in the detector
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
                self.lifetime = np.full(num_tpcs, lifetime)
            else:
                assert len(lifetime) == num_tpcs, \
                        '`lifetime` list must provide one value per TPC'
                self.lifetime = lifetime

            # Initialize electron drift velocity
            if np.isscalar(driftv):
                self.driftv = np.full(num_tpcs, driftv)
            else:
                assert len(driftv) == num_tpcs, \
                        '`driftv` list must provide one value per TPC'
                self.drift = driftv

        # If databases are provided, load them in
        if lifetime_db:
            # Set the method
            self.use_db = True

            # Inialize lifetime database
            self.lifetime = CalibrationDatabase(lifetime_db, num_tpcs)

            # Initialize electron drift velocity database
            self.driftv = CalibrationDatabase(driftv_db, num_tpcs)

    def process(self, points, values, geo, tpc_id, run_id=None):
        '''
        Apply the lifetime correction.

        Parameters
        ----------
        points : np.ndarray
            (N, 3) array of point coordinates
        values : np.ndarray
            (N) array of values associated with each point
        geo : Geometry
            Detector geometry object
        tpc_id : int
            ID of the TPC to use
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

        # Compute the distance to the anode plane
        m, t = tpc_id // geo.num_modules, tpc_id % geo.num_modules
        daxis, position = geo.anodes[m, t]
        drifts = np.abs(points[:, daxis] - position)

        # Clip down to the physical range of possible drift distances
        max_drift = geo.ranges[m, t][daxis]
        drifts = np.clip(drifts, 0., max_drift)

        # Convert the drift distances to correction factors
        corrections = np.exp(drifts/lifetime[tpc_id]/driftv[tpc_id])
        return corrections * values
