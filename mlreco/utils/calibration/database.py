import numpy as np
import pandas as pd
import sqlite3 as sql
from pathlib import Path


class CalibrationDatabase:
    '''
    Wraps basic SQLite loading/querying functions to provide a more
    user-friendly API to the calibration classes.

    Notes
    -----
    This class assumes that the structure of the SQLite libraries used
    is that of ICARUS calibration databases, for now.
    '''
    def __init__(self, db_path, num_tpcs, db_type='value', value_key='scale'):
        '''
        Given a path to a calibration data base, load the information
        into a dictionary.

        Parameters
        ----------
        db_path : str
            Path to a SQLite database
        num_tpcs : int
            Expected number of TPCs
        db_type : str, default 'value'
            Type of database (One 'value' or one 'map per TPC)
        value_key : str, default 'scale'
            Name of the quantity to load for each bin when using 'map' db_type

        Returns
        -------
        dict
            Dictionary which maps a run onto a set of values (one per TPC)

        Notes
        -----
        This makes assumptions about how the database is structured for
        ICARUS calibration for now as of the time of implementation.
        '''
        # Make sure the type of database is recognized
        if db_type not in ['value', 'map']:
            raise ValueError(f'Type of database not recognized: {db_type}. ' \
                    'Must be either `value` or `map`.')

        # Load the database into a pandas dataframe
        stem = Path(db_path).stem
        quantity = '_'.join(stem.split('_')[1:-1])

        db = sql.connect(db_path)
        df = pd.read_sql_query(f'SELECT * from {stem}_data', db)
        run_df = pd.read_sql_query(f'SELECT * from {stem}_iovs', db)
        db.close()

        df = df.merge(run_df, left_on='__iov_id', right_on='iov_id')
        df = df[df.active == 1]

        # Loop over unique runs, store the values per TPCs for each run
        self.num_tpcs = num_tpcs
        self.dict = {}
        for run in np.unique(df.begin_time):
            df_run = df[df.begin_time == run]
            run_id = run - int(1e9)
            if db_type == 'value':
                self.dict[run_id] = self.load_values(df_run, quantity)
            else:
                self.dict[run_id] = self.load_tables(df_run, value_key)

        # Create a list of boundary runs
        self.runs = np.sort(list(self.dict.keys()))

    def load_values(self, df_run, quantity):
        '''
        Loads one value per TPC

        Parameters
        ----------
        df_run : pd.DataFrame
            Dataframe which corresponds to the run being loaded
        quantity : str
            Name of the quantity to load

        Returns
        -------
        np.ndarray
            (N_tpc) Array of calibration values
        '''
        # Check that there is exactly one value per tpc
        assert len(df_run) == self.num_tpcs, \
                'There should be one quantity specified per TPC'

        # Store the values into an array
        array = np.empty(self.num_tpcs)
        for i in range(len(df_run)):
            channel = int(df_run.iloc[i].channel)
            value = df_run.iloc[i][quantity]
            array[channel] = value

        return array

    def load_tables(self, df_run, quantity):
        '''
        Loads one look-up table per TPC

        Parameters
        ----------
        df_run : pd.DataFrame
            Dataframe which corresponds to the run being loaded
        quantity : str
            Name of the quantity to load for each bin

        Returns
        -------
        np.ndarray
            (N_tpc) Array of calibration look-up tables
        '''
        tpc_luts = []
        tpc_keys = ['EE', 'EW', 'WE', 'WW']
        for tpc_id, tpc_key in enumerate(tpc_keys):
            df_tpc = df_run[df_run.tpc == tpc_key]
            bins_y = np.max(df_tpc.ybin) + 1
            bins_z = np.max(df_tpc.zbin) + 1
            range_y = [np.min(df_tpc.ylow), np.max(df_tpc.yhigh)]
            range_z = [np.min(df_tpc.zlow), np.max(df_tpc.zhigh)]
            values = df_tpc[quantity].to_numpy().reshape(bins_y, bins_z)

            lut = CalibrationLUT([1,2], [bins_y, bins_z],
                    [range_y, range_z], values)
            tpc_luts.append(lut)

        return tpc_luts

    def __getitem__(self, run_id):
        '''
        Mirrors the `query` function.

        Parameters
        ----------
        run_id : int
            ID of the run to get the values for

        Returns
        -------
        np.ndarray
            List of values per channel
        '''
        return self.query(run_id)

    def query(self, run_id):
        '''
        Gets the database information for a given run. If the run does not
        exist in the list, pick the one closest but earlier than it.

        Parameters
        ----------
        run_id : int
            ID of the run to get the values for

        Returns
        -------
        np.ndarray
            List of values per channel
        '''
        # Identify the closest run that is before the queried run
        if run_id < self.runs[0]:
            raise IndexError('No calibration information for run ' \
                    f'{run_id} < {self.runs[0]}')

        closest_run = self.runs[np.where(self.runs <= run_id)[0][-1]]

        return self.dict[closest_run]


class CalibrationLUT:
    '''
    Look-up table for calibration values. Given a set of coordinates,
    returns a calibration value.
    '''
    def __init__(self, dims, bins, range, values):
        '''
        Initialize the calibration map

        Parameters
        ----------
        dims : List[int]
            List of dimensions (0: x, 1: y, 2: z)
        bins : List[int]
            Number of bins in each dimension
        range : List[List[float]]
            Axis range in each dimension
        values : np.ndarray
            Values in each bin
        '''
        # Store metadata information
        assert len(range) == len(dims) and len(bins) == len(dims), \
                'Must provide a bin count and range per dimension'
        self.dims = dims
        self.range = np.array(range)
        self.bins = np.array(bins)
        self.bin_sizes = (self.range[:,1]-self.range[:,0])/self.bins

        # Store the values in each bin. Should be a dense matrix
        assert np.all(values.shape == self.bins), \
                'Must provide one calibration value per bin'
        self.values = values

    def query(self, points):
        '''
        Queries the LUT to get the calibration values for a set of points.

        Parameters
        ----------
        points: np.ndarry
            (N, 3) Coordinates of the points to query a calibration for

        Returns
        -------
        np.ndarray
            Calibration constants
        '''
        # Get the bin the position belongs to:
        offsets = points[:,self.dims] - self.range[:,0]
        bin_ids = (offsets/self.bin_sizes).astype(int)

        # Collapse to the closest bin if it is outisde of range
        #assert np.all(bin_ids > -1) and np.all(bin_ids < self.bins), \
        #        'Some of the points fall outside of the look-up table'
        bad_mask = np.where(bin_ids < 0)
        bin_ids[bad_mask] = 0
        bad_mask = np.where(bin_ids >= self.bins)
        bin_ids[bad_mask] = self.bins[bad_mask[-1]] - 1

        # Get the corrections
        return self.values[tuple(bin_ids.T)]
