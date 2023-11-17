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
    def __init__(self, db_path, size = None):
        '''
        Given a path to a calibration data base, load the information
        into a dictionary.

        Parameters
        ----------
        db_path : str
            Path to a SQLite database
        size : int, optional
            Expected number of channels per run ID

        Returns
        -------
        dict
            Dictionary which maps a run onto a set of values (one per TPC)
        
        Notes
        -----
        This makes assumptions about how the database is structured for
        ICARUS calibration for now as of the time of implementation.
        '''
        # Load the database into a pandas dataframe
        stem = Path(db_path).stem
        quantity = stem.split('_')[1]

        db = sql.connect(db_path)
        df = pd.read_sql_query(f'SELECT * from {stem}_data', db)
        run_df = pd.read_sql_query(f'SELECT * from {stem}_iovs', db)
        db.close()

        df = df.merge(run_df, left_on='__iov_id', right_on='iov_id')
        df = df[df.active == 1]

        # Loop over unique runs, store the values per TPCs for each run
        self.dict = {}
        for run in np.unique(df.begin_time):
            # Narrow down the run, check that the
            df_run = df[df.begin_time == run]
            assert size is None or len(df_run) == size, \
                    'There should be one quantity specified per TPC'

            run = run - int(1e9)
            self.dict[run] = np.empty(size) if size else np.empty(len(df_run))
            for i in range(len(df_run)):
                channel = int(df_run.iloc[i].channel)
                value = df_run.iloc[i][quantity]
                self.dict[run][channel] = value

        # Create a list of boundary runs
        self.runs = np.sort(list(self.dict.keys()))

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
