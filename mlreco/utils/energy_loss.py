import os
import pathlib
import pandas as pd

from functools import lru_cache
from scipy.interpolate import CubicSpline
from mlreco.utils.globals import PDG_TO_PID, ARGON_DENSITY


@lru_cache
def csda_table_spline(particle_type, table_dir='csda_tables'):
    '''
    Interpolates a CSDA table to form a spline which maps
    a range to a kinematic energy estimate.

    Parameters
    ----------
    particle_type : int
        Particle type ID to construct splines. Maps are
        avaible for muons, pions, kaons and protons.
    table_dir : str, default 'csda_tables'
        Relative path to the CSDA range tables

    Returns
    -------
    callable
        Function mapping range (cm) to Kinetic E (MeV)
    '''
    path = pathlib.Path(__file__).parent
    suffix = 'E_liquid_argon'
    name_mapping = {PDG_TO_PID[13]: 'mu',
                    PDG_TO_PID[211]: 'pi',
                    PDG_TO_PID[321]: 'ka',
                    PDG_TO_PID[2212]: 'p'}

    if particle_type in name_mapping.keys():
        pid = name_mapping[particle_type]
        file_name = os.path.join(path, table_dir, f'{pid}{suffix}')
        if os.path.isfile(f'{file_name}.txt'):
            path = f'{file_name}.txt'
        else:
            path = f'{file_name}_bethe.txt'

        tab = pd.read_csv(path, delimiter=' ', index_col=False)

    else:
        raise ValueError(f'CSDA table for particle type {particle_type} is not available')

    f = CubicSpline(tab['CSDARange'] / ARGON_DENSITY, tab['T'])
    return f
