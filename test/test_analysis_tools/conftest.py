import pytest
import os, sys

from mlreco.iotools.readers import HDF5Reader
from analysis.classes.builders import ParticleBuilder, InteractionBuilder

import pathlib
from pprint import pprint
from collections import defaultdict

file_key_load = '../data/nue_corsika_test.h5'
file_key_build = '../data/nue_corsika_test_raw.h5'

@pytest.fixture(scope='session')
def bnb_nue_cosmic_organized():
    reader = HDF5Reader(file_key_load, to_larcv=True)
    num_entries = reader.num_entries
    data_dict, result_dict = defaultdict(list), defaultdict(list)
    for i in range(num_entries):
        data, res = reader.get(i, nested=True)
        for key, val in data.items():
            data_dict[key].extend(val)
        for key, val in res.items():
            result_dict[key].extend(val)
    return data_dict, result_dict

@pytest.fixture(scope='session')
def bnb_nue_cosmic_raw():
    reader = HDF5Reader(file_key_build, to_larcv=True)
    num_entries = reader.num_entries
    data_dict, result_dict = defaultdict(list), defaultdict(list)
    for i in range(num_entries):
        data, res = reader.get(i, nested=True)
        for key, val in data.items():
            data_dict[key].extend(val)
        for key, val in res.items():
            result_dict[key].extend(val)
    return data_dict, result_dict

def test_load_hdf5(bnb_nue_cosmic_raw,
                   bnb_nue_cosmic_organized):
    pass