import os
import yaml
import h5py
import inspect
import numpy as np
from collections import defaultdict
from larcv import larcv
from analysis import classes as analysis


class HDF5Writer:
    '''
    Class which builds an HDF5 file to store the input
    and/or the output of the reconstruction chain. It
    can also be used to append an existing HDF5 file with
    information coming out of the analysis tools.

    More documentation to come.
    '''

    # LArCV object attributes that do not need to be stored to HDF5
    LARCV_SKIP = [
            'add_trajectory_point', 'dump', 'momentum', 'boundingbox_2d', 'boundingbox_3d',
            *[k + a for k in ['', 'parent_', 'ancestor_'] for a in ['x', 'y', 'z', 't']]
    ]

    # Analysis object attributes that do not need to be stored to HDF5
    ANA_SKIP = [
        'points', 'true_points', 'particles', 'fragments', 'asis',
        'depositions', 'depositions_MeV', 'true_depositions', 'true_depositions_MeV',
        'particles_summary'
        # 'index', 'true_index'
    ]

    # List of recognized LArCV objects
    LARCV_DATAOBJS = [
        larcv.Particle,
        larcv.Neutrino,
        larcv.Flash,
        larcv.CRTHit
    ]

    # List of recognized Analysis objects
    ANA_DATAOBJS = [
        analysis.ParticleFragment,
        analysis.TruthParticleFragment,
        analysis.Particle,
        analysis.TruthParticle,
        analysis.Interaction,
        analysis.TruthInteraction
    ]

    # List of recognized objects
    DATAOBJS = tuple(LARCV_DATAOBJS + ANA_DATAOBJS)

    def __init__(self,
                 file_name: str = 'output.h5',
                 input_keys: list = None,
                 skip_input_keys: list = [],
                 result_keys: list = None,
                 skip_result_keys: list = [],
                 append_file: bool = False):
        '''
        Initializes the basics of the output file

        Parameters
        ----------
        file_name : str, default 'output.h5'
            Name of the output HDF5 file
        input_keys : list, optional
            List of input keys to store. If not specified, stores all of the input keys
        skip_input_keys: list, optional
            List of input keys to skip
        result_keys : list, optional
            List of result keys to store. If not specified, stores all of the result keys
        skip_result_keys: list, optional
            List of result keys to skip
        append_file: bool, default False
            Add new values to the end of an existing file
        '''
        # Store attributes
        self.file_name        = file_name
        self.input_keys       = input_keys
        self.skip_input_keys  = skip_input_keys
        self.result_keys      = result_keys
        self.skip_result_keys = skip_result_keys
        self.append_file      = append_file
        self.ready            = False
        self.object_dtypes    = {}

    def create(self, data_blob, result_blob=None, cfg=None):
        '''
        Create the output file structure based on the data and result blobs.

        Parameters
        ----------
        data_blob : dict
            Dictionary containing the input data
        result_blob : dict
            Dictionary containing the output of the reconstruction chain
        cfg : dict
            Dictionary containing the ML chain configuration
        '''
        # Make sure there is something to store
        assert data_blob or result_blob, 'Must provide a non-empty data blob or result blob'

        # Get the expected batch_size (index is alaways provided by the reco. chain)
        self.batch_size = len(data_blob['index'])

        # Initialize a dictionary to store keys and their properties (dtype and shape)
        self.key_dict = defaultdict(lambda: {'category': None, 'dtype':None, 'width':0, 'merge':False, 'scalar':False, 'larcv':False})

        # If requested, loop over input_keys and add them to what needs to be tracked
        if self.input_keys is None: self.input_keys = data_blob.keys()
        self.input_keys = set(self.input_keys)
        if 'index' not in self.input_keys:
            self.input_keys.add('index')
        for key in self.skip_input_keys:
            if key in self.input_keys:
                self.input_keys.remove(key)
        for key in self.input_keys:
            self.register_key(data_blob, key, 'data')

        # If requested, loop over the result_keys and add them to what needs to be tracked
        if self.result_keys is None: self.result_keys = result_blob.keys() if result_blob is not None else []
        self.result_keys = set(self.result_keys)
        for key in self.skip_result_keys:
            if key in self.result_keys:
                self.result_keys.remove(key)
        for key in self.result_keys:
            self.register_key(result_blob, key, 'result')

        # Initialize the output HDF5 file
        with h5py.File(self.file_name, 'w') as file:
            # Initialize the info dataset that stores top-level description of what is stored
            if cfg is not None:
                file.create_dataset('info', (0,), maxshape=(None,), dtype=None)
                file['info'].attrs['cfg'] = yaml.dump(cfg)

            # Initialize the event dataset and the corresponding reference array datasets
            self.initialize_datasets(file)

            # Mark file as ready for use
            self.ready = True

    def add_keys(self, result_blob):
        '''
        Add more keys to the results group of an existing file.

        Parameters
        ----------
        result_blob : dict
            Dictionary containing the additional output to store
        '''
        # Make sure there is something to store
        assert result_blob, 'Must provide a non-empty result blob'

        # Get the expected batch_size from the data_blob (index must be present)
        self.batch_size = 1

        # Initialize a dictionary to store keys and their properties (dtype and shape)
        self.key_dict = defaultdict(lambda: {'category': None, 'dtype':None, 'width':0, 'merge':False, 'scalar':False})

        # Loop over the result_keys and add them to what needs to be tracked
        self.result_keys = list(result_blob.keys())
        for key in self.result_keys:
            self.register_key(result_blob, key, 'result')

        # Initialize the output HDF5 file
        with h5py.File(self.file_name, 'a') as file:
            # Initialize the event dataset and the corresponding reference array datasets
            self.initialize_datasets(file)

            # Mark file as ready for use
            self.ready = True

    def register_key(self, blob, key, category):
        '''
        Identify the dtype and shape objects to be dealt with.

        Parameters
        ----------
        blob : dict
            Dictionary containing the information to be stored
        key : string
            Dictionary key name
        category : string
            Data category: `data` or `result`
        '''
        # Store the necessary information to know how to store a key
        self.key_dict[key]['category'] = category
        if self.is_scalar(blob[key]):
            # Single scalar
            self.key_dict[key]['dtype']  = h5py.string_dtype() if isinstance(blob[key], str) else type(blob[key])
            self.key_dict[key]['scalar'] = True

        else:
            if len(blob[key]) != self.batch_size: # TODO: Get rid of this possibility upstream
                # List with a single scalar, regardless of batch_size
                assert len(blob[key]) == 1 and self.is_scalar(blob[key][0]),\
                        'If there is an array of length mismatched with batch_size, '+\
                        'it must contain a single scalar.'

            if self.is_scalar(blob[key][0]):
                # List containing a single scalar per batch ID
                self.key_dict[key]['dtype']  = h5py.string_dtype() if isinstance(blob[key][0], str) else type(blob[key][0])
                self.key_dict[key]['scalar'] = True

            else:
                # List containing a list/array of objects per batch ID
                if isinstance(blob[key][0][0], self.DATAOBJS):
                    # List containing a single list of dataclass objects per batch ID
                    object_type = type(blob[key][0][0])
                    if not object_type in self.object_dtypes:
                        self.object_dtypes[object_type] = self.get_object_dtype(blob[key][0][0])
                    self.key_dict[key]['dtype'] = self.object_dtypes[object_type]
                    self.key_dict[key]['larcv'] = object_type in self.LARCV_DATAOBJS

                elif not hasattr(blob[key][0][0], '__len__'):
                    # List containing a single list of scalars per batch ID
                    self.key_dict[key]['dtype'] = type(blob[key][0][0])

                elif not isinstance(blob[key][0], list) and not blob[key][0].dtype == np.object:
                    # List containing a single ndarray of scalars per batch ID
                    self.key_dict[key]['dtype'] = blob[key][0].dtype
                    self.key_dict[key]['width'] = blob[key][0].shape[1] if len(blob[key][0].shape) == 2 else 0

                elif isinstance(blob[key][0][0], np.ndarray):
                    # List containing a list (or ndarray) of ndarrays per batch ID
                    widths = []
                    for i in range(len(blob[key][0])):
                        widths.append(blob[key][0][i].shape[1] if len(blob[key][0][i].shape) == 2 else 0)
                    same_width = np.all([widths[i] == widths[0] for i in range(len(widths))])

                    self.key_dict[key]['dtype'] = blob[key][0][0].dtype
                    self.key_dict[key]['width'] = widths
                    self.key_dict[key]['merge'] = same_width
                else:
                    raise TypeError('Do not know how to store output of type', type(blob[key][0]))

    def get_object_dtype(self, obj):
        '''
        Loop over the members of a class to figure out what to store. This
        function assumes that the the class only posses getters that return
        either a scalar, a string, a larcv.Vertex, a list, np.ndarrary or a set.

        Parameters
        ----------
        object : class instance
            Instance of an object used to identify attribute types

        Returns
        -------
        list
            List of (key, dtype) pairs
        '''
        object_dtype = []
        members = inspect.getmembers(obj)
        is_larcv = type(obj) in self.LARCV_DATAOBJS
        skip_keys = self.LARCV_SKIP if is_larcv else self.ANA_SKIP
        attr_names = [k for k, _ in members if k[0] != '_' and k not in skip_keys]
        for key in attr_names:
            # Fetch the attribute value
            if is_larcv:
                val = getattr(obj, key)()
            else:
                val = getattr(obj, key)
                if callable(val):
                    continue

            # Append the relevant data type
            if isinstance(val, str):
                # String
                object_dtype.append((key, h5py.string_dtype()))
            elif np.isscalar(val):
                # Scalar
                object_dtype.append((key, type(val)))
            elif isinstance(val, larcv.Vertex):
                # Three-vector
                object_dtype.append((key, h5py.vlen_dtype(np.float32)))
            elif hasattr(val, '__len__'):
                # List/array of values
                if hasattr(val, 'dtype'):
                    # Numpy array
                    object_dtype.append((key, h5py.vlen_dtype(val.dtype)))
                elif len(val) and np.isscalar(val[0]):
                    # List of scalars
                    object_dtype.append((key, h5py.vlen_dtype(type(val[0]))))
                else:
                    # Empty list (typing unknown, cannot store)
                    pass
            else:
                raise ValueError('Unexpected key')

        return object_dtype

    def initialize_datasets(self, file):
        '''
        Create place hodlers for all the datasets to be filled.

        Parameters
        ----------
        file : h5py.File
            HDF5 file instance
        '''
        self.event_dtype = []
        ref_dtype = h5py.special_dtype(ref=h5py.RegionReference)
        for key, val in self.key_dict.items():
            cat = val['category']
            grp = file[cat] if cat in file else file.create_group(cat)
            self.event_dtype.append((key, ref_dtype))
            if not val['merge'] and not isinstance(val['width'], list):
                # If the key contains a list of objects of identical shape
                w = val['width']
                shape, maxshape = [(0, w), (None, w)] if w else [(0,), (None,)]
                grp.create_dataset(key, shape, maxshape=maxshape, dtype=val['dtype'])
                grp[key].attrs['scalar'] = val['scalar']
                grp[key].attrs['larcv']  = val['larcv']

            elif not val['merge']:
                # If the elements of the list are of variable widths, refer to one
                # dataset per element. An index is stored alongside the dataset to break
                # each element downstream.
                n_arrays = len(val['width'])
                subgrp = grp.create_group(key)
                subgrp.create_dataset(f'index', (0, n_arrays), maxshape=(None, n_arrays), dtype=ref_dtype)
                for i, w in enumerate(val['width']):
                    shape, maxshape = [(0, w), (None, w)] if w else [(0,), (None,)]
                    subgrp.create_dataset(f'element_{i}', shape, maxshape=maxshape, dtype=val['dtype'])

            else:
                # If the  elements of the list are of equal width, store them all 
                # to one dataset. An index is stored alongside the dataset to break
                # it into individual elements downstream.
                subgrp = grp.create_group(key)
                w = val['width'][0]
                shape, maxshape = [(0, w), (None, w)] if w else [(0,), (None,)]
                subgrp.create_dataset('elements', shape, maxshape=maxshape, dtype=val['dtype'])
                subgrp.create_dataset('index', (0,), maxshape=(None,), dtype=ref_dtype)

        file.create_dataset('events', (0,), maxshape=(None,), dtype=self.event_dtype)

    def append(self, data_blob=None, result_blob=None, cfg=None):
        '''
        Append the HDF5 file with the content of a batch.

        Parameters
        ----------
        result_blob : dict
            Dictionary containing the output of the reconstruction chain
        data_blob : dict
            Dictionary containing the input data
        cfg : dict
            Dictionary containing the ML chain configuration
        '''
        # If this function has never been called, initialiaze the HDF5 file
        if not self.ready and (not self.append_file or os.path.isfile(self.file_name)):
            self.create(data_blob, result_blob, cfg)
            self.ready = True

        # Append file
        with h5py.File(self.file_name, 'a') as file:
            # Loop over batch IDs
            for batch_id in range(self.batch_size):
                # Initialize a new event
                event = np.empty(1, self.event_dtype)

                # Initialize a dictionary of references to be passed to the event
                # dataset and store the relevant array input and result keys
                ref_dict = {}
                for key in self.input_keys:
                    self.append_key(file, event, data_blob, key, batch_id)
                for key in self.result_keys:
                    self.append_key(file, event, result_blob, key, batch_id)

                # Append event
                event_id  = len(file['events'])
                events_ds = file['events']
                events_ds.resize(event_id + 1, axis=0)
                events_ds[event_id] = event

    def append_key(self, file, event, blob, key, batch_id):
        '''
        Stores array in a specific dataset of an HDF5 file

        Parameters
        ----------
        file : h5py.File
            HDF5 file instance
        event : dict
            Dictionary of objects that make up one event
        blob : dict
            Dictionary containing the information to be stored
        key : string
            Dictionary key name
        batch_id : int
            Batch ID to be stored
        '''
        val = self.key_dict[key]
        cat = val['category']
        if not val['merge'] and not isinstance(val['width'], list):
            # Store single object
            if self.is_scalar(blob[key]):
                obj = blob[key]
            else:
                obj = blob[key][batch_id] if len(blob[key]) == self.batch_size else blob[key][0]
            if not hasattr(obj, '__len__'):
                obj = [obj]

            if val['dtype'] in self.object_dtypes.values():
                self.store_objects(file[cat], event, key, obj, val['dtype'])
            else:
                self.store(file[cat], event, key, obj)

        elif not val['merge']:
            # Store the array and its reference for each element in the list
            self.store_jagged(file[cat], event, key, blob[key][batch_id])

        else:
            # Store one array of for all in the list and a index to break them
            self.store_flat(file[cat], event, key, blob[key][batch_id])

    @staticmethod
    def is_scalar(obj):
        '''
        Returns true if the object has no __len__
        attribute or is a string object.

        Parameters
        ----------
        object : class instance
            Instance of an object used to check typing

        Returns
        -------
        bool
            True if the object is a scalar or a string
        '''
        return not hasattr(obj, '__len__') or isinstance(obj, str)

    @staticmethod
    def store(group, event, key, array):
        '''
        Stores an `ndarray` in the file and stores its mapping
        in the event dataset.

        Parameters
        ----------
        group : h5py.Group
            Dataset group under which to store this array
        event : dict
            Dictionary of objects that make up one event
        key: str
            Name of the dataset in the file
        array : np.ndarray
            Array to be stored
        '''
        # Extend the dataset, store array
        dataset = group[key]
        current_id = len(dataset)
        dataset.resize(current_id + len(array), axis=0)
        dataset[current_id:current_id + len(array)] = array

        # Define region reference, store it at the event level
        region_ref = dataset.regionref[current_id:current_id + len(array)]
        event[key] = region_ref

    @staticmethod
    def store_jagged(group, event, key, array_list):
        '''
        Stores a jagged list of arrays in the file and stores
        an index mapping for each array element in the event dataset.

        Parameters
        ----------
        group : h5py.Group
            Dataset group under which to store this array
        event : dict
            Dictionary of objects that make up one event
        key: str
            Name of the dataset in the file
        array_list : list(np.ndarray)
            List of arrays to be stored
        '''
        # Extend the dataset, store combined array
        region_refs = []
        for i, array in enumerate(array_list):
            dataset = group[key][f'element_{i}']
            current_id = len(dataset)
            dataset.resize(current_id + len(array), axis=0)
            dataset[current_id:current_id + len(array)] = array

            region_ref = dataset.regionref[current_id:current_id + len(array)]
            region_refs.append(region_ref)

        # Define the index which stores a list of region_refs
        index = group[key]['index']
        current_id = len(dataset)
        index.resize(current_id+1, axis=0)
        index[current_id] = region_refs

        # Define a region reference to all the references, store it at the event level
        region_ref = index.regionref[current_id:current_id+1]
        event[key] = region_ref

    @staticmethod
    def store_flat(group, event, key, array_list):
        '''
        Stores a concatenated list of arrays in the file and stores
        its index mapping in the event dataset to break them.

        Parameters
        ----------
        group : h5py.Group
            Dataset group under which to store this array
        event : dict
            Dictionary of objects that make up one event
        key: str
            Name of the dataset in the file
        array_list : list(np.ndarray)
            List of arrays to be stored
        '''
        # Extend the dataset, store combined array
        array = np.concatenate(array_list)
        dataset = group[key]['elements']
        first_id = len(dataset)
        dataset.resize(first_id + len(array), axis=0)
        dataset[first_id:first_id + len(array)] = array

        # Loop over arrays in the list, create a reference for each
        index = group[key]['index']
        current_id = len(index)
        index.resize(first_id + len(array_list), axis=0)
        last_id = first_id
        for i, el in enumerate(array_list):
            first_id = last_id
            last_id += len(el)
            el_ref = dataset.regionref[first_id:last_id]
            index[current_id + i] = el_ref

        # Define a region reference to all the references, store it at the event level
        region_ref = index.regionref[current_id:current_id + len(array_list)]
        event[key] = region_ref

    @staticmethod
    def store_objects(group, event, key, array, obj_dtype):
        '''
        Stores a list of objects with understandable attributes in 
        the file and stores its mapping in the event dataset.

        Parameters
        ----------
        group : h5py.Group
            Dataset group under which to store this array
        event : dict
            Dictionary of objects that make up one event
        key: str
            Name of the dataset in the file
        array : np.ndarray
            Array to be stored
        obj_dtype : list
            List of (key, dtype) pairs which specify what's to store
        '''
        # Convert list of objects to list of storable objects
        objects = np.empty(len(array), obj_dtype)
        for i, o in enumerate(array):
            for k, dtype in obj_dtype:
                attr = getattr(o, k)() if callable(getattr(o, k)) else getattr(o, k)
                if isinstance(attr, (int, float, str)):
                    objects[i][k] = attr
                elif isinstance(attr, larcv.Vertex):
                    vertex = np.array([getattr(attr, a)() for a in ['x', 'y', 'z', 't']], dtype=np.float32)
                    objects[i][k] = vertex
                elif hasattr(attr, '__len__'):
                    vals = attr
                    if not isinstance(attr, np.ndarray):
                        vals = np.array([attr[i] for i in range(len(attr))])
                    objects[i][k] = vals

        # Extend the dataset, store array
        dataset = group[key]
        current_id = len(dataset)
        dataset.resize(current_id + len(array), axis=0)
        dataset[current_id:current_id + len(array)] = objects

        # Define region reference, store it at the event level
        region_ref = dataset.regionref[current_id:current_id + len(array)]
        event[key] = region_ref


class CSVWriter:
    '''
    Class which builds a CSV file to store the output
    of analysis tools. It can only be used to store
    relatively basic quantities (scalars, strings, etc.)

    More documentation to come.
    '''

    def __init__(self,
                 file_name: str = 'output.csv',
                 append_file: bool = False):
        '''
        Initialize the basics of the output file

        Parameters
        ----------
        file_name : str, default 'output.csv'
            Name of the output CSV file
        append_file : bool, default False
            Add more rows to an existing CSV file
        '''
        self.file_name   = file_name
        self.append_file = append_file
        self.result_keys = None
        if self.append_file:
            if not os.path.isfile(file_name):
                msg = "File not found at path: {}. When using append=True "\
                "in CSVWriter, the file must exist at the prescribed path "\
                "before data is written to it.".format(file_name)
                raise FileNotFoundError(msg)
            with open(self.file_name, 'r') as file:
                self.result_keys = file.readline().split(', ')

    def create(self, result_blob: dict):
        '''
        Initialize the header of the CSV file,
        record the keys to be stored.

        Parameters
        ----------
        result_blob : dict
            Dictionary containing the output of the reconstruction chain
        '''
        # Save the list of keys to store
        self.result_keys = list(result_blob.keys())

        # Create a header and write it to file
        with open(self.file_name, 'w') as file:
            header_str = ', '.join(self.result_keys)+'\n'
            file.write(header_str)

    def append(self, result_blob: dict):
        '''
        Append the CSV file with the output

        Parameters
        ----------
        result_blob : dict
            Dictionary containing the output of the reconstruction chain
        '''
        # If this function has never been called, initialiaze the CSV file
        if self.result_keys is None:
            self.create(result_blob)

        # Append file
        with open(self.file_name, 'a') as file:
            result_str = ', '.join([str(result_blob[k]) for k in self.result_keys])+'\n'
            file.write(result_str)
