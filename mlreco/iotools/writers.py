import os
import yaml
import h5py
import inspect
import numpy as np
from collections import defaultdict
from larcv import larcv
from analysis import classes as analysis

from mlreco.utils.globals import SHAPE_LABELS, PID_LABELS, NU_CURR_TYPE, NU_INT_TYPE


class HDF5Writer:
    '''
    Class which builds an HDF5 file to store the input
    and/or the output of the reconstruction chain. It
    can also be used to append an existing HDF5 file with
    information coming out of the analysis tools.

    More documentation to come.
    '''
    # Analysis object attributes to be stored as enumerated types and their associated rules
    ANA_ENUM = {
        'semantic_type': {v : k for k, v in SHAPE_LABELS.items()},
        'pid': {v : k for k, v in PID_LABELS.items()},
        'nu_current_type': {v : k for k, v in NU_CURR_TYPE.items()},
        'nu_interaction_type': {v : k for k, v in NU_INT_TYPE.items()},
        'nu_interaction_mode': {v : k for k, v in NU_INT_TYPE.items()}
    }

    # Analysis object array attributes which have a fixed length
    ANA_FIXED_LENGTH = [
        'start_point', 'end_point', 'start_dir', 'truth_start_dir', 'end_dir',
        'start_position', 'end_position', 'vertex', 'truth_vertex', 'momentum',
        'truth_momentum', 'pid_scores', 'primary_scores', 'particle_counts',
        'primary_counts'
    ]

    # Object attributes that do not need to be stored to HDF5
    LARCV_SKIP_ATTRS = [
        'add_trajectory_point', 'dump', 'momentum', 'boundingbox_2d', 'boundingbox_3d',
        *[k + a for k in ['', 'parent_', 'ancestor_'] for a in ['x', 'y', 'z', 't']]
    ]

    ANA_SKIP_ATTRS = [
        'points', 'truth_points', 'sed_points', 'particles', 'fragments',
        'sources', 'depositions', 'depositions_MeV', 'truth_depositions',
        'truth_depositions_MeV', 'sed_depositions_MeV', 'particles_summary'
    ]

    SKIP_ATTRS = {
        larcv.Particle: LARCV_SKIP_ATTRS,
        larcv.Neutrino: LARCV_SKIP_ATTRS,
        larcv.Flash:    ['wireCenters', 'wireWidths'],
        larcv.CRTHit:   ['feb_id', 'pesmap'],
        analysis.ParticleFragment:      ANA_SKIP_ATTRS,
        analysis.TruthParticleFragment: ANA_SKIP_ATTRS,
        analysis.Particle:              ANA_SKIP_ATTRS,
        analysis.TruthParticle:         ANA_SKIP_ATTRS,
        analysis.Interaction:           ANA_SKIP_ATTRS + ['index', 'truth_index', 'sed_index'],
        analysis.TruthInteraction:      ANA_SKIP_ATTRS + ['index', 'truth_index', 'sed_index']
    }
    if hasattr(larcv, 'Trigger'): # TMP until a new singularity
        SKIP_ATTRS.update({larcv.Trigger:  ['clear']})

    # Output with default types. TODO: move this, make it not name-dependant
    DEFAULT_OBJS = {
        'particles':          analysis.Particle(),
        'truth_particles':    analysis.TruthParticle(),
        'interactions':       analysis.Interaction(),
        'truth_interactions': analysis.TruthInteraction(),
    }

    # Outputs that have a fixed number of tensors. #TODO: Inherit from unwrap rules
    TENSOR_LISTS = ['encoderTensors', 'decoderTensors', 'ppn_masks', 'ppn_layers', 'ppn_coords']

    # List of recognized objects
    DATA_OBJS  = tuple(list(SKIP_ATTRS.keys()))
    LARCV_OBJS = [larcv.Particle, larcv.Neutrino, larcv.Flash, larcv.CRTHit]
    if hasattr(larcv, 'Trigger'): # TMP until a new singularity
        LARCV_OBJS.append(larcv.Trigger)
    LARCV_OBJS = tuple(LARCV_OBJS)

    def __init__(self,
                 file_name: str = 'output.h5',
                 input_keys: list = None,
                 skip_input_keys: list = [],
                 result_keys: list = None,
                 skip_result_keys: list = [],
                 append_file: bool = False,
                 merge_groups: bool = False):
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
        merge_groups: bool, default False
            Merge `data` and `result` blobs in the root directory of the HDF5 file
        '''
        # Store attributes
        self.file_name        = file_name
        self.input_keys       = input_keys
        self.skip_input_keys  = skip_input_keys
        self.result_keys      = result_keys
        self.skip_result_keys = skip_result_keys
        self.append_file      = append_file
        self.merge_groups     = merge_groups
        self.ready            = False
        self.object_dtypes    = []

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
        assert data_blob or result_blob, \
                'Must provide a non-empty data blob or result blob'

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
        with h5py.File(self.file_name, 'w') as out_file:
            # Initialize the info dataset that stores top-level description of what is stored
            if cfg is not None:
                out_file.create_dataset('info', (0,), maxshape=(None,), dtype=None)
                out_file['info'].attrs['cfg'] = yaml.dump(cfg)

            # Initialize the event dataset and the corresponding reference array datasets
            self.initialize_datasets(out_file)

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
        if np.isscalar(blob[key]):
            # Single scalar
            self.key_dict[key]['dtype']  = h5py.string_dtype() \
                    if isinstance(blob[key], str) else type(blob[key])
            self.key_dict[key]['scalar'] = True

        else:
            if np.isscalar(blob[key][0]):
                # List containing a single scalar per batch ID
                self.key_dict[key]['dtype']  = h5py.string_dtype() \
                        if isinstance(blob[key][0], str) else type(blob[key][0])
                self.key_dict[key]['scalar'] = True

            else:
                # List containing a list/array of objects per batch ID
                lengths = np.array([len(blob[key][i]) for i in range(len(blob[key]))])
                index   = np.where(lengths)[0]
                if len(index):
                    ref_id  = index[0]
                    ref_obj = blob[key][ref_id][0]
                elif key in self.DEFAULT_OBJS.keys():
                    ref_obj = self.DEFAULT_OBJS[key]
                else:
                    msg = f'Cannot infer the dtype of a list of empty lists ({key}) and hence cannot initialize the output HDF5 file'
                    raise AssertionError(msg) # TODO: In this case, fall back on a default dtype specified elsewhere

                if isinstance(ref_obj, dict):
                    # List containing a single list of dictionary objects per batch ID
                    dict_dtype = self.get_dict_dtype(ref_obj)
                    self.object_dtypes.append(dict_dtype)

                    self.key_dict[key]['dtype'] = dict_dtype

                elif isinstance(ref_obj, self.DATA_OBJS):
                    # List containing a single list of dataclass objects per batch ID
                    object_dtype = self.get_object_dtype(ref_obj)
                    self.object_dtypes.append(object_dtype)

                    self.key_dict[key]['dtype'] = object_dtype
                    self.key_dict[key]['larcv'] = type(ref_obj) in self.LARCV_OBJS

                elif not hasattr(ref_obj, '__len__'):
                    # List containing a single list of scalars per batch ID
                    self.key_dict[key]['dtype'] = type(ref_obj)

                elif not isinstance(blob[key][ref_id], list) and not blob[key][ref_id].dtype == np.object:
                    # List containing a single ndarray of scalars per batch ID
                    self.key_dict[key]['dtype'] = blob[key][ref_id].dtype
                    self.key_dict[key]['width'] = blob[key][ref_id].shape[1] if len(blob[key][ref_id].shape) == 2 else 0

                elif isinstance(ref_obj, np.ndarray):
                    # List containing a list (or ndarray) of ndarrays per batch ID
                    widths = []
                    for i in range(len(blob[key][ref_id])):
                        widths.append(blob[key][ref_id][i].shape[1] if len(blob[key][ref_id][i].shape) == 2 else 0)
                    same_width = np.all([widths[i] == widths[0] for i in range(len(widths))])

                    self.key_dict[key]['dtype'] = ref_obj.dtype
                    self.key_dict[key]['width'] = widths
                    self.key_dict[key]['merge'] = same_width and key not in self.TENSOR_LISTS

                else:
                    dtype = type(blob[key][ref_id])
                    raise TypeError(f'Do not know how to store output of type {dtype} in key {key}')

    def get_dict_dtype(self, obj_dict, obj=dict, is_larcv=False):
        '''
        Loop over the keys in a dictonary to figure out what to store. This
        function assumes that the dictionary only maps tovalues that are
        either a scalar, string, larcv.Vertex, list, np.ndarrary or set.

        Parameters
        ----------
        obj_dict : dict
            Instance of an object used to identify attribute types
        obj : type, default dict
            Type of object this dictionary was built from
        is_larcv : bool, default False
            Whether or not this dictionary was built from a LArCV class

        Returns
        -------
        list
            List of (key, dtype) pairs
        '''
        object_dtype = []
        for key, val in obj_dict.items():
            # Append the relevant data type
            if isinstance(val, str):
                # String
                object_dtype.append((key, h5py.string_dtype()))
            elif not is_larcv and key in self.ANA_ENUM:
                # Known enumerator
                object_dtype.append((key, h5py.enum_dtype(self.ANA_ENUM[key], basetype=type(val))))
            elif np.isscalar(val):
                # Scalar
                dtype = type(val) if not isinstance(val, bool) else np.uint8
                object_dtype.append((key, dtype))
            elif isinstance(val, larcv.Vertex):
                # Three-vector
                object_dtype.append((key, np.float32, 4)) # x, y, z, t
            elif hasattr(val, '__len__'):
                # List/array of values
                dtype, shape = None, None
                if hasattr(val, 'dtype'):
                    # Numpy array
                    dtype, shape = val.dtype, val.shape
                elif len(val) and np.isscalar(val[0]):
                    # List of scalars
                    dtype, shape = type(val[0]), len(val)
                else:
                    # Empty list (typing unknown, cannot store)
                    if key == 'children_id':
                        dtype, shape = np.int64, 0
                    else:
                        raise ValueError(f'Attribute {key} of {obj} is an untyped empty list')

                if key in self.ANA_FIXED_LENGTH:
                    object_dtype.append((key, dtype, shape))
                else:
                    object_dtype.append((key, h5py.vlen_dtype(dtype)))
            else:
                raise ValueError(f'Attribute {key} of {obj} has unrecognized type {type(val)}')

        return object_dtype

    def get_object_dtype(self, obj):
        '''
        Loop over the members of a class to figure out what to store. This
        function assumes that the class only posseses getters that return
        either a scalar, string, larcv.Vertex, list, np.ndarrary or set.

        Parameters
        ----------
        object : class
            Instance of an object used to identify attribute types

        Returns
        -------
        list
            List of (key, dtype) pairs
        '''
        obj_dict = {}
        members = inspect.getmembers(obj)
        is_larcv = type(obj) in self.LARCV_OBJS
        skip_keys = self.SKIP_ATTRS[type(obj)]
        attr_names = [k for k, _ in members if k[0] != '_' and k not in skip_keys]
        for key in attr_names:
            # Fetch the attribute value
            if is_larcv:
                val = getattr(obj, key)()
                obj_dict[key] = val
            else:
                val = getattr(obj, key)
                if not callable(val):
                    obj_dict[key] = val

        return self.get_dict_dtype(obj_dict, obj, is_larcv)

    def initialize_datasets(self, out_file):
        '''
        Create place hodlers for all the datasets to be filled.

        Parameters
        ----------
        out_file : h5py.File
            HDF5 file instance
        '''
        self.event_dtype = []
        ref_dtype = h5py.special_dtype(ref=h5py.RegionReference)
        for key, val in self.key_dict.items():
            group = out_file
            if not self.merge_groups:
                cat   = val['category']
                group = out_file[cat] if cat in out_file else out_file.create_group(cat)
            self.event_dtype.append((key, ref_dtype))

            if not val['merge'] and not isinstance(val['width'], list):
                # If the key contains a list of objects of identical shape
                w = val['width']
                shape, maxshape = [(0, w), (None, w)] if w else [(0,), (None,)]
                group.create_dataset(key, shape, maxshape=maxshape, dtype=val['dtype'])
                group[key].attrs['scalar'] = val['scalar']
                group[key].attrs['larcv']  = val['larcv']

            elif not val['merge']:
                # If the elements of the list are of variable widths, refer to one
                # dataset per element. An index is stored alongside the dataset to break
                # each element downstream.
                n_arrays = len(val['width'])
                subgroup = group.create_group(key)
                subgroup.create_dataset(f'index', (0, n_arrays), maxshape=(None, n_arrays), dtype=ref_dtype)
                for i, w in enumerate(val['width']):
                    shape, maxshape = [(0, w), (None, w)] if w else [(0,), (None,)]
                    subgroup.create_dataset(f'element_{i}', shape, maxshape=maxshape, dtype=val['dtype'])

            else:
                # If the  elements of the list are of equal width, store them all
                # to one dataset. An index is stored alongside the dataset to break
                # it into individual elements downstream.
                subgroup = group.create_group(key)
                w = val['width'][0]
                shape, maxshape = [(0, w), (None, w)] if w else [(0,), (None,)]
                subgroup.create_dataset('elements', shape, maxshape=maxshape, dtype=val['dtype'])
                subgroup.create_dataset('index', (0,), maxshape=(None,), dtype=ref_dtype)

        out_file.create_dataset('events', (0,), maxshape=(None,), dtype=self.event_dtype)

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
        self.batch_size = len(data_blob['index'])
        with h5py.File(self.file_name, 'a') as out_file:
            # Loop over batch IDs
            for batch_id in range(self.batch_size):
                # Initialize a new event
                event = np.empty(1, self.event_dtype)

                # Initialize a dictionary of references to be passed to the event
                # dataset and store the relevant array input and result keys
                ref_dict = {}
                for key in self.input_keys:
                    self.append_key(out_file, event, data_blob, key, batch_id)
                for key in self.result_keys:
                    self.append_key(out_file, event, result_blob, key, batch_id)

                # Append event
                event_id  = len(out_file['events'])
                events_ds = out_file['events']
                events_ds.resize(event_id + 1, axis=0)
                events_ds[event_id] = event

    def append_key(self, out_file, event, blob, key, batch_id):
        '''
        Stores array in a specific dataset of an HDF5 file

        Parameters
        ----------
        out_file : h5py.File
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
        val   = self.key_dict[key]
        group = out_file
        if not self.merge_groups:
            cat   = val['category']
            group = out_file[cat]

        if not val['merge'] and not isinstance(val['width'], list):
            # Store single arrays
            if key not in blob:
                # If an output does not exists, give an empty array
                array = []
            elif np.isscalar(blob[key]):
                # If an output is a scalar, nest it
                array = blob[key]
            else:
                # If an output is a nested scalar, get it for every batch ID
                assert len(blob[key]) == self.batch_size or len(blob[key]) == 1
                array = blob[key][batch_id] \
                        if len(blob[key]) != 1 else blob[key][0]

            if not hasattr(array, '__len__'):
                array = [array]

            if val['dtype'] in self.object_dtypes:
                self.store_objects(group, event, key, array, val['dtype'])
            else:
                self.store(group, event, key, array)

        elif not val['merge']:
            # Store the array and its reference for each element in the list
            array_list = blob[key][batch_id] if key in blob else \
                    [[] for _ in range(len(val['width']))]
            self.store_jagged(group, event, key, array_list)

        else:
            # Store one array of for all in the list and a index to break them
            array_list = blob[key][batch_id] if key in blob else []
            self.store_flat(group, event, key, array_list)

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
        array = np.concatenate(array_list) if len(array_list) else []
        dataset = group[key]['elements']
        first_id = len(dataset)
        dataset.resize(first_id + len(array), axis=0)
        dataset[first_id:first_id + len(array)] = array

        # Loop over arrays in the list, create a reference for each
        index = group[key]['index']
        current_id = len(index)
        index.resize(current_id + len(array_list), axis=0)
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
            Array of objects or dictionaries to be stored
        obj_dtype : list
            List of (key, dtype) pairs which specify what's to store
        '''
        # Convert list of objects to list of storable objects
        objects = np.empty(len(array), obj_dtype)
        for i, o in enumerate(array):
            for row in obj_dtype:
                k, dtype, _ = row if len(row) == 3 else [*row, None]
                if type(o) is dict:
                    attr = o[k]
                else:
                    attr = getattr(o, k)() if callable(getattr(o, k)) else getattr(o, k)

                if np.isscalar(attr):
                    objects[i][k] = attr
                elif isinstance(attr, larcv.Vertex):
                    vertex = np.array([getattr(attr, a)() for a in ['x', 'y', 'z', 't']], dtype=np.float32)
                    objects[i][k] = vertex
                elif hasattr(attr, '__len__'):
                    vals = attr
                    if not isinstance(attr, np.ndarray):
                        vals = np.array([attr[i] for i in range(len(attr))])
                    objects[i][k] = vals
                else:
                    raise ValueError(f'Type {type(attr)} of attribute {k} of object {o} does not match an expected dtype')

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
                 append_file: bool = False,
                 accept_missing: bool = True):
        '''
        Initialize the basics of the output file

        Parameters
        ----------
        file_name : str, default 'output.csv'
            Name of the output CSV file
        append_file : bool, default False
            Add more rows to an existing CSV file
        accept_missing : bool, default True
            Tolerate missing keys
        '''
        self.file_name      = file_name
        self.append_file    = append_file
        self.accept_missing = accept_missing
        self.result_keys    = None
        if self.append_file:
            if not os.path.isfile(file_name):
                msg = 'File not found at path: {}. When using append=True '\
                'in CSVWriter, the file must exist at the prescribed path '\
                'before data is written to it.'.format(file_name)
                raise FileNotFoundError(msg)
            with open(self.file_name, 'r') as out_file:
                self.result_keys = out_file.readline().split(',')

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
        with open(self.file_name, 'w') as out_file:
            header_str = ','.join(self.result_keys)+'\n'
            out_file.write(header_str)

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
        else:
            if not (list(result_blob.keys()) == self.result_keys):
                if self.accept_missing:
                    new_result_blob = {k:-1 for k in self.result_keys}
                    for k, v in result_blob.items():
                        if k not in new_result_blob:
                            raise KeyError(f'Key {k} has not been seen in previous iterations')
                        new_result_blob[k] = v
                    result_blob = new_result_blob
                else:
                    diff1 = set(list(result_blob.keys())).difference(set(self.result_keys))
                    diff2 = set(self.result_keys).difference(set(list(result_blob.keys())))
                    msg = "Must provide a dictionary with the expected set of keys: "\
                        "difference = {}, {}".format(str(diff1), str(diff2))
                    raise AssertionError(msg)

        # Append file
        with open(self.file_name, 'a') as out_file:
            result_str = ','.join([str(result_blob[k]) for k in self.result_keys])+'\n'
            out_file.write(result_str)
