import numpy as np
import h5py
import yaml
import inspect
from collections import defaultdict
from larcv import larcv


class HDF5Writer:
    '''
    Class which build an HDF5 file to store the output
    (and optionally input) of the reconstruction chain.

    More documentation to come.
    '''

    def __init__(self, cfg):
        '''
        Initialize the basics of the output file

        Parameters
        ----------
        cfg : dict
            Writer configuration parameter (TODO: turn this into a list of named parameters)
        '''
        # Store attributes
        self.file_name        = cfg.get('file_name', 'output.h5')
        self.input_keys       = cfg.get('input_keys', None)
        self.skip_input_keys  = cfg.get('skip_input_keys', [])
        self.result_keys      = cfg.get('result_keys', None)
        self.skip_result_keys = cfg.get('skip_result_keys', [])
        self.created          = False

    def create(self, cfg, data_blob, result_blob=None):
        '''
        Create the output file structure based on the data and result blobs.

        Parameters
        ----------
        cfg : dict
            Dictionary containing the ML chain configuration
        data_blob : dict
            Dictionary containing the input data
        result_blob : dict
            Dictionary containing the output of the reconstruction chain
        '''
        # Get the expected batch_size from the data_blob (index must be present)
        self.batch_size = len(data_blob['index'])

        # Initialize a dictionary to store keys and their properties (dtype and shape)
        self.key_dict = defaultdict(lambda: {'category': None, 'dtype':None, 'width':0, 'merge':False})

        # If requested, loop over input_keys and add them to what needs to be tracked
        if self.input_keys is None: self.input_keys = data_blob.keys()
        self.input_keys = set(self.input_keys)
        if 'index' not in self.input_keys: self.input_keys.add('index')
        for key in self.skip_input_keys:
            if key in self.input_keys:
                self.input_keys.remove(key)
        for key in self.input_keys:
            self.register_key(data_blob, key, 'data')

        # Loop over the result_keys and add them to what needs to be tracked
        assert self.result_keys is None or result_blob is not None,\
                'No result provided, cannot request keys from it'
        if self.result_keys is None: self.result_keys = result_blob.keys()
        self.result_keys = set(self.result_keys)
        for key in self.skip_result_keys:
            if key in self.result_keys:
                self.result_keys.remove(key)
        for key in self.result_keys:
            self.register_key(result_blob, key, 'result')

        # Initialize the output HDF5 file
        with h5py.File(self.file_name, 'w') as file:
            # Initialize the info dataset that stores top-level description of what is stored
            # TODO: This needs to be fleshed out, currently dumping the config as a single string...
            file.create_dataset('info', (0,), maxshape=(None,), dtype=None)
            file['info'].attrs['cfg'] = yaml.dump(cfg)

            # Initialize the event dataset and the corresponding reference array datasets
            self.initialize_datasets(file)

            # Mark file as ready for use
            self.created = True

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
        if not isinstance(blob[key], list):
            # Single scalar (TODO: Is that thing? If not, why not?)
            self.key_dict[key]['dtype'] = type(blob[key])

        else:
            if len(blob[key]) != self.batch_size:
                # List with a single scalar, regardless of batch_size
                # TODO: understand why single scalars are in arrays...
                assert len(blob[key]) == 1 and\
                        not hasattr(blob[key][0], '__len__'),\
                        'If there is an array of length mismatched with batch_size, '+\
                        'it must contain a single scalar.'
                self.key_dict[key]['dtype'] = type(blob[key][0])

            elif not hasattr(blob[key][0], '__len__'):
                # List containing a single scalar per batch ID
                self.key_dict[key]['dtype'] = type(blob[key][0])

            elif isinstance(blob[key][0], (list, np.ndarray)) and\
                    isinstance(blob[key][0][0], larcv.Particle):
                # List containing a single list of larcv.Particle object per batch ID
                if not hasattr(self, 'particle_dtype'):
                    self.particle_dtype = self.get_object_dtype(blob[key][0][0])
                self.key_dict[key]['dtype'] = self.particle_dtype

            elif isinstance(blob[key][0], (list, np.ndarray)) and\
                    isinstance(blob[key][0][0], larcv.Neutrino):
                # List containing a single list of larcv.Neutrino object per batch ID
                if not hasattr(self, 'neutrino_dtype'):
                    self.neutrino_dtype = self.get_object_dtype(blob[key][0][0])
                self.key_dict[key]['dtype'] = self.neutrino_dtype

            elif isinstance(blob[key][0], (list, np.ndarray)) and\
                    isinstance(blob[key][0][0], larcv.Flash):
                # List containing a single list of larcv.Flash object per batch ID
                if not hasattr(self, 'flash_dtype'):
                    self.flash_dtype = self.get_object_dtype(blob[key][0][0])
                self.key_dict[key]['dtype'] = self.flash_dtype

            elif isinstance(blob[key][0], (list, np.ndarray)) and\
                    isinstance(blob[key][0][0], larcv.CRTHit):
                # List containing a single list of larcv.CRTHit object per batch ID
                if not hasattr(self, 'crthit_dtype'):
                    self.crthit_dtype = self.get_object_dtype(blob[key][0][0])
                self.key_dict[key]['dtype'] = self.crthit_dtype

            elif isinstance(blob[key][0], list) and\
                    not hasattr(blob[key][0][0], '__len__'):
                # List containing a single list of scalars per batch ID
                self.key_dict[key]['dtype'] = type(blob[key][0][0])

            elif isinstance(blob[key][0], np.ndarray) and\
                    not blob[key][0].dtype == np.object:
                # List containing a single ndarray of scalars per batch ID
                self.key_dict[key]['dtype'] = blob[key][0].dtype
                self.key_dict[key]['width'] = blob[key][0].shape[1] if len(blob[key][0].shape) == 2 else 0

            elif isinstance(blob[key][0], (list, np.ndarray)) and isinstance(blob[key][0][0], np.ndarray):
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
        skip_keys = ['add_trajectory_point', 'dump', 'momentum', 'boundingbox_2d', 'boundingbox_3d'] +\
                [k+a for k in ['', 'parent_', 'ancestor_'] for a in ['x', 'y', 'z', 't']]
        attr_names = [k for k, _ in members if '__' not in k and k not in skip_keys]
        for key in attr_names:
            val = getattr(obj, key)()
            if isinstance(val, (int, float)):
                object_dtype.append((key, type(val)))
            elif isinstance(val, str):
                object_dtype.append((key, h5py.string_dtype()))
            elif isinstance(val, larcv.Vertex):
                object_dtype.append((key, h5py.vlen_dtype(np.float32)))
            elif hasattr(val, '__len__') and len(val) and isinstance(val[0], (int, float)):
                object_dtype.append((key, h5py.vlen_dtype(type(val[0]))))
            elif hasattr(val, '__len__'):
                pass # Empty list, no point in storing
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

    def append(self, cfg, data_blob, result_blob):
        '''
        Append the HDF5 file with the content of a batch.

        Parameters
        ----------
        cfg : dict
            Dictionary containing the ML chain configuration
        data_blob : dict
            Dictionary containing the input data
        result_blob : dict
            Dictionary containing the output of the reconstruction chain
        '''
        # If this function has never been called, initialiaze the HDF5 file
        if not self.created:
            self.create(cfg, data_blob, result_blob)

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
            obj = blob[key][batch_id] if len(blob[key]) == self.batch_size else blob[key][0]
            if not hasattr(obj, '__len__'):
                obj = [obj]

            if hasattr(self, 'particle_dtype') and val['dtype'] == self.particle_dtype:
                self.store_objects(file[cat], event, key, obj, self.particle_dtype)
            elif hasattr(self, 'neutrino_dtype') and val['dtype'] == self.neutrino_dtype:
                self.store_objects(file[cat], event, key, obj, self.neutrino_dtype)
            elif hasattr(self, 'flash_dtype') and val['dtype'] == self.flash_dtype:
                self.store_objects(file[cat], event, key, obj, self.flash_dtype)
            elif hasattr(self, 'crthit_dtype') and val['dtype'] == self.crthit_dtype:
                self.store_objects(file[cat], event, key, obj, self.crthit_dtype)
            else:
                self.store(file[cat], event, key, obj)

        elif not val['merge']:
            # Store the array and its reference for each element in the list
            self.store_jagged(file[cat], event, key, blob[key][batch_id])

        else:
            # Store one array of for all in the list and a index to break them
            self.store_flat(file[cat], event, key, blob[key][batch_id])

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
                attr = getattr(o, k)()
                if isinstance(attr, (int, float, str)):
                    objects[i][k] = attr
                elif isinstance(attr, larcv.Vertex):
                    vertex = np.array([getattr(attr, a)() for a in ['x', 'y', 'z', 't']], dtype=np.float32)
                    objects[i][k] = vertex
                elif hasattr(attr, '__len__'):
                    vals = np.array([attr[i] for i in range(len(attr))], dtype=np.int32)
                    objects[i][k] = vals

        # Extend the dataset, store array
        dataset = group[key]
        current_id = len(dataset)
        dataset.resize(current_id + len(array), axis=0)
        dataset[current_id:current_id + len(array)] = objects

        # Define region reference, store it at the event level
        region_ref = dataset.regionref[current_id:current_id + len(array)]
        event[key] = region_ref
