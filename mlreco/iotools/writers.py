import numpy as np
import h5py
import yaml
from collections import defaultdict
from pathlib import Path


class HDF5Writer:
    '''
    Class which build an HDF5 file to store events which contain:
     - Voxel tensors with their values
     - Feature tensors
     - Particle-level objects
     - ...

    More documentation to come.
    '''

    def __init__(self, cfg):
        '''
        Initialize the basics of the output file

        Parameters
        ---------
        cfg : dict
            Writer configuration parameter (TODO: turn this into a list of named parameters)
        '''
        # Store attributes
        self.file_name        = cfg.get('file_name', 'output.h5')
        self.store_input      = cfg.get('store_input', False)
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
        self.key_dict = defaultdict(lambda: {'dtype':None, 'width':0, 'ref':False, 'index':False})

        # If requested, loop over input_keys and add them to what needs to be tracked
        if self.store_input:
            if self.input_keys is None: self.input_keys = data_blob.keys()
            self.input_keys = set(self.input_keys)
            for key in self.input_keys:
                if key not in self.skip_input_keys:
                    self.register_key(data_blob, key)
                else:
                    self.input_keys.pop(key)
        else:
            self.input_keys = {}

        # Loop over the result_keys and add them to what needs to be tracked
        assert self.result_keys is None or result_blob is not None,\
                'No result provided, cannot request keys from it'
        if self.result_keys is None: self.result_keys = result_blob.keys()
        self.result_keys = set(self.result_keys)
        for key in self.result_keys:
            if key not in self.skip_result_keys:
                self.register_key(result_blob, key)
            else:
                self.result_keys.pop(key)

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

    def register_key(self, blob, key):
        '''
        Identify the dtype and shape objects to be dealt with.

        Parameters
        ----------
        blob : dict
            Dictionary containing the information to be stored
        key : string
            Dictionary key name
        '''
        # If the data under the key is a scalar, store it as is
        if not isinstance(blob[key], list):
            # Single scalar (TODO: Is that thing? If not, why not?)
            self.key_dict[key]['dtype'] = type(blob[key])
        else:
            if len(blob[key]) != self.batch_size:
                # List with a single scalar, regardless of batch_size
                # TODO: understand why single scalars are in arrays...
                assert len(blob[key]) == 1 and\
                        not hasattr(blob[key][0], '__len__'),\
                        'If there is an array of mismatched length, it must contain a single scalar'
                self.key_dict[key]['dtype'] = type(blob[key][0])
            elif not hasattr(blob[key][0], '__len__'):
                # List containing a single scalar per batch ID
                self.key_dict[key]['dtype'] = type(blob[key][0])
            elif isinstance(blob[key][0], np.ndarray) and\
                    not blob[key][0].dtype == np.object:
                # List containing a single ndarray of scalars per batch ID
                self.key_dict[key]['dtype'] = blob[key][0].dtype
                self.key_dict[key]['width'] = blob[key][0].shape[1] if len(blob[key][0].shape) == 2 else 0
                self.key_dict[key]['ref']   = True
            elif isinstance(blob[key][0], (list, np.ndarray)) and isinstance(blob[key][0][0], np.ndarray):
                # List containing a list (or ndarray) of ndarrays per batch ID
                widths = []
                for i in range(len(blob[key][0])):
                    widths.append(blob[key][0][i].shape[1] if len(blob[key][0][i].shape) == 2 else 0)
                same_width = np.all([widths[i] == widths[0] for i in range(len(widths))])

                self.key_dict[key]['dtype'] = blob[key][0][0].dtype
                self.key_dict[key]['width'] = widths
                self.key_dict[key]['ref']   = True
                self.key_dict[key]['index'] = same_width
            else:
                raise TypeError('Do not know how to store output of type', type(blob[key][0]))

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
            if not val['ref'] and not val['index']:
                # If the key has <= 1 scalar per batch ID: store in the event dataset
                self.event_dtype.append((key, val['dtype'].__name__))
            elif val['ref'] and not isinstance(val['width'], list):
                # If the key contains one ndarray: store as its own dataset + store a reference in the event dataset 
                w = val['width']
                shape, maxshape = [(0, w), (None, w)] if w else [(0,), (None,)]
                file.create_dataset(key, shape, maxshape=maxshape, dtype=val['dtype'])
                self.event_dtype.append((f'{key}_ref_', ref_dtype))
            elif val['ref'] and not val['index']:
                # If the elements of the list are of variable widths, refer to one dataset per element
                for i, w in enumerate(val['width']):
                    name = f'{key}_el{i:d}'
                    shape, maxshape = [(0, w), (None, w)] if w else [(0,), (None,)]
                    file.create_dataset(name, shape, maxshape, dtype=val['dtype'])
                    self.event_dtype.append((f'{name}_ref_', ref_dtype))
            elif val['index']:
                # If the  elements of the list are of equal width, store them all 
                # to one dataset. An index is stored alongside the dataset to break
                # it into individual elements downstream.
                w = val['width'][0]
                shape, maxshape = [(0, w), (None, w)] if w else [(0,), (None,)]
                file.create_dataset(key, shape, maxshape=maxshape, dtype=val['dtype'])
                file.create_dataset(f'{key}_index_', (0,), maxshape=(None,), dtype=ref_dtype)
                self.event_dtype.append((f'{key}_index_ref_', ref_dtype))

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
        if not val['ref'] and not val['index']:
            # Store the scalar
            event[key] = blob[key][batch_id] if len(blob[key]) == self.batch_size else blob[key][0] # Does not handle scalar case. TODO: Useful?
        elif val['ref'] and not isinstance(val['width'], list):
            # Store the array and its reference
            self.store(file, event, key, blob[key][batch_id])
        elif val['ref'] and not val['index']:
            # Store the array and its reference for each element in the list
            for i in range(len(val['width'])):
                self.store(file, event, f'{key}_el{i:d}', blob[key][batch_id][i])
        elif val['index']:
            # Store one array of for all in the list and a index to break them
            self.store_indexed(file, event, key, blob[key][batch_id])

    @staticmethod
    def store(file, event, key, array):
        '''
        Stores an `ndarray`in the file and stores its mapping
        in the event dataset.

        Parameters
        ----------
        file : h5py.File
            HDF5 file instance
        event : dict
            Dictionary of objects that make up one event
        key: str
            Name of the dataset in the file
        array : np.ndarray
            Array to be stored
        '''
        # Extend the dataset, store array
        dataset = file[key]
        current_id = len(dataset)
        dataset.resize(current_id + len(array), axis=0)
        dataset[current_id:current_id + len(array)] = array

        # Define region reference, store it at the event level
        region_ref = dataset.regionref[current_id:current_id + len(array)]
        event[f'{key}_ref_'] = region_ref

    @staticmethod
    def store_indexed(file, event, key, array_list):
        '''
        Stores a list of arrays in the file and stores
        its index mapping in the event dataset.

        Parameters
        ----------
        file : h5py.File
            HDF5 file instance
        event : dict
            Dictionary of objects that make up one event
        key: str
            Name of the dataset in the file
        array_list : list(np.ndarray)
            List of arrays to be stored
        '''
        # Extend the dataset, store combined array
        array = np.concatenate(array_list)
        dataset = file[key]
        first_id = len(dataset)
        dataset.resize(first_id + len(array), axis=0)
        dataset[first_id:first_id + len(array)] = array

        # Loop over arrays in the list, create a reference for each
        index = file[f'{key}_index_']
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
        event[f'{key}_index_ref_'] = region_ref
