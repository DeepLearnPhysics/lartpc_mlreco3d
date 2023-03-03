import numpy as np
import h5py
import yaml

class HDF5Reader:
    '''
    Class which reads back information stored in HDF5 files.

    More documentation to come.
    '''
    
    def __init__(self, file_path, entry_list=[], skip_entry_list=[]):
        '''
        Load up the HDF5 file.

        Parameters
        ----------
        file_path : str
            Path to the HDF5 file to be read
        entry_list: list(int)
            Entry IDs to be accessed
        skip_entry_list: list(int)
            Entry IDs to be skipped
        '''
        # Store attributes
        self.file_path = file_path
        with h5py.File(self.file_path, 'r') as file:
            assert 'events' in file, 'File does not contain an event tree'
            self.n_entries = len(file['events'])

        self.entry_list = self.get_entry_list(entry_list, skip_entry_list)

    def __len__(self):
        '''
        Returns the number of entries in the file

        Returns
        -------
        int
            Number of entries in the file
        '''
        return self.n_entries

    def __getitem__(self, idx):
        '''
        Returns a specific entry in the file

        Returns
        -------
        data_blob : dict
            Ditionary of input data products corresponding to one event
        result_blob : dict
            Ditionary of result data products corresponding to one event
        '''
        # Get the appropriate entry index
        entry_idx = self.entry_list[idx]

        # Use the events tree to find out what needs to be loaded
        data_blob, result_blob = {}, {}
        with h5py.File(self.file_path, 'r') as file:
            event = file['events'][entry_idx]
            for key in event.dtype.names:
                self.load_key(file, event, data_blob, result_blob, key)

        return data_blob, result_blob

    def get_entry_list(self, entry_list, skip_entry_list):
        '''
        Create a list of events that can be accessed by `__getitem__`
        '''
        if not entry_list:
            entry_list = np.arange(self.n_entries, dtype=int)
        if skip_entry_list:
            entry_list = set(entry_list)
            for s in skip_entry_list:
                entry_list.pop(s)
            entry_list = list(entry_list)
        
        assert len(entry_list), 'Must at least have one entry to load'
        return entry_list

    def load_key(self, file, event, data_blob, result_blob, key):
        '''
        Fetch a specific key for a specific event.

        Parameters
        ----------
        file : h5py.File
            HDF5 file instance
        event : dict
            Dictionary of objects that make up one event
        data_blob : dict
            Dictionary used to store the loaded input data
        result_blob : dict
            Dictionary used to store the loaded result data
        key: str
            Name of the dataset in the event
        '''
        # The event-level information is a region reference: fetch it
        region_ref = event[key]
        cat = 'data' if key in file['data'] else 'result'
        blob = data_blob if cat == 'data' else result_blob
        group = file[cat]
        if isinstance(group[key], h5py.Dataset):
            # If the reference points at a dataset, return
            if not group[key].dtype.names:
                blob[key] = group[key][region_ref]
            else:
                names = group[key].dtype.names
                blob[key] = []
                for i in range(len(group[key][region_ref])):
                    blob[key].append(dict(zip(names, group[key][region_ref][i])))
        else:
            # If the reference points at a group, unpack
            el_refs = group[key]['index'][region_ref].flatten()
            if len(group[key]['index'].shape) == 1:
                ret = [group[key]['elements'][r] for r in el_refs]
            else:
                ret = [group[key][f'element_{i}'][r] for i, r in enumerate(el_refs)]
            blob[key] = ret
