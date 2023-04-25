import yaml
import h5py
import glob
import numpy as np

class HDF5Reader:
    '''
    Class which reads back information stored in HDF5 files.

    More documentation to come.
    '''
    
    def __init__(self, file_keys, entry_list=[], skip_entry_list=[], to_larcv=False):
        '''
        Load up the HDF5 file.

        Parameters
        ----------
        file_paths : list
            List of paths to the HDF5 files to be read
        entry_list: list(int), optional
            Entry IDs to be accessed. If not specified, expose all entries
        skip_entry_list: list(int), optional
            Entry IDs to be skipped
        to_larcv : bool, default False
            Convert dictionary of LArCV object properties to LArCV objects
        '''
        # Convert the file keys to a list of file paths with glob
        self.file_paths = []
        if isinstance(file_keys, str):
            file_keys = [file_keys]
        for file_key in file_keys:
            file_paths = glob.glob(file_key)
            assert len(file_paths), f'File key {file_key} yielded no compatible path'
            self.file_paths.extend(sorted(file_paths))

        # Loop over the input files, build a map from index to file ID
        self.num_entries  = 0
        self.file_index   = []
        self.split_groups = None
        for i, path in enumerate(self.file_paths):
            with h5py.File(path, 'r') as file:
                # Check that there are events in the file and the storage mode
                assert 'events' in file, 'File does not contain an event tree'
                split_groups = 'data' in file and 'result' in file
                assert self.split_groups is None or self.split_groups == split_groups,\
                        'Cannot load files with different storing schemes'
                self.split_groups = split_groups

                self.num_entries += len(file['events'])
                self.file_index.append(i*np.ones(len(file['events']), dtype=np.int32))

                print('Registered', path)
        self.file_index = np.concatenate(self.file_index)

        # Build an entry index to access, modify file index accordingly
        self.entry_index = self.get_entry_list(entry_list, skip_entry_list)

        # Set whether or not to initialize LArCV objects as such
        self.to_larcv = to_larcv

    def __len__(self):
        '''
        Returns the number of entries in the file

        Returns
        -------
        int
            Number of entries in the file
        '''
        return self.num_entries

    def __getitem__(self, idx):
        '''
        Returns a specific entry in the file

        Parameters
        ----------
        idx : int
            Integer entry ID to access

        Returns
        -------
        data_blob : dict
            Ditionary of input data products corresponding to one event
        result_blob : dict
            Ditionary of result data products corresponding to one event
        '''
        return self.get(idx)

    def get(self, idx, nested=False):
        '''
        Returns a specific entry in the file

        Parameters
        ----------
        idx : int
            Integer entry ID to access
        nested : bool
            If true, nest the output in an array of length 1 (for analysis tools)

        Returns
        -------
        data_blob : dict
            Ditionary of input data products corresponding to one event
        result_blob : dict
            Ditionary of result data products corresponding to one event
        '''
        # Get the appropriate entry index
        assert idx < len(self.entry_index)
        entry_idx = self.entry_index[idx]
        file_idx  = self.file_index[idx]

        # Use the events tree to find out what needs to be loaded
        data_blob, result_blob = {}, {}
        with h5py.File(self.file_paths[file_idx], 'r') as file:
            event = file['events'][entry_idx]
            for key in event.dtype.names:
                self.load_key(file, event, data_blob, result_blob, key, nested)

        if self.split_groups:
            return data_blob, result_blob
        else:
            return dict(data_blob, **result_blob)

    def get_entry_list(self, entry_list, skip_entry_list):
        '''
        Create a list of events that can be accessed by `self.get`

        Parameters
        ----------
        entry_list : list
            List of integer entry IDs to add to the index
        skip_entry_list : list
            List of integer entry IDs to skip from the index

        Returns
        -------
        list
            List of integer entry IDs in the index
        '''
        entry_index = np.empty(self.num_entries, dtype=int)
        for i in np.unique(self.file_index):
            file_mask = np.where(self.file_index==i)[0]
            entry_index[file_mask] = np.arange(len(file_mask))

        if skip_entry_list:
            assert np.all(np.asarray(entry_list) < self.num_entries)
            entry_list = set(entry_list)
            for s in skip_entry_list:
                if s in entry_list:
                    entry_list.pop(s)
            entry_list = list(entry_list)

        if entry_list:
            entry_index = entry_index[entry_list]
            self.file_index = self.file_index[entry_list]

        assert len(entry_index), 'Must at least have one entry to load'
        return entry_index

    def load_key(self, file, event, data_blob, result_blob, key, nested):
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
        nested : bool
            If true, nest the output in an array of length 1 (for analysis tools)
        '''
        # The event-level information is a region reference: fetch it
        region_ref = event[key]
        group = file
        blob  = result_blob
        if self.split_groups:
            cat   = 'data' if key in file['data'] else 'result'
            blob  = data_blob if cat == 'data' else result_blob
            group = file[cat]
        if isinstance(group[key], h5py.Dataset):
            if not group[key].dtype.names:
                # If the reference points at a simple dataset, return
                blob[key] = group[key][region_ref]
                if 'scalar' in group[key].attrs and group[key].attrs['scalar']:
                    blob[key] = blob[key][0]
            else:
                # If the dataset has multiple attributes, it contains an object
                array = group[key][region_ref]
                names = array.dtype.names
                if self.to_larcv and ('larcv' not in group[key].attrs or group[key].attrs['larcv']):
                    blob[key] = self.make_larcv_objects(array, names)
                else:
                    blob[key] = []
                    for i in range(len(array)):
                        blob[key].append(dict(zip(names, array[i])))
        else:
            # If the reference points at a group, unpack
            el_refs = group[key]['index'][region_ref].flatten()
            if len(group[key]['index'].shape) == 1:
                ret = [group[key]['elements'][r] for r in el_refs]
            else:
                ret = [group[key][f'element_{i}'][r] for i, r in enumerate(el_refs)]
            blob[key] = ret

        if nested:
            blob[key] = [blob[key]]

    @staticmethod
    def make_larcv_objects(array, names):
        '''
        Rebuild `larcv` objects from the stored information. Supports
        `larcv.Particle`, `larcv.Neutrino`, `larcv.Flash` and `larcv.CRTHit`

        Parameters
        ----------
        array : list
            List of dictionary of larcv object attributes
        names: 
            List of class attribute names

        Returns
        -------
        list
            List of filled `larcv` objects
        '''
        from larcv import larcv
        if len(array):
            obj_class = larcv.Particle
            if 'bjorken_x' in names: obj_class = larcv.Neutrino
            elif 'TotalPE' in names: obj_class = larcv.Flash
            elif 'tagger'  in names: obj_class = larcv.CRTHit

        ret = []
        for i in range(len(array)):
            # Initialize new larcv.Particle or larcv.Neutrino object
            obj_dict = array[i]
            obj = obj_class()

            # Momentum is particular, deal with it first
            if isinstance(obj, (larcv.Particle, larcv.Neutrino)):
                obj.momentum(*[obj_dict[f'p{k}'] for k in ['x', 'y', 'z']])

            # Trajectory for neutrino is also particular, deal with it
            if isinstance(obj, larcv.Neutrino):
                obj.add_trajectory_point(*[obj_dict[f'traj_{k}'] for k in ['x', 'y', 'z', 't', 'px', 'py', 'pz', 'e']])

            # Now deal with the rest
            for name in names:
                if name in ['px', 'py', 'pz', 'p', 'TotalPE'] or name[:5] == 'traj_':
                    continue # Addressed by other setters
                if 'position' in name or 'step' in name:
                    getattr(obj, name)(*obj_dict[name])
                else:
                    cast = lambda x: x.item() if type(x) != bytes and not isinstance(x, np.ndarray) else x
                    getattr(obj, name)(cast(obj_dict[name]))

            ret.append(obj)

        return ret
