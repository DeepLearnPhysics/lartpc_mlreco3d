import os
import yaml
import h5py
import glob
import warnings
import numpy as np


class HDF5Reader:
    '''
    Class which reads back information stored in HDF5 files.

    More documentation to come.
    '''
    def __init__(self, file_keys, n_entry=None, n_skip=None, entry_list=None,
            skip_entry_list=None, run_event_list=None,
            skip_run_event_list=None, create_run_map=False, to_larcv=False):
        '''
        Load up the HDF5 file.

        Parameters
        ----------
        file_keys : list
            List of paths to the HDF5 files to be read
        n_entry: int, optional
            Maximum number of entries to load
        n_skip: int, optional
            Number of entries to skip at the beginning
        entry_list : list
            List of integer entry IDs to add to the index
        skip_entry_list : list
            List of integer entry IDs to skip from the index
        run_event_list: list((int, int)), optional
            List of [run, event] pairs to add to the index
        skip_run_event_list: list((int, int)), optional
            List of [run, event] pairs to skip from the index
        create_run_map : bool, default True
            Initialize a map between [run, event] pairs and entries
        to_larcv : bool, default False
            Convert dictionary of LArCV object properties to LArCV objects
        '''
        # Convert the file keys to a list of file paths with glob
        self.file_paths = []
        if isinstance(file_keys, str):
            file_keys = [file_keys]
        for file_key in file_keys:
            file_paths = glob.glob(file_key)
            assert len(file_paths), \
                    f'File key {file_key} yielded no compatible path'
            self.file_paths.extend(file_paths)

        self.file_paths = sorted(self.file_paths)

        # Loop over the input files, build a map from index to file ID
        self.num_entries  = 0
        self.file_index   = []
        self.run_info     = []
        self.split_groups = None
        self.max_print    = 10
        for i, path in enumerate(self.file_paths):
            with h5py.File(path, 'r') as in_file:
                # Check that there are events in the file and the storage mode
                assert 'events' in in_file, \
                        'File does not contain an event tree'

                split_groups = 'data' in in_file and 'result' in in_file
                assert self.split_groups is None \
                        or self.split_groups == split_groups, \
                        'Cannot load files with different storing schemes'
                self.split_groups = split_groups

                # If requested, register the [run, event] information pair
                if create_run_map:
                    source = in_file['data'] if split_groups else in_file
                    assert 'run_info' in source, \
                            'Must provide run info to create run map'
                    run_info = source['run_info']
                    for r, e in zip(run_info['run'], run_info['event']):
                        self.run_info.append([r, e])

                # Update the total number of entries
                num_entries = len(in_file['events'])
                self.num_entries += num_entries
                self.file_index.append(i*np.ones(num_entries, dtype=np.int32))

                # Print
                if i < self.max_print:
                    print('Registered', path)
                elif i == self.max_print:
                    print('...')

        print(f'Registered {len(self.file_paths)} file(s)')

        # Turn file index and run info into np.ndarray, check for duplicates
        self.file_index = np.concatenate(self.file_index)
        if len(self.run_info):
            assert len(self.run_info) == self.num_entries
            self.run_info = np.vstack(self.run_info)
            has_duplicates = not np.all(np.unique(self.run_info,
                axis = 0, return_counts=True)[-1] == 1)
            if has_duplicates:
                warnings.warn('There are duplicated [run, event] pairs',
                        RuntimeWarning)

        # If run_info is set, flip it into a map from info to entry
        self.run_map = None
        if len(self.run_info):
            self.run_map = {tuple(v):i for i, v in enumerate(self.run_info)}

        # Build an entry index to access, modify file index accordingly
        self.entry_index = self.initialize_entry_list(n_entry,
                n_skip, entry_list, skip_entry_list,
                run_event_list, skip_run_event_list)

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
        nested : bool, default `False`
            Whether to nest the output in an array or not (for analysis tools)

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
        with h5py.File(self.file_paths[file_idx], 'r') as in_file:
            event = in_file['events'][entry_idx]
            for key in event.dtype.names:
                self.load_key(in_file, event,
                        data_blob, result_blob, key, nested)

        if self.split_groups:
            return data_blob, result_blob
        else:
            return dict(data_blob, **result_blob)

    def get_run_event(self, run, event, nested=False):
        '''
        Returns an entry corresponding to a specific (run, event) pair

        Parameters
        ----------
        run : int
            Run number
        event : int
            Event number
        nested : bool, default `False`
            Whether to nest the output in an array or not (for analysis tools)

        Returns
        -------
        data_blob : dict
            Ditionary of input data products corresponding to one event
        result_blob : dict
            Ditionary of result data products corresponding to one event
        '''
        return self.get(self.get_run_event_index(run, event), nested)

    def get_run_event_index(self, run, event):
        '''
        Returns an entry index corresponding to a specific (run, event) pair

        Parameters
        ----------
        run : int
            Run number
        event : int
            Event number
        '''
        # Get the appropriate entry index
        assert self.run_map is not None, \
                'Must build a run map to get entries by [run, event]'
        assert (run, event) in self.run_map, \
                f'Could not find (run={run}, event={event}) pair'

        return self.run_map[(run, event)]

    def initialize_entry_list(self, n_entry, n_skip, entry_list,
            skip_entry_list, run_event_list, skip_run_event_list):
        '''
        Create a list of events that can be accessed by `self.get`

        Parameters
        ----------
        n_entry: int, optional
            Maximum number of entries to load
        n_skip: int, optional
            Number of entries to skip at the beginning
        entry_list : list
            List of integer entry IDs to add to the index
        skip_entry_list : list
            List of integer entry IDs to skip from the index
        run_event_list: list((int, int)), optional
            List of [run, event] pairs to add to the index
        skip_run_event_list: list((int, int)), optional
            List of [run, event] pairs to skip from the index

        Returns
        -------
        list
            List of integer entry IDs in the index
        '''
        # Make sure the parameters are sensible
        if np.any([n_entry, n_skip, entry_list, skip_entry_list,
            run_event_list, skip_run_event_list]):
            assert bool(n_entry or n_skip) \
                    ^ bool(entry_list or skip_entry_list) \
                    ^ bool(run_event_list or skip_run_event_list), \
                    'Cannot specify n_entry or n_skip ' \
                    'at the same time as entry_list or skip_entry_list or ' \
                    'at the same time as run_event_list or skip_run_event_list'
        if n_entry is not None or n_skip is not None:
            n_skip = n_skip if n_skip else 0
            n_entry = n_entry if n_entry else self.num_entries - n_skip
            assert n_skip + n_entry <= self.num_entries, \
                'Mismatch between n_entry, n_skip and the ' \
                'available number of entries in the files'
        assert not entry_list or not skip_entry_list,\
                'Cannot specify both entry_list and ' \
                'skip_entry_list at the same time'
        assert not run_event_list or not skip_run_event_list,\
                'Cannot specify both run_event_list and ' \
                'skip_run_event_list at the same time'

        # Create a list of entry indices within each file in the file list
        entry_index = np.empty(self.num_entries, dtype=int)
        for i in np.unique(self.file_index):
            file_mask = np.where(self.file_index==i)[0]
            entry_index[file_mask] = np.arange(len(file_mask))

        # Create a list of entries to be loaded
        if n_entry or n_skip:
            entry_list = np.arange(self.num_entries)
            if n_skip > 0: entry_list = entry_list[n_skip:]
            if n_entry > 0: entry_list = entry_list[:n_entry]
        elif entry_list:
            entry_list = self.parse_entry_list(entry_list)
            assert np.all(np.asarray(entry_list) < self.num_entries),\
                    'Values in entry_list outside of bounds'
        elif run_event_list:
            run_event_list = self.parse_run_event_list(run_event_list)
            entry_list = [self.get_run_event_index(r, e) \
                    for r, e in run_event_list]
        elif skip_entry_list or skip_run_event_list:
            if skip_entry_list:
                skip_entry_list = self.parse_entry_list(skip_entry_list)
                assert np.all(np.asarray(skip_entry_list) < self.num_entries),\
                        'Values in skip_entry_list outside of bounds'
            else:
                skip_run_event_list = \
                        self.parse_run_event_list(skip_run_event_list)
                skip_entry_list = [self.get_run_event_index(r, e) \
                        for r, e in skip_run_event_list]
            entry_mask = np.ones(self.num_entries, dtype=bool)
            entry_mask[skip_entry_list] = False
            entry_list = np.where(entry_mask)[0]

        # Apply entry list to the indexes
        if entry_list is not None:
            entry_index = entry_index[entry_list]
            self.file_index = self.file_index[entry_list]
            self.num_entries = len(entry_list)
            if len(self.run_info):
                self.run_info = self.run_info[entry_list]
                self.run_map = {tuple(v):i for i, v in enumerate(self.run_info)}

        assert len(entry_index), 'Must at least have one entry to load'

        return entry_index


    def load_key(self, in_file, event, data_blob, result_blob, key, nested):
        '''
        Fetch a specific key for a specific event.

        Parameters
        ----------
        in_file : h5py.File
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
        group = in_file
        blob  = result_blob
        if self.split_groups:
            cat   = 'data' if key in in_file['data'] else 'result'
            blob  = data_blob if cat == 'data' else result_blob
            group = in_file[cat]
        if isinstance(group[key], h5py.Dataset):
            if not group[key].dtype.names:
                # If the reference points at a simple dataset, return
                blob[key] = group[key][region_ref]
                if 'scalar' in group[key].attrs and group[key].attrs['scalar']:
                    blob[key] = blob[key][0]
                if len(group[key].shape) > 1:
                    blob[key] = blob[key].reshape(-1, group[key].shape[1])
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
                        for k in blob[key][-1].keys():
                            if isinstance(blob[key][-1][k], bytes):
                                blob[key][-1][k] = blob[key][-1][k].decode()
        else:
            # If the reference points at a group, unpack
            el_refs = group[key]['index'][region_ref].flatten()
            if len(group[key]['index'].shape) == 1:
                ret = np.empty(len(el_refs), dtype=np.object)
                ret[:] = [group[key]['elements'][r] for r in el_refs]
                if len(group[key]['elements'].shape) > 1:
                    for i in range(len(el_refs)):
                        ret[i] = ret[i].reshape(-1, group[key]['elements'].shape[1])
            else:
                ret = [group[key][f'element_{i}'][r] for i, r in enumerate(el_refs)]
                for i in range(len(el_refs)):
                    if len(group[key][f'element_{i}'].shape) > 1:
                        ret[i] = ret[i].reshape(-1, group[key][f'element_{i}'].shape[1])
            blob[key] = ret

        if nested:
            blob[key] = [blob[key]]

    @staticmethod
    def parse_entry_list(list_source):
        '''
        Parses a list into an np.ndarray. The list can be passed as a simple
        python list or a path to a file which contains space or comma separated
        numbers (can be on multiple lines or not)

        Parameters
        ----------
        list_source : Union[list, str]
            List as a python list or a text file path

        Returns
        -------
        np.ndarray
            List as a numpy array
        '''
        if list_source is None:
            return np.empty(0, dtype=np.int64)
        elif not np.isscalar(list_source):
            return np.asarray(list_source, dtype=np.int64)
        elif isinstance(list_source, str):
            assert os.path.isfile(list_source), \
                    'The list source file does not exist'
            lines = open(list_source, 'r').read().splitlines()
            list_source = [w for l in lines for w in l.replace(',', ' ').split()]
            return np.array(list_source, dtype=np.int64)
        else:
            raise ValueError('list format not recognized')

    @staticmethod
    def parse_run_event_list(list_source):
        '''
        Parses a list of [run, event] pairs into an np.ndarray. The list can be
        passed as a simple python list or a path to a file which contains
        one [run, event] pair per line.

        Parameters
        ----------
        list_source : Union[list, str]
            List as a python list or a text file path

        Returns
        -------
        np.ndarray
            List as a numpy array
        '''
        if list_source is None:
            return np.empty((0,2), dtype=np.int64)
        elif not np.isscalar(list_source):
            return np.asarray(list_source, dtype=np.int64).reshape(-1, 2)
        elif isinstance(list_source, str):
            assert os.path.isfile(list_source), \
                    'The list source file does not exist'
            lines = open(list_source, 'r').read().splitlines()
            list_source = [l.replace(',', ' ').split() for l in lines]
            return np.array(list_source, dtype=np.int64)
        else:
            raise ValueError('list format not recognized')

    @staticmethod
    def make_larcv_objects(array, names):
        '''
        Rebuild `larcv` objects from the stored information. Supports
        `larcv.Particle`, `larcv.Neutrino`, `larcv.Flash`, `larcv.CRTHit`
        and `larcv.Trigger`.

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
            elif 'time_ns' in names: obj_class = larcv.Trigger

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
