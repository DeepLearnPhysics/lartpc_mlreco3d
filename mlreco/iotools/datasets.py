import os, glob, inspect
import numpy as np
from torch.utils.data import Dataset
import mlreco.iotools.parsers

class LArCVDataset(Dataset):
    '''
    A generic interface for LArCV data files.

    This Dataset is designed to produce a batch of arbitrary number
    of data chunks (e.g. input data matrix, segmentation label, point proposal
    target, clustering labels, etc.). Each data chunk is processed by parser
    functions defined in the iotools.parsers module. LArCVDataset object can be
    configured with arbitrary number of parser functions where each function
    can take arbitrary number of LArCV event data objects. The assumption is
    that each data chunk respects the LArCV event boundary.
    '''
    def __init__(self, data_schema, data_keys, limit_num_files=0,
            limit_num_samples=0, event_list=None, skip_event_list=None):
        '''
        Instantiates the LArCVDataset.

        Parameters
        ----------
        data_schema : dict
            A dictionary of (string, dictionary) pairs. The key is a unique
            name of a data chunk in a batch and the associated dictionary
            must include:
              - parser: name of the parser
              - args: (key, value) pairs that correspond to parser argument
                names and their values
            The nested dictionaries can replaced be lists, in which case
            they will be considered as parser argument values, in order.
        data_keys : list
            a list of strings that is required to be present in the file paths
        limit_num_files : int
            an integer limiting number of files to be taken per data directory
        limit_num_samples : int
            an integer limiting number of samples to be taken per data
        event_list : list
            a list of integers to specify which event (ttree index) to process
        skip_event_list : list
            a list of integers to specify which events (ttree index) to skip
        '''

        # Create file list
        self._files = []
        if isinstance(data_keys, str):
            with open(data_keys, 'r') as f:
                data_keys = f.read().splitlines()
        for key in data_keys:
            fs = sorted(glob.glob(key))
            for f in fs:
                self._files.append(f)
                if limit_num_files > 0 and len(self._files) >= limit_num_files: break
            if limit_num_files > 0 and len(self._files) >= limit_num_files: break

        if len(self._files)<1:
            raise FileNotFoundError
        elif len(self._files)>10: print(len(self._files),'files loaded')
        else:
            for f in self._files: print('Loading file:',f)

        # Instantiate parsers
        self._data_keys = []
        self._data_parsers = []
        self._trees = {}
        for key, value in data_schema.items():
            # Check that the schema is a dictionary
            if not isinstance(value, dict):
                raise ValueError('A data schema must be expressed as a dictionary')

            # Identify the parser and its parameter names
            assert 'parser' in value, 'A parser needs to be specified for %s' % key
            if not hasattr(mlreco.iotools.parsers, value['parser']):
                print('The specified parser name %s does not exist!' % value['parser'])
            assert 'args' in value, 'Parser arguments must be provided for %s' % key
            fn = getattr(mlreco.iotools.parsers, value['parser'])
            keys = list(inspect.signature(fn).parameters.keys())
            assert isinstance(value['args'], dict), 'Parser arguments must be a list or dictionary for %s' % key
            for k in value['args'].keys():
                assert k in keys, 'Argument %s does not exist in parser %s' % (k, value['parser'])

            # Append data key and parsers
            self._data_keys.append(key)
            self._data_parsers.append((getattr(mlreco.iotools.parsers, value['parser']), value['args']))
            for arg_name, data_key in value['args'].items():
                if 'event' not in arg_name: continue
                if 'event_list' not in arg_name: data_key = [data_key]
                for k in data_key:
                    if k not in self._trees: self._trees[k] = None

        self._data_keys.append('index')

        # Prepare TTrees and load files
        from ROOT import TChain
        self._entries = None
        for data_key in self._trees.keys():
            # Check data TTree exists, and entries are identical across >1 trees.
            # However do NOT register these TTrees in self._trees yet in order to support >1 workers by DataLoader
            print('Loading tree',data_key)
            chain = TChain(data_key + "_tree")
            for f in self._files:
                chain.AddFile(f)
            if self._entries is not None: assert(self._entries == chain.GetEntries())
            else: self._entries = chain.GetEntries()

        # If event list is provided, register
        if event_list is None:
            self._event_list = np.arange(0, self._entries)
        elif isinstance(event_list, tuple):
            event_list = np.arange(event_list[0], event_list[1])
            self._event_list = event_list
            self._entries = len(self._event_list)
        else:
            if isinstance(event_list,list): event_list = np.array(event_list).astype(np.int32)
            assert(len(event_list.shape)==1)
            where = np.where(event_list >= self._entries)
            removed = event_list[where]
            if len(removed):
                print('WARNING: ignoring some of specified events in event_list as they do not exist in the sample.')
                print(removed)
            self._event_list = event_list[np.where(event_list < self._entries)]
            self._entries = len(self._event_list)

        if skip_event_list is not None:
            self._event_list = self._event_list[~np.isin(self._event_list, skip_event_list)]
            self._entries = len(self._event_list)

        # Set total sample size
        if limit_num_samples > 0 and self._entries > limit_num_samples:
            self._entries = limit_num_samples

        print('Found %d events in file(s)' % len(self._event_list))

        # Flag to identify if Trees are initialized or not
        self._trees_ready=False

    @staticmethod
    def list_data(f):
        from ROOT import TFile
        f=TFile.Open(f,"READ")
        data={'sparse3d':[],'cluster3d':[],'particle':[]}
        for k in f.GetListOfKeys():
            name = k.GetName()
            if not name.endswith('_tree'): continue
            if not len(name.split('_')) < 3: continue
            key = name.split('_')[0]
            if not key in data.keys(): continue
            data[key] = name[:name.rfind('_')]
        return data

    @staticmethod
    def get_event_list(cfg, key):
        event_list = None
        if key in cfg:
            if os.path.isfile(cfg[key]):
                event_list = [int(val) for val in open(cfg[key],'r').read().replace(',',' ').split() if val.isdigit()]
            else:
                try:
                    import ast
                    event_list = ast.literal_eval(cfg[key])
                except SyntaxError:
                    print('iotool.dataset.%s has invalid representation:' % key,event_list)
                    raise ValueError
        return event_list

    @staticmethod
    def create(cfg):
        data_schema = cfg['schema']
        data_keys   = cfg['data_keys']
        lnf         = 0 if not 'limit_num_files' in cfg else int(cfg['limit_num_files'])
        lns         = 0 if not 'limit_num_samples' in cfg else int(cfg['limit_num_samples'])
        event_list  = LArCVDataset.get_event_list(cfg, 'event_list')
        skip_event_list = LArCVDataset.get_event_list(cfg, 'skip_event_list')

        return LArCVDataset(data_schema=data_schema, data_keys=data_keys, limit_num_files=lnf, event_list=event_list, skip_event_list=skip_event_list)

    def data_keys(self):
        return self._data_keys

    def __len__(self):
        return self._entries

    def __getitem__(self,idx):

        # convert to actual index: by default, it is idx, but not if event_list provided
        event_idx = self._event_list[idx]

        # If this is the first data loading, instantiate chains
        if not self._trees_ready:
            from ROOT import TChain
            for key in self._trees.keys():
                chain = TChain(key + '_tree')
                for f in self._files: chain.AddFile(f)
                self._trees[key] = chain
            self._trees_ready=True

        # Move the event pointer
        for tree in self._trees.values():
            tree.GetEntry(event_idx)

        # Create data chunks
        result = {}
        for index, (parser, args) in enumerate(self._data_parsers):
            kwargs = {}
            for k, v in args.items():
                if   'event_list' in k:
                    kwargs[k] = [getattr(self._trees[vi], vi+'_branch') for vi in v]
                elif 'event' in k:
                    kwargs[k] = getattr(self._trees[v], v+'_branch')
                else:
                    kwargs[k] = v
            name = self._data_keys[index]
            result[name] = parser(**kwargs)

        result['index'] = event_idx
        return result
