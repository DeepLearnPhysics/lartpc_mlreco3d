from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os, glob
import numpy as np
from torch.utils.data import Dataset
import mlreco.iotools.parsers

class LArCVDataset(Dataset):
    """
    class: a generic interface for LArCV data files. This Dataset is designed to produce a batch of arbitrary number
           of data chunks (e.g. input data matrix, segmentation label, point proposal target, clustering labels, etc.).
           Each data chunk is processed by parser functions defined in the iotools.parsers module. LArCVDataset object
           can be configured with arbitrary number of parser functions where each function can take arbitrary number of
           LArCV event data objects. The assumption is that each data chunk respects the LArCV event boundary.
    """
    def __init__(self, data_schema, data_keys, limit_num_files=0, limit_num_samples=0, event_list=None):
        """
        Args: data_dirs ..... a list of data directories to find files (up to 10 files read from each dir)
              data_schema ... a dictionary of string <=> list of strings. The key is a unique name of a data chunk in a batch.
                              The list must be length >= 2: the first string names the parser function, and the rest of strings
                              identifies data keys in the input files.
              data_key ..... a string that is required to be present in the filename
              limit_num_files ... an integer limiting number of files to be taken per data directory
              limit_num_samples ... an integer limiting number of samples to be taken per data
              event_list ... a list of integers to specify which event (ttree index) to process
        """

        # Create file list
        #self._files = _list_files(data_dirs,data_key,limit_num_files)
        self._files = []
        for key in data_keys:
            fs = glob.glob(key)
            for f in fs:
                self._files.append(f)
                if len(self._files) >= limit_num_files: break
            if len(self._files) >= limit_num_files: break

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
            if len(value) < 2:
                print('iotools.datasets.schema contains a key %s with list length < 2!' % key)
                raise ValueError
            if not hasattr(mlreco.iotools.parsers,value[0]):
                print('The specified parser name %s does not exist!' % value[0])
            self._data_keys.append(key)
            self._data_parsers.append((getattr(mlreco.iotools.parsers,value[0]),value[1:]))
            for data_key in value[1:]:
                if data_key in self._trees: continue
                self._trees[data_key] = None
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
        else:
            if isinstance(event_list,list): event_list = np.array(event_list).astype(np.int32)
            assert(len(event_list.shape)==1)
            where = np.where(event_list >= self._entries)
            removed = event_list[where]
            if len(removed):
                print('WARNING: ignoring some of specified events in event_list as they do not exist in the sample.')
                print(removed)
            self._event_list=event_list[np.where(event_list < self._entries)]
            self._entries = len(self._event_list)

        # Set total sample size
        if limit_num_samples > 0 and self._entries > limit_num_samples:
            self._entries = limit_num_samples
            
        # Flag to identify if Trees are initialized or not
        self._trees_ready=False

    @staticmethod
    def create(cfg):
        data_schema = cfg['schema']
        data_keys   = cfg['data_keys']
        lnf         = 0 if not 'limit_num_files' in cfg else int(cfg['limit_num_files'])
        lns         = 0 if not 'limit_num_samples' in cfg else int(cfg['limit_num_samples'])
        event_list  = None
        if 'event_list' in cfg:
            if os.path.isfile(cfg['event_list']):
                event_list = [int(val) for val in open(cfg['event_list'],'r').read().replace(',',' ').split() if val.digit()]
            else:
                try:
                    import ast
                    event_list = ast.literal_eval(cfg['event_list'])
                except SyntaxError:
                    print('iotool.dataset.event_list has invalid representation:',event_list)
                    raise ValueError
        return LArCVDataset(data_schema=data_schema, data_keys=data_keys, limit_num_files=lnf, event_list=event_list)

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
        for index, (parser, datatree_keys) in enumerate(self._data_parsers):
            data = [getattr(self._trees[key], key + '_branch') for key in datatree_keys]
            name = self._data_keys[index]
            result[name] = parser(data)

        result['index'] = event_idx
        return result
