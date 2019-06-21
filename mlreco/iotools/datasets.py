from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from torch.utils.data import Dataset
from . import parsers

def _list_files(data_dirs, data_key=None, limit_num_files=0):
    """
    Args: data_dirs ... a list of data directories to find files (up to 10 files read from each dir)
          data_key ..... a string that is required to be present in the filename
          limit_num_files ... an integer limiting number of files to be taken per data directory
    Return: list of files
    """
    files = []
    # Load files from each directory in data_dirs list
    for d in data_dirs:
        file_list = [ os.path.join(d,f) for f in os.listdir(d) if data_key is None or data_key in f ]
        if limit_num_files: file_list = file_list[0:limit_num_files]
        files += file_list
    return files

class LArCVDataset(Dataset):
    """
    class: a generic interface for LArCV data files. This Dataset is designed to produce a batch of arbitrary number
           of data chunks (e.g. input data matrix, segmentation label, point proposal target, clustering labels, etc.).
           Each data chunk is processed by parser functions defined in the iotools.parsers module. LArCVDataset object
           can be configured with arbitrary number of parser functions where each function can take arbitrary number of
           LArCV event data objects. The assumption is that each data chunk respects the LArCV event boundary.
    """
    def __init__(self, data_schema, data_dirs, data_key=None, limit_num_files=0):
        """
        Args: data_dirs ..... a list of data directories to find files (up to 10 files read from each dir)
              data_schema ... a dictionary of string <=> list of strings. The key is a unique name of a data chunk in a batch.
                              The list must be length >= 2: the first string names the parser function, and the rest of strings
                              identifies data keys in the input files.
              data_key ..... a string that is required to be present in the filename
              limit_num_files ... an integer limiting number of files to be taken per data directory
        """

        # Create file list
        self._files = _list_files(data_dirs,data_key,limit_num_files)
        if len(self._files)>10: print(len(self._files),'files loaded')
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
            if not hasattr(parsers,value[0]):
                print('The specified parser name %s does not exist!' % value[0])
            self._data_keys.append(key)
            self._data_parsers.append((getattr(parsers,value[0]),value[1:]))
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
        # Flag to identify if Trees are initialized or not
        self._trees_ready=False

    @staticmethod
    def create(cfg):
        data_dirs   = cfg['data_dirs']
        data_schema = cfg['schema']
        data_key = None if not 'data_key' in cfg         else str(cfg['data_key'])
        lns     = 0    if not 'limit_num_files' in cfg else int(cfg['limit_num_files'])
        return LArCVDataset(data_dirs=data_dirs, data_schema=data_schema, data_key=data_key, limit_num_files=lns)

    def data_keys(self):
        return self._data_keys

    def __len__(self):
        return self._entries

    def __getitem__(self,idx):
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
            tree.GetEntry(idx)
        # Create data chunks
        result = []
        for parser, data_keys in self._data_parsers:
            data = [getattr(self._trees[key], key + '_branch') for key in data_keys]
            result.append(parser(data))

        result.append([idx])
        return tuple(result)
