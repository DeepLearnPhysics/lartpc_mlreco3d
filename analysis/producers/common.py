import numpy as np
from functools import partial
from collections import defaultdict, OrderedDict


class ScriptProcessor:
    """Simple class for handling script functions used to
    generate output csv files for high level analysis. 
    
    Parameters
    ----------
    data : dict
        data dictionary from either model forwarding or HDF5 reading.
    result: dict
        result dictionary containing ML chain outputs
    """
    def __init__(self, data, result):
        self._funcs = defaultdict(list)
        self._num_batches = len(data['index'])
        self.data = data
        self.index = data['index']
        self.result = result

    def register_function(self, f, priority, script_cfg={}):
        filenames     = f._filenames
        pf            = partial(f, **script_cfg)
        pf._filenames = filenames
        self._funcs[priority].append(pf)
    
    def process(self, iteration):
        """
        """
        fname_to_update_list = defaultdict(list)
        sorted_processors = sorted([x for x in self._funcs.items()], reverse=True)
        for priority, f_list in sorted_processors:
            for f in f_list:
                dict_list = f(self.data, self.result, iteration=iteration)
                filenames = f._filenames
                for i, analysis_dict in enumerate(dict_list):
                    fname_to_update_list[filenames[i]].extend(analysis_dict)
        return fname_to_update_list
