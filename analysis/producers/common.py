import numpy as np
from functools import partial
from collections import defaultdict, OrderedDict

from pprint import pprint

class ScriptProcessor:

    def __init__(self, data, result, debug=True):
        self._funcs = defaultdict(list)
        self._num_batches = len(data['index'])
        self.data = data
        self.index = data['index']
        self.result = result
        self.debug = debug

    def register_function(self, f, priority, script_cfg={}):
        filenames     = f._filenames
        pf            = partial(f, **script_cfg)
        pf._filenames = filenames
        self._funcs[priority].append(pf)
    
    def process(self):
        """
        """
        fname_to_update_list = defaultdict(list)
        sorted_processors = sorted([x for x in self._funcs.items()], reverse=True)
        for priority, f_list in sorted_processors:
            for f in f_list:
                dict_list = f(self.data, self.result)
                filenames = f._filenames
                for i, analysis_dict in enumerate(dict_list):
                    fname_to_update_list[filenames[i]].extend(analysis_dict)
        return fname_to_update_list
