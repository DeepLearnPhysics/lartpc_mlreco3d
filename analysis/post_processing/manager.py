import numpy as np
from warnings import warn
from collections import defaultdict, OrderedDict

from .factories import post_processor_factory


class PostProcessorManager:
    '''
    Manager in charge of handling post-processing scripts. It loads all
    the post-processor objects once and feeds them data.
    '''
    def __init__(self, cfg, parent_path=''):
        '''
        Initialize the manager

        Parameters
        ----------
        cfg : dict
            Post-processor configurations
        parent_path : str, optional
            Path to the analysis tools configuration file
        '''
        # Loop over the post-processor modules and get their priorities
        keys = np.array(list(cfg.keys()))
        priorities = np.empty(len(cfg))
        for i, k in enumerate(keys):
            priorities[i] = \
                    cfg[k].pop('priority') if 'priority' in cfg[k] else -1

        # Add the modules to a processor list in decreasing order of priority
        self.modules   = OrderedDict()
        self.profilers = {} # TODO: Replace this with StopWatchManager
        keys = keys[np.argsort(-priorities)]
        for k in keys:
            # If requested, profile the module (default True)
            profile = cfg[k].pop('profile') if 'profile' in cfg[k] else True
            if profile:
                self.profilers[k] = 0.

            # Append
            self.modules[k] = post_processor_factory(k, cfg[k], parent_path)

    def run(self, data_dict, result_dict):
        '''
        Main post-processor driver

        Parameters
        ----------
        data_dict : dict
            Input data dictionary
        result_dict : dict
            Chain output dictionary
        '''
        # Reset the profilers
        for key in self.profilers:
            self.profilers[key] = 0.

        # Loop over the post-processor modules
        num_entries = len(data_dict['index'])
        for key, module in self.modules.items():
            # Run the post-processor on each entry
            data_update, result_update = defaultdict(list), defaultdict(list)
            for image_id in range(num_entries):
                data, result, dt = module.run(data_dict, result_dict, image_id)
                if key in self.profilers:
                    self.profilers[key] += dt
                for key, val in data.items():
                    data_update[key].append(val)
                for key, val in result.items():
                    result_update[key].append(val)

            # Update the input dictionaries
            for key, val in data_update.items():
                assert len(val) == num_entries
                data_dict[key] = val

            for key, val in result_update.items():
                assert len(val) == num_entries
                result_dict[key] = val
