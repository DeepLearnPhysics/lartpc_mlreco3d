from collections import defaultdict
from functools import wraps
import os
from tabnanny import verbose
import pandas as pd
from pprint import pprint
import torch
import time

from mlreco.main_funcs import cycle
from mlreco.trainval import trainval
from mlreco.iotools.factories import loader_factory
from mlreco.iotools.readers import HDF5Reader
from mlreco.iotools.writers import CSVWriter
from mlreco.main_funcs import run_post_processing

def evaluate(filenames):
    '''
    Inputs
    ------
        - analysis_function: algorithm that runs on a single image given by
        data_blob[data_idx], res
    '''
    def decorate(func):

        @wraps(func)
        def process_dataset(analysis_config, cfg, profile=True):

            # Total number of iterations to process
            max_iteration = analysis_config['analysis']['iteration']

            # Initialize the process which produces the reconstruction output
            if 'reader' not in analysis_config:
                # If there is not reader, initialize the full chain
                io_cfg = cfg['iotool']

                module_config = cfg['model']['modules']
                event_list = cfg['iotool']['dataset'].get('event_list', None)
                if event_list is not None:
                    event_list = eval(event_list)
                    if isinstance(event_list, tuple):
                        assert event_list[0] < event_list[1]
                        event_list = list(range(event_list[0], event_list[1]))

                loader = loader_factory(cfg, event_list=event_list)
                dataset = iter(cycle(loader))
                Trainer = trainval(cfg)
                loaded_iteration = Trainer.initialize()

                if max_iteration == -1:
                    max_iteration = len(loader.dataset)
                assert max_iteration <= len(loader.dataset)

            else:
                # If there is a reader, simply load reconstructed data
                file_keys = analysis_config['reader']['file_keys']
                entry_list = analysis_config['reader'].get('entry_list', [])
                skip_entry_list = analysis_config['reader'].get('skip_entry_list', [])
                Reader = HDF5Reader(file_keys, entry_list, skip_entry_list, True)
                if max_iteration == -1:
                    max_iteration = len(Reader)
                assert max_iteration <= len(Reader)

            # Initialize the writer(s)
            log_dir = analysis_config['logger']['log_dir']
            append  = analysis_config['logger'].get('append', False)

            writers = {}
            for file_name in filenames:
                path = os.path.join(log_dir, file_name+'.csv')
                writers[file_name] = CSVWriter(path, append)

            # Loop over the number of requested iterations
            iteration = 0
            while iteration < max_iteration:

                # 1. Forwarding or Reading HDF5 file
                if profile:
                    start = time.time()
                if 'reader' not in analysis_config:
                    data_blob, res = Trainer.forward(dataset)
                else:
                    data_blob, res = Reader.get(iteration, nested=True)
                if profile:
                    print('------------------------------------------------')
                    print("Forward took %.2f s" % (time.time() - start))
                img_indices = data_blob['index']
                
                # 2. Run post-processing scripts
                stime = time.time()
                if 'post_processing' in analysis_config:
                    run_post_processing(analysis_config, data_blob, res)
                if profile:
                    end = time.time()
                    print("Post-processing took %.2f s" % (time.time() - stime))
                
                # 3. Run analysis tools script
                stime = time.time()
                fname_to_update_list = defaultdict(list)
                for batch_index, img_index in enumerate(img_indices):
                    dict_list = func(data_blob, res, batch_index, analysis_config, cfg)
                    for i, analysis_dict in enumerate(dict_list):
                        fname_to_update_list[filenames[i]].extend(analysis_dict)

                # 4. Store information to csv file.
                for i, fname in enumerate(fname_to_update_list):
                    for row_dict in fname_to_update_list[fname]:
                        writers[fname].append(row_dict)

                if profile:
                    end = time.time()
                    print("Analysis tools and logging took %.2f s" % (time.time() - stime))

                # Increment iteration count
                iteration += 1
                if profile:
                    end = time.time()
                    print("Iteration %d (total %.2f s)" % (iteration, end - start))
                    print('------------------------------------------------')

        process_dataset._filenames = filenames
        return process_dataset
    return decorate
