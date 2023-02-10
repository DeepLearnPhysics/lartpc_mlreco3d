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

from mlreco.utils.utils import ChunkCSVData


def evaluate(filenames, mode='per_image'):
    '''
    Inputs
    ------
        - analysis_function: algorithm that runs on a single image given by
        data_blob[data_idx], res
    '''
    def decorate(func):

        @wraps(func)
        def process_dataset(cfg, analysis_config, profile=True):

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
            max_iteration = analysis_config['analysis']['iteration']
            if max_iteration == -1:
                max_iteration = len(loader.dataset)

            iteration = 0

            log_dir = analysis_config['analysis']['log_dir']
            append = analysis_config['analysis'].get('append', True)
            chunksize = analysis_config['analysis'].get('chunksize', 100)

            output_logs = []
            header_recorded = []

            for fname in filenames:
                fout = os.path.join(log_dir, fname + '.csv')
                output_logs.append(ChunkCSVData(fout, append=append, chunksize=chunksize))
                header_recorded.append(False)

            while iteration < max_iteration:
                if profile:
                    start = time.time()
                data_blob, res = Trainer.forward(dataset)
                if profile:
                    print("Forward took %d s" % (time.time() - start))
                img_indices = data_blob['index']
                fname_to_update_list = defaultdict(list)
                if mode == 'per_batch':
                    # list of (list of dicts)
                    dict_list = func(data_blob, res, None, analysis_config, cfg)
                    for i, analysis_dict in enumerate(dict_list):
                        fname_to_update_list[filenames[i]].extend(analysis_dict)
                elif mode == 'per_image':
                    for batch_index, img_index in enumerate(img_indices):
                        dict_list = func(data_blob, res, batch_index, analysis_config, cfg)
                        for i, analysis_dict in enumerate(dict_list):
                            fname_to_update_list[filenames[i]].extend(analysis_dict)
                else:
                    raise Exception("Evaluation mode {} is invalid!".format(mode))
                for i, fname in enumerate(fname_to_update_list):
                    df = pd.DataFrame(fname_to_update_list[fname])
                    if len(df):
                        output_logs[i].record(df)
                        header_recorded[i] = True
                    # disable pandas from appending additional header lines
                    if header_recorded[i]: output_logs[i].header = False
                iteration += 1
                if profile:
                    end = time.time()
                    print("Iteration %d (total %d s)" % (iteration, end - start))
                torch.cuda.empty_cache()

        process_dataset._filenames = filenames
        process_dataset._mode = mode
        return process_dataset
    return decorate
