import numpy as np
from mlreco.output_formatters.input import input
from mlreco.output_formatters.uresnet_ppn import uresnet_ppn
from mlreco import output_formatters
from mlreco.utils import utils


def output(output_formatters_list, data_blob, res, cfg, idx, **kwargs):
    """
    Break down the data_blob and res dictionary into events.

    Need to account for: multi-gpu, minibatching, multiple outputs, batches.

    Input
    =====
    output_formatters_list: list of strings refering to the output formatters
        functions that will be applied to each event
    data_blob: from I/O
    res: results dictionary, output of trainval
    cfg: configuration
    idx: iteration index (to number events correctly and avoid overwriting)
    kwargs: other keyword arguments that will be passed to formatter functions
    """
    event_id = idx * cfg['iotool']['batch_size']
    num_forward = len(data_blob['input_data'])
    assert num_forward == cfg['iotool']['batch_size'] / (cfg['training']['minibatch_size'] * len(cfg['training']['gpus']))
    for i in range(num_forward):
        num_gpus = len(data_blob['input_data'][i])
        for j in range(num_gpus):
            batch_idx = np.unique(data_blob['input_data'][i][j][:, -2])
            for b in batch_idx:
                new_data_blob = {}
                data_index = data_blob['input_data'][i][j][:, 3] == b
                for key in data_blob:
                    if isinstance(data_blob[key][i][j], np.ndarray) and len(data_blob[key][i][j].shape) == 2:
                        new_data_blob[key] = data_blob[key][i][j][data_blob[key][i][j][:, 3] == b]
                    elif isinstance(data_blob[key][i][j], list):
                        new_data_blob[key] = data_blob[key][i][j][int(b)]
                # FIXME with minibatch
                new_res = {}
                if 'analysis_keys' in cfg['model']:
                    for key in cfg['model']['analysis_keys']:
                        idx = i * num_forward + j
                        if res[key][idx].shape[0] == data_index.shape[0]:
                            new_res[key] = res[key][idx][data_index]
                        else:  # FIXME assumes batch is in column 3 otherwise
                            new_res[key] = res[key][idx][res[key][idx][:, 3] == b]

                csv_logger = utils.CSVData("%s/output-%.07d.csv" % (cfg['training']['log_dir'], event_id))
                for output in output_formatters_list:
                    f = getattr(output_formatters, output)
                    f(csv_logger, new_data_blob, new_res, **kwargs)
                csv_logger.close()
                event_id += 1
