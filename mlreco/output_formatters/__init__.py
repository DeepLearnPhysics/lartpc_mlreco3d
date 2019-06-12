import numpy as np
from mlreco.output_formatters.input import input
from mlreco.output_formatters.uresnet_ppn import uresnet_ppn
from mlreco import output_formatters
from mlreco.utils import utils


def output(output_formatters_list, data_blob, res, cfg, idx):
    event_id = 0
    for i in range(len(data_blob['input_data'])):
        for j in range(len(data_blob['input_data'][i])):
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
                        if res[key][j].shape[0] == data_index.shape[0]:
                            new_res[key] = res[key][j][data_index]
                        else:  # assumes batch is in column 3
                            new_res[key] = res[key][j][res[key][j][:, 3] == b]

                csv_logger = utils.CSVData("%s/output-%.07d.csv" % (cfg['training']['log_dir'], event_id))
                for output in output_formatters_list:
                    f = getattr(output_formatters, output)
                    f(csv_logger, new_data_blob, new_res)
                csv_logger.close()
                event_id += 1
