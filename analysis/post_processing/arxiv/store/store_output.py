import numpy as np
import scipy
import os
from mlreco.post_processing import post_processing


@post_processing(['save-output'],
                [],
                [])
def store_output(cfg, module_cfg, data_blob, res, logdir, iteration):
    row_names = ()
    row_values = ()
    for output in module_cfg['keys_list']:
        if output in res and isinstance(res[output], float):
            row_names += (output,)
            row_values += (res[output],)
    return row_names, row_values
