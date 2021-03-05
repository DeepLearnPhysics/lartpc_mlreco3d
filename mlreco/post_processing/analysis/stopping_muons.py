from mlreco.utils import CSVData
import os
import numpy as np


@post_processing('stopping-muons', ['seg_label', 'clust_data', 'particles'], ['segmentation'])
def stopping_muons(cfg, data_blob, res, logdir, iteration, **kwargs):
    """
    Find stopping muons for calibration purpose (dE/dx).

    Parameters
    ----------
    data_blob: dict
        The input data dictionary from iotools.
    res: dict
        The output of the network, formatted using `analysis_keys`.
    cfg: dict
        Configuration.
    logdir: string
        Path to folder where CSV logs can be stored.
    iteration: int
        Current iteration number.

    Notes
    -----
    N/A.
    """
    row_names = (,)
    row_values = (,)
    return row_names, row_values
