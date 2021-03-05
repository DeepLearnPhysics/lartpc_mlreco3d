import numpy as np


@post_processing('through-muons', ['seg_label', 'clust_data', 'particles'], ['segmentation'])
def through_muons(cfg, data_blob, res, logdir, iteration, **kwargs):
    """
    Find through-going muons for detector calibration purpose.

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
    row_names = ()
    row_values = ()
    return row_names, row_values
