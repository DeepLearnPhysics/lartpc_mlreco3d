import numpy as np

from mlreco.post_processing import post_processing


@post_processing('nue-selection', ['seg_label', 'clust_data', 'particles_asis'], ['segmentation'])
def nue_selection(cfg, module_cfg, data_blob, res, logdir, iteration, **kwargs):
    """
    Find electron neutrinos.

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
