import numpy as np

from mlreco.post_processing import post_processing


@post_processing('through-muons', ['seg_label', 'clust_data', 'particles_asis'], ['segmentation'])
def through_muons(cfg, module_cfg, data_blob, res, logdir, iteration, **kwargs):
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
    spatial_size = module_cfg.get('spatial_size', 768)
    track_label = module_cfg.get('track_label', 1)
    threshold = module_cfg.get('threshold', 5)
    cords_col = module_cfg.get('coords_col', (1, 4))

    row_names, row_values = [], []
    for p in particles[data_idx][particles_seg[data_idx] == track_label]:
        voxels = input_data[data_idx][p][:, coords_col[0]:coords_col[1]]
        delta_x = voxels[:, 0].max() - voxels[:, 0].min()
        # Is it touching along x axis?

        # Is the delta x consistent with the size of the detector?
        #if np.abs(delta_x - spatial_size) >= threshold:


    return row_names, row_values
