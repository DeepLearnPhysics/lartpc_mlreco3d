import numpy as np

from mlreco.utils import cm_to_pixel

from .boxes import box_traces


def detector_traces(boundary_file, meta=None, to_pixel=False, draw_faces=False, shared_legend=True, legend_prefix='Detector', color='rgba(0,0,0,0.150)',  **kwargs):
    '''
    Function which takes loads a file with detector boundaries and
    produces a list of traces which represent them in a 3D event display.

    The detector boundary file is a `.npy` or `.npz` file which contains
    a single tensor of shape (N, 3, 2), with N the number of detector
    volumes. The first column for each volume represents the lower boundary
    and the second the upper boundary. The boundaries must be ordered.

    The metadata is assumed to have the following structure:
    [lower_x, lower_y(, lower_z), upper_x, upper_y, (upper_z), size_x, size_y(, size_z)],
    i.e. lower and upper bounds of the volume and pixel/voxel size.

    Parameters
    ----------
    boundary_file : str
        Path to the boundary file
    meta : np.ndarray, optional
        (9) Array of metadata information (only needed if pixel_coordinates is True)
    to_pixel : bool, default False
        If True, the coordinates are converted to pixel indices
    draw_faces : bool, default False
        Weather or not to draw the box faces, or only the edges
    shared_legend : bool, default True
        If True, the legend entry in plotly is shared between all the detector volumes
    legend_prefix : Union[str, List[str]], default 'Detector'
        Name(s) of the detector volumes
     color : Union[int, str, np.ndarray]
        Color of boxes or list of color of boxes

    **kwargs : dict, optional
        List of additional arguments to pass to mlreco.viusalization.boxes.box_traces
    '''
    # Load the list of boundaries
    boundaries = np.load(boundary_file)

    # If required, convert to pixel coordinates
    if to_pixel:
        assert meta is not None,\
                'Must provide meta information to convert to pixel coordinates'
        boundaries = cm_to_pixel(boundaries, meta)

    # Get a trace per detector volume
    detectors = box_traces(boundaries[...,0], boundaries[...,1], draw_faces=draw_faces, color=color, **kwargs)

    # Update the trace names
    for i, d in enumerate(detectors):
        d['name'] = legend_prefix if shared_legend else f'{legend_prefix}_{i}'
        if shared_legend:
            d['legendgroup'] = 'group2'
            d['showlegend'] = i == 0

    return detectors
