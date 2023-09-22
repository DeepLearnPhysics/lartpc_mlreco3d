import numpy as np
from copy import deepcopy

import plotly
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from mlreco.utils.geometry import Geometry


PLOTLY_COLORS = plotly.colors.qualitative.Plotly
PLOTLY_COLORS_WGRAY = ['#808080'] + PLOTLY_COLORS
HIGH_CONTRAST_COLORS = np.concatenate([plotly.colors.qualitative.Dark24, plotly.colors.qualitative.Light24])


def plotly_layout3d(ranges=None, meta=None, detector=None, titles=None,
        detector_coords=False, backgroundcolor='white', gridcolor='lightgray',
        width=900, height=900, showlegend=True, camera=None,
        aspectmode='manual', aspectratio=None, dark=False,
        margin=dict(r=0, l=0, b=0, t=0), **kwargs):
    """
    Produces plotly.graph_objs.Layout object for a certain format.

    Parameters
    ----------
    ranges : np.ndarray, optional
        (3, 2) or (N, 3) Array used to specify the plot region in (x,y,z)
        directions. If not specified (None), the range will be set to include
        all points. Alternatively can be an array of shape (3,2) specifying
        (x,y,z) axis (min,max) range for a display, or simply a list of points
        with shape (N,3+) where [:,0],[:,1],[:,2] correspond to (x,y,z) values
        and the plotting region is decided by measuring the min,max range in
        each coordinates. This last option is useful if one wants to define
        the region based on a set of points that is not same as what's plotted.
    meta : np.ndarray, optional
        (9) Metadata information used to infer the full image range
    detector : str
        Name of a recognized detector to get the geometry from or path to a
        `.npy` boundary file to load the boundaries from.
    titles : List[str], optional
        (3) Array of strings for (x,y,z) axis title respectively
    detector_coords : bool, default False
        Whether or not the coordinates being drawn are in detector_coordinates
        or pixel IDs
    backgroundcolor : Union[str, int], default 'white'
        Color of the layout background
    gridcolor : Union[str, int], default 'lightgray'
        Color of the grid
    width : int, default 900
        Width of the layout in pixels
    height : int, default 900
        Height of the layout in pixels
    showlegend : bool, default True
        Whether or not to show the image legend
    aspectmode : str, default manual
        Plotly aspect mode. If manual, will define it based on the ranges
    aspectratio : dict, optional
        Plotly dictionary which specifies the aspect ratio for x, y an d z
    dark : bool, default False
        Dark layout
    margin : dict, default dict(r=0, l=0, b=0, t=0)
        Specifies the margin in each subplot
    **kwargs : dict, optional
        List of additional arguments to pass to plotly.graph_objs.Layout

    Results
    -------
    plotly.graph_objs.Layout
        Object that can be given to plotly.graph_objs.Figure for visualization (together with traces)
    """
    # Figure out the drawing range
    if ranges is None:
        ranges = [None, None, None]
    else:
        # If the range is provided, just use it
        if ranges.shape != (3, 2):
            assert len(ranges.shape) == 2 and ranges.shape[1] == 3, \
                'If ranges is not of shape (3, 2), it must be of shape (N, 3)'
            ranges = np.vstack([np.min(ranges, axis=0), np.max(ranges, axis=0)]).T

        # Check that the range is sensible
        assert np.all(ranges[:,1] >= ranges[:,0])

    if meta is not None:
        # If meta information is provided, make the full image the range
        assert ranges is None or None in ranges, \
                'Should not specify both `ranges` and `meta` parameters'
        assert len(np.asarray(meta).reshape(-1)) == 9,\
                'Metadata should be an array of 9 values'
        lowers, uppers, sizes = np.split(np.asarray(meta).reshape(-1), 3)
        if detector_coords:
            ranges = np.vstack([lowers, uppers]).T
        else:
            ranges = np.vstack([[0, 0, 0], np.round((uppers-lowers)/sizes)]).T

    if detector is not None:
        # If detector geometry is provided, make the full detector the range
        assert (ranges is None or None in ranges) and meta is None, \
                'Should not specify `detector` along with `ranges` or `meta`'
        geo = Geometry(detector)
        lengths = geo.detector[:,1] - geo.detector[:,0]
        ranges = geo.detector

        # Add some padding
        ranges[:,0] -= lengths*0.1
        ranges[:,1] += lengths*0.1

        # Define detector-style camera, unless explicitely provided
        if camera is None:
            camera = dict(eye    = dict(x = -2, y = 1,    z = -0.01),
                          up     = dict(x = 0., y = 1.,   z = 0.),
                          center = dict(x = 0., y = -0.1, z = -0.01))


    # Infer the image width/height and aspect ratios, unless they are specified
    if aspectmode == 'manual':
        if aspectratio is None:
            axes = ['x', 'y', 'z']
            ratios = np.ones(len(ranges)) if ranges[0] is None else (ranges[:,1]-ranges[:,0])/np.max(ranges[:,1]-ranges[:,0])
            aspectratio = {axes[i]: v for i, v in enumerate(ratios)}

    # Check on the axis titles, define default
    assert titles is None or len(titles) == 3, 'Must specify one title per axis'
    if titles is None:
        unit = 'cm' if detector_coords else 'pixel'
        titles = [f'x [{unit}]', f'y [{unit}]', f'z [{unit}]']

    # Initialize some default legend behavior
    if 'legend' not in kwargs:
        kwargs['legend'] = dict(title = 'Legend', tracegroupgap = 1, itemsizing = 'constant')

    # If a dark layout is requested, set the theme and the background color accordingly
    if dark:
        kwargs['template'] = 'plotly_dark'
        kwargs['paper_bgcolor'] = 'black'
        backgroundcolor = 'black'

    # Initialize the general scene layout
    scene = dict(
            xaxis = dict(nticks = 10, range = ranges[0], showticklabels = True,
                         title = dict(text=titles[0], font=dict(size=20)),
                         tickfont = dict(size=14),
                         backgroundcolor = backgroundcolor,
                         gridcolor = gridcolor,
                         showbackground = True
                        ),
            yaxis = dict(nticks = 10, range = ranges[1], showticklabels = True,
                         title = dict(text=titles[1], font=dict(size=20)),
                         tickfont = dict(size=14),
                         backgroundcolor = backgroundcolor,
                         gridcolor = gridcolor,
                         showbackground = True
                        ),
            zaxis = dict(nticks = 10, range = ranges[2], showticklabels = True,
                         title = dict(text=titles[2], font=dict(size=20)),
                         tickfont = dict(size=14),
                         backgroundcolor = backgroundcolor,
                         gridcolor = gridcolor,
                         showbackground = True
                        ),
            aspectmode = aspectmode,
            aspectratio = aspectratio,
            camera = camera
            )

    # Initialize layout
    layout = go.Layout(
        showlegend = showlegend,
        width = width,
        height = height,
        margin = margin,
        scene1 = scene,
        scene2 = deepcopy(scene),
        scene3 = deepcopy(scene),
        **kwargs
    )

    return layout


def dualplot(traces_left, traces_right, layout=None, titles=[None, None], width=1000, height=500, synchronize=False, margin=dict(r=20, l=20, b=20, t=20), **kwargs):
    '''
    Function which returns a plotly.graph_objs.Figure with two set of traces
    side-by-side in separate subplots.

    Parameters
    ----------
    traces_left : List[object]
        List of plotly traces to draw in the left subplot
    traces_right : List[object]
        List of plotly traces to draw in the right subplot
    layout : plotly.Layout, optional
        Plotly layout
    titles : List[str], optional
        Titles of the two subplots
    width : int, default 1000
        Width of the layout in pixels
    height : int, default 500
        Height of the layout in pixels
    synchronize : bool, default False
        If True, matches the camera position/angle of one plot to the other
    margin : dict, default dict(r=20, l=20, b=20, t=20)
        Specifies the margin in each subplot
    **kwargs : dict, optional
        List of additional arguments to pass to plotly.graph_objs.Layout

    Returns
    -------
    plotly.graph_objs.Figure
        Plotly figure with the two subplots
    '''
    # Make subplot and add traces
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=(titles[0], titles[1]),
                        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
                        horizontal_spacing=0.05, vertical_spacing=0.04)
    fig.add_traces(traces_left, rows=[1]*len(traces_left), cols=[1]*len(traces_left))
    fig.add_traces(traces_right, rows=[1]*len(traces_right), cols=[2]*len(traces_right))

    # Inialize and set layout
    if layout is None:
        layout = plotly_layout3d(width=width, height=height, margin=margin, **kwargs)
    fig.layout.update(layout)

    # If requested, synchronize the two cameras
    if synchronize:
        fig = go.FigureWidget(fig)

        def cam_change_left(layout, camera):
            fig.layout.scene2.camera = camera
        def cam_change_right(layout, camera):
            fig.layout.scene1.camera = camera

        fig.layout.scene1.on_change(cam_change_left,  'camera')
        fig.layout.scene2.on_change(cam_change_right, 'camera')

    return fig


def white_layout(**kwargs):
    from warnings import warn
    warn('white_layout is deprecated, use plotly_layout3d instead', DeprecationWarning, stacklevel=2)
    return plotly_layout3d(**kwargs)
