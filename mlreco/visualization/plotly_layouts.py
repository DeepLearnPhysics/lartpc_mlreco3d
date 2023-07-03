import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots


def high_contrast_colors():
    '''
    Produces a list of 48 easily distinguishable colors.

    Returns
    -------
    List[str]
        List of easily distinguishable plotly colors
    '''
    import plotly.express as px

    return np.concatenate([px.colors.qualitative.Dark24, px.colors.qualitative.Light24])


def high_contrast_colorscale():
    '''
    Produces a discrete plotly colorscale based on 48 easily
    distinguishable colors.

    Returns
    -------
    List[[float, str]]
        List of colorscale boundaries and colors
    '''
    colors = high_contrast_colors()
    step = 1./len(colors)

    colorscale = []
    for i, c in enumerate(colors):
        colorscale.append([i*step, c])
        colorscale.append([(i+1)*step, c])

    return colorscale


def plotly_layout3d(meta=None, ranges=None, titles=None, detector_coords=False, backgroundcolor='white', gridcolor='lightgray', width=900, height=900, showlegend=True, camera=None, aspectmode='manual', aspectratio=None, **kwargs):
    """
    Produces go.Layout object for a certain format.

    Parameters
    ----------
    meta : np.ndarray, optional
        (9) Metadata information used to infer the full image range
    ranges : np.ndarray, optional
        (3, 2) or (N, 3) Array used to specify the plot region in (x,y,z) directions.
        The default (None) will determine the range to include all points.
        Alternatively can be an array of shape (3,2) specifying (x,y,z) axis (min,max) range for a display,
        or simply a list of points with shape (N,3+) where [:,0],[:,1],[:,2] correspond to (x,y,z) values and
        the plotting region is decided by measuring the min,max range in each coordinates. This last option
        is useful if one wants to define the region based on a set of points that is not same as what's plotted.
    titles : List[str], optional
        (3) Array of strings for (x,y,z) axis title respectively
    detector_coords : bool, default False
        Whether or not the coordinates being drawn are in detector_coordinates or pixel IDs
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
    camera : dict, optional
        Plotly dictionary which specifies the orientation of the camera in 3D
    aspectmode : str, default manual
        Plotly aspect mode. If manual, will define it based on the ranges
    aspectratio : dict, optional
        Plotly dictionary which specifies the aspect ratio for x, y an d z
    **kwargs : dict, optional
        List of additional arguments to pass to plotly.graph_objs.Layout

    Results
    -------
    plotly.graph_objs.Layout
        Object that can be given to go.Figure for visualization (together with traces)
    """
    # Figure out the drawing range
    if ranges is None:
        ranges = [None, None, None]
    else:
        if ranges.shape != (3, 2):
            assert len(ranges.shape) == 2 and ranges.shape[1] == 3,\
                'If ranges is not of shape (3, 2), it must be of shape (N, 3)'
            ranges = np.vstack([np.min(ranges, axis=0), np.max(ranges, axis=0)]).T

        # Check that the range is sensible
        assert np.all(ranges[:,1] >= ranges[:,0])

    if meta is not None:
        assert ranges is None or None in ranges, 'Should not specify both `ranges` and `meta` parameters'
        assert len(np.asarray(meta).reshape(-1)) == 9, 'Metadata should be an array of 9 values'
        lowers, uppers, sizes = np.split(np.asarray(meta).reshape(-1), 3)
        if detector_coords:
            ranges = np.vstack([lowers, uppers]).T
        else:
            ranges = np.vstack([[0, 0, 0], np.round((uppers-lowers)/sizes)]).T

    # Get the default units
    unit = 'cm' if detector_coords else 'pixel'

    # Infer the image width/height and aspect ratios, unless they are specified
    if aspectmode == 'manual':
        if aspectratio is None:
            axes = ['x', 'y', 'z']
            ratios = np.ones(len(ranges)) if ranges[0] is None else (ranges[:,1]-ranges[:,0])/np.max(ranges[:,1]-ranges[:,0])
            aspectratio = {axes[i]: v for i, v in enumerate(ratios)}

    # Check on the axis titles
    assert titles is None or len(titles) == 3, 'Must specify one title per axis'

    # Initialize some default camera angle if it is not specified
    if camera is None:
        camera = dict(up = dict(x = 0, y = 0, z = 1),
                      center = dict(x = 0, y = 0, z = 0),
                      eye = dict(x = 1.2, y = 1.2, z = 0.075))

    # Initialize layout
    layout = go.Layout(
        showlegend = showlegend,
        width = width,
        height = height,
        margin = dict(l = 0, r = 0, b = 0, t = 0),
        legend = dict(title = 'Legend', tracegroupgap = 1),
        scene = dict(
            xaxis = dict(nticks = 10, range = ranges[0], showticklabels = True,
                         title = dict(text=f'x [{unit}]' if titles is None else titles[0], font=dict(size=20)),
                         tickfont = dict(size=14),
                         backgroundcolor = backgroundcolor,
                         gridcolor = gridcolor,
                         showbackground = True
                        ),
            yaxis = dict(nticks = 10, range = ranges[1], showticklabels = True,
                         title = dict(text=f'y [{unit}]' if titles is None else titles[1], font=dict(size=20)),
                         tickfont = dict(size=14),
                         backgroundcolor = backgroundcolor,
                         gridcolor = gridcolor,
                         showbackground = True
                        ),
            zaxis = dict(nticks = 10, range = ranges[2], showticklabels = True,
                         title = dict(text=f'z [{unit}]' if titles is None else titles[2], font=dict(size=20)),
                         tickfont = dict(size=14),
                         backgroundcolor = backgroundcolor,
                         gridcolor = gridcolor,
                         showbackground = True
                        ),
            aspectmode = aspectmode,
            aspectratio = aspectratio,
            camera = camera
        ),
        **kwargs
    )

    return layout


def white_layout():
    bg_color = 'rgba(0,0,0,0)'
    grid_color = 'rgba(220,220,220,100)'
    layout = dict(showlegend=False,
        autosize=True,
        height=1000,
        width=1000,
        margin=dict(r=20, l=20, b=20, t=20),
        plot_bgcolor='rgba(255,255,255,0)',
        paper_bgcolor='rgba(255,255,255,0)',
        scene1=dict(xaxis=dict(dict(backgroundcolor=bg_color,
                                    gridcolor=grid_color)),
                    yaxis=dict(dict(backgroundcolor=bg_color,
                                    gridcolor=grid_color)),
                    zaxis=dict(dict(backgroundcolor=bg_color,
                                    gridcolor=grid_color)),
                    aspectmode='cube'),
        scene2=dict(xaxis=dict(dict(backgroundcolor=bg_color,
                                    gridcolor=grid_color)),
                    yaxis=dict(dict(backgroundcolor=bg_color,
                                    gridcolor=grid_color)),
                    zaxis=dict(dict(backgroundcolor=bg_color,
                                    gridcolor=grid_color)),
                    aspectmode='cube'),
        scene3=dict(xaxis=dict(dict(backgroundcolor=bg_color,
                                    gridcolor=grid_color)),
                    yaxis=dict(dict(backgroundcolor=bg_color,
                                    gridcolor=grid_color)),
                    zaxis=dict(dict(backgroundcolor=bg_color,
                                    gridcolor=grid_color)),
                    aspectmode='cube'))
    return layout


def dualplot(traces_left, traces_right, spatial_size=768, layout=None,
             titles=['Left', 'Right']):

    if layout is None:
        layout = white_layout()

    fig = make_subplots(rows=1, cols=2,
                        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
                        subplot_titles=(titles[0], titles[1]),
                        horizontal_spacing=0.05, vertical_spacing=0.04)
    fig.add_traces(traces_left, rows=[1]*len(traces_left), cols=[1]*len(traces_left))
    fig.add_traces(traces_right, rows=[1]*len(traces_right), cols=[2]*len(traces_right))

    fig.layout = layout
    fig.update_layout(showlegend=True,
                      legend=dict(xanchor="left"),
                      autosize=True,
                      height=500,
                      width=1000)
    fig.update_layout(
        scene1 = dict(
            xaxis = dict(range=[0,spatial_size],),
                         yaxis = dict(range=[0,spatial_size],),
                         zaxis = dict(range=[0,spatial_size],),),
        scene2 = dict(
            xaxis = dict(range=[0,spatial_size],),
                         yaxis = dict(range=[0,spatial_size],),
                         zaxis = dict(range=[0,spatial_size],),),
        margin=dict(r=20, l=10, b=10, t=10))
    return fig
