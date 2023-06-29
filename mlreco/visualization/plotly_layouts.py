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
            aspectratio = {axes[i]: 4*v for i, v in enumerate((ranges[:,1]-ranges[:,0])/np.max(ranges[:,1]-ranges[:,0]))}

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


def trace_particles(particles, color='id', size=1,
                    scatter_points=False,
                    mode='points',
                    highlight_primaries=False,
                    randomize_labels=False,
                    colorscale='rainbow', prefix='', opacity=None):
    '''
    Get Scatter3d traces for a list of <Particle> instances.
    Each <Particle> will be drawn with the color specified
    by its unique particle ID.

    Inputs:
        - particles: List[Particle]

    Returns:
        - traces: List[go.Scatter3d]
    '''
    traces, colors = [], []
    for p in particles:
        colors.append(getattr(p, color))
    colors = np.array(colors)
    color_dict = {}
    for p in particles:
        att = getattr(p, color)
        assert np.isscalar(att)
        color_dict[int(att)] = int(att)
    if randomize_labels:
        perm = np.random.permutation(len(color_dict))
        for i, key in enumerate(color_dict):
            color_dict[key] = perm[i]
    colors = np.array(list(color_dict.values()))
    cmin, cmax = int(colors.min()), int(colors.max())
    if color == 'pid':
        cmin, cmax = 0, 6
    alpha = 1
    for p in particles:
        if getattr(p, mode).shape[0] <= 0:
            continue
        c = color_dict[int(getattr(p, color))] * np.ones(getattr(p, mode).shape[0])
        if highlight_primaries and opacity is None:
            if p.is_primary:
                alpha = 1
            else:
                alpha = 0.01
        else:
            alpha = opacity
        plot = go.Scatter3d(x=getattr(p, mode)[:, 0],
                            y=getattr(p, mode)[:, 1],
                            z=getattr(p, mode)[:, 2],
                            mode='markers',
                            marker=dict(
                                size=size,
                                color=c,
                                colorscale=colorscale,
                                cmin=cmin, cmax=cmax,
                                # reversescale=True,
                                opacity=alpha),
                               hovertext=c,
                       name='{}Particle {}'.format(prefix, p.id)
                              )
        traces.append(plot)
        if scatter_points:
            if p.start_point is not None and (p.start_point > 0).any():
                plot = go.Scatter3d(x=np.array([p.start_point[0]]),
                    y=np.array([p.start_point[1]]),
                    z=np.array([p.start_point[2]]),
                    mode='markers',
                    marker=dict(
                        size=5,
                        color='red',
                        # colorscale=colorscale,
                        opacity=0.6),
                        # hovertext=p.ppn_candidates[:, 4],
                    name='{}start_point {}'.format(prefix, p.id))
                traces.append(plot)
            if p.end_point is not None and (p.end_point > 0).any():
                plot = go.Scatter3d(x=np.array([p.end_point[0]]),
                    y=np.array([p.end_point[1]]),
                    z=np.array([p.end_point[2]]),
                    mode='markers',
                    marker=dict(
                        size=5,
                        color='cyan',
                        # line=dict(width=2, color='red'),
                        # cmin=cmin, cmax=cmax,
                        # colorscale=colorscale,
                        opacity=0.6),
                        # hovertext=p.ppn_candidates[:, 4],
                    name='end_point {}'.format(prefix, p.id))
                traces.append(plot)
    return traces


def trace_interactions(interactions, color='id', colorscale="rainbow", prefix=''):
    '''
    Get Scatter3d traces for a list of <Interaction> instances.
    Each <Interaction> will be drawn with the color specified
    by its unique interaction ID.

    Inputs:
        - particles: List[Particle]

    Returns:
        - traces: List[go.Scatter3d]
    '''
    traces, colors = [], []
    for inter in interactions:
        colors.append(getattr(inter, color))
    colors = np.array(colors)
    cmin, cmax = int(colors.min()), int(colors.max())

    for idx, inter in enumerate(interactions):
        particles = inter.particles
        voxels = []
        # Merge all particles' voxels into one tensor
        for p in particles:
            if p.points.shape[0] > 0:
                voxels.append(p.points)
        voxels = np.vstack(voxels)
        plot = go.Scatter3d(x=voxels[:,0],
                            y=voxels[:,1],
                            z=voxels[:,2],
                            mode='markers',
                            marker=dict(
                                size=2,
                                color=int(getattr(inter, color)) * np.ones(voxels.shape[0]),
                                colorscale=colorscale,
                                cmin=cmin, cmax=cmax,
                                reversescale=True,
                                opacity=1),
                               hovertext=int(getattr(inter, color)),
                       name='{}Interaction {}'.format(prefix, getattr(inter, color))
                              )
        traces.append(plot)
        if inter.vertex is not None and (inter.vertex > -1).all():
            plot = go.Scatter3d(x=np.array([inter.vertex[0]]),
                y=np.array([inter.vertex[1]]),
                z=np.array([inter.vertex[2]]),
                mode='markers',
                marker=dict(
                    size=5,
                    color='red',
                    # colorscale=colorscale,
                    opacity=0.6),
                    # hovertext=p.ppn_candidates[:, 4],
                name='{}Vertex {}'.format(prefix, inter.id))
            traces.append(plot)
    return traces
