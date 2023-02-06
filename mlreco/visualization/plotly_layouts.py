import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots


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
                    scatter_ppn=False,
                    highlight_primaries=False,
                    colorscale='rainbow'):
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
    cmin, cmax = int(colors.min()), int(colors.max())
    opacity = 1
    for p in particles:
        c = int(getattr(p, color)) * np.ones(p.points.shape[0])
        if highlight_primaries:
            if p.is_primary:
                opacity = 1
            else:
                opacity = 0.01
        plot = go.Scatter3d(x=p.points[:,0],
                            y=p.points[:,1],
                            z=p.points[:,2],
                            mode='markers',
                            marker=dict(
                                size=size,
                                color=c,
                                colorscale=colorscale,
                                cmin=cmin, cmax=cmax,
                                # reversescale=True,
                                opacity=opacity),
                               hovertext=int(getattr(p, color)),
                       name='Particle {}'.format(p.id)
                              )
        traces.append(plot)
        if scatter_points:
            if p.startpoint is not None:
                plot = go.Scatter3d(x=np.array([p.startpoint[0]]),
                    y=np.array([p.startpoint[1]]),
                    z=np.array([p.startpoint[2]]),
                    mode='markers',
                    marker=dict(
                        size=5,
                        color='red',
                        # colorscale=colorscale,
                        opacity=0.6),
                        # hovertext=p.ppn_candidates[:, 4],
                    name='Startpoint {}'.format(p.id))
                traces.append(plot)
            if p.endpoint is not None:
                plot = go.Scatter3d(x=np.array([p.endpoint[0]]),
                    y=np.array([p.endpoint[1]]),
                    z=np.array([p.endpoint[2]]),
                    mode='markers',
                    marker=dict(
                        size=5,
                        color='cyan',
                        # line=dict(width=2, color='red'),
                        # cmin=cmin, cmax=cmax,
                        # colorscale=colorscale,
                        opacity=0.6),
                        # hovertext=p.ppn_candidates[:, 4],
                    name='Endpoint {}'.format(p.id))
                traces.append(plot)
        elif scatter_ppn:
            plot = go.Scatter3d(x=p.ppn_candidates[:, 0],
                    y=p.ppn_candidates[:, 1],
                    z=p.ppn_candidates[:, 2],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color='red',
                        # colorscale=colorscale,
                        opacity=1),
                        # hovertext=p.ppn_candidates[:, 4],
                    name='PPN {}'.format(p.id))
            traces.append(plot)
    return traces


def trace_interactions(interactions, color='id', colorscale="rainbow"):
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
                       name='Interaction {}'.format(getattr(inter, color))
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
                name='Vertex {}'.format(inter.id))
            traces.append(plot)
    return traces


def plotly_layout3d(ranges=None, titles=None, **kwargs):
    """
    Produces go.Layout object for a certain format.

    INPUTS
        - ranges can be used to specify the plot region in (x,y,z) directions.
          The default (None) will determine the range to include all points.
          Alternatively can be an array of shape (3,2) specifying (x,y,z) axis (min,max) range for a display,
          or simply a list of points with shape (N,3+) where [:,0],[:,1],[:,2] correspond to (x,y,z) values and
          the plotting region is decided by measuring the min,max range in each coordinates. This last option
          is useful if one wants to define the region based on a set of points that is not same as what's plotted.
        - titles can be specified as a length 3 array of strings for (x,y,z) axis title respectively
    OUTPUTS
        - The return is go.Layout object that can be given to go.Figure for visualization (together with traces)
    """
    xrange,yrange,zrange=None,None,None
    if ranges is None:

        ranges=[None,None,None]
    elif np.shape(ranges) == (3,2):
        xrange,yrange,zrange=ranges
    else:
        xrange = (np.min(ranges[:,0]),np.max(ranges[:,0]))
        yrange = (np.min(ranges[:,1]),np.max(ranges[:,1]))
        zrange = (np.min(ranges[:,2]),np.max(ranges[:,2]))

    layout = go.Layout(
        showlegend=True,
        width=768,
        height=768,
        #xaxis=titles[0], yaxis=titles[1], zaxis=titles[2],
        margin=dict(l=0,r=0,b=0,t=0),
        scene = dict(
            xaxis = dict(nticks=10, range = xrange, showticklabels=True,
                         title='x' if titles is None else titles[0],
                         backgroundcolor=None,
                         gridcolor="rgb(255, 255, 255)",
                         showbackground=True,
                        ),
            yaxis = dict(nticks=10, range = yrange, showticklabels=True,
                         title='y' if titles is None else titles[1],
                         backgroundcolor=None,
                         gridcolor="rgb(255, 255, 255)",
                         showbackground=True
                        ),
            zaxis = dict(nticks=10, range = zrange, showticklabels=True,
                         title='z' if titles is None else titles[2],
                         backgroundcolor=None,
                         gridcolor="rgb(255, 255, 255)",
                         showbackground=True,
                        ),
            aspectmode='cube',
            camera = dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.2, y=1.2, z=0.075)
            ),
        ),
        **kwargs
    )
    return layout
