import plotly.graph_objs as go
import numpy as np


def scatter_cubes(coords, cubesize=[1.,1.,1.], color='orange', opacity=0.8, hovertext=None, colorscale=None, **kwargs):
    """
    Produces go.Mesh3d object to be plotted in plotly
    - coords is a list of cubes coordinates (Nx3 matrix)
    """
    base_x = np.array([0, 0, 1, 1, 0, 0, 1, 1]) * cubesize[0]
    base_y = np.array([0, 1, 1, 0, 0, 1, 1, 0]) * cubesize[1]
    base_z = np.array([0, 0, 0, 0, 1, 1, 1, 1]) * cubesize[2]
    trace = []
    cmin, cmax = None, None
    if not isinstance(color, str):
        cmin = min(color)
        cmax = max(color)
    for i in range(len(coords)):
        trace.append(
                go.Mesh3d(
                    x=(coords[i][0]-0.5) * cubesize[0] + base_x,
                    y=(coords[i][1]-0.5) * cubesize[1] + base_y,
                    z=(coords[i][2]-0.5) * cubesize[2] + base_z,
                    i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                    j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                    k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
                    opacity=opacity,
                    color=color if isinstance(color, str) else color[i],
                    colorscale=colorscale,
                    cmin=cmin,
                    cmax=cmax,
                    hoverinfo=['x','y','z'] if hovertext is None else ['x', 'y', 'z','text'],
                    hovertext=hovertext,
                    **kwargs
                    )
                )
    return trace
