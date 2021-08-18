import plotly.graph_objs as go
import numpy as np

def scatter_points(points, dim=3, markersize=5, color='orange', colorscale=None, opacity=0.8, hovertext=None, cmin=None, cmax=None, **kwargs):
    """
    Produces go.Scatter3d or go.Scatter object to be plotted in plotly
    - voxels is a list of voxel coordinates (Nx2 or Nx3 matrix)
    - labels is a list of voxel labels (N vector)
    INPUTS:
     - points is (N,2+) shaped array for N points of (x,y,[z],...) coordinate information
     - markersize specifies the size of a marker (drawn per point)
     - color can be a string to specify a color for all points, or an array of (N) values for gradient color
     - colorscale defines the gradient colors to be used when color values are specified per point
     - opacity is a transparency of each points when drawn
     - hovertext is an additional text to be shown when a cursor hovers on a point (interactive legend)
    OUTPUT:
     - go.Scatter3d object
    """
    hoverinfo=['x','y','text'] if dim == 2 else ['x','y','z','text']
    if not type(color) == type(str()):
        if not len(points) == len(color):
            print('ERROR: size of the points (%d) not matching with the color (%d)' % (len(points),len(color)))
            raise ValueError
    if hovertext is None:
        if color is not None and not type(color) == type(str()):
            if dim == 2:
                hovertext = ['x: %.2f<br>y: %.2f<br>value: %.2f' % tuple(np.concatenate([points[i,1:3].flatten(),color[i].flatten()])) for i in range(len(points))]
            elif dim == 3:
                hovertext = ['x: %.2f<br>y: %.2f<br>z: %.2f<br>value: %.2f' % tuple(np.concatenate([points[i,1:4].flatten(),color[i].flatten()])) for i in range(len(points))]
            hoverinfo = 'text'
        else:
            if dim == 2:
                hovertext = ['x: %.2f<br>y: %.2f' % tuple(points[i,1:3].flatten()) for i in range(len(points))]
            if dim == 3:
                hovertext = ['x: %.2f<br>y: %.2f<br>z: %.2f' % tuple(points[i,1:4].flatten()) for i in range(len(points))]
            hoverinfo = 'text'
    if cmin is None and color is not None and not type(color) == type(str()):
        cmin = min(color)
    if cmax is None and color is not None and not type(color) == type(str()):
        cmax = max(color)

    if not dim in [2,3]:
        print('dim argument must be 2 or 3!')
        raise ValueError
    if points.shape[1] == 2:
        dim = 2

    args=dict(
        x=points[:,1],
        y=points[:,2],
        mode='markers',
        marker = dict(
            size = markersize,
            color=color,
            colorscale=colorscale,
            opacity=opacity,
            cmin=cmin,
            cmax=cmax,
        ),
        hoverinfo = hoverinfo,
        text = hovertext,
        )
    args.update(kwargs)

    if dim == 3:
        args['z'] = points[:,3]
        return [go.Scatter3d(**args)]
    else:
        return [go.Scatter(**args)]
