import plotly.graph_objs as go

def scatter_points(points, markersize=5, color='orange', colorscale=None, opacity=0.8, hovertext=None, cmin=None, cmax=None):
    """
    Produces go.Scatter3d object to be plotted in plotly
    - voxels is a list of voxel coordinates (Nx3-matrix)
    - labels is a list of voxel labels (N-vector)
    INPUTS:
     - points is (N,3+) shaped array for N points of (x,y,z,...) coordinate information
     - markersize specifies the size of a marker (drawn per point)
     - color can be a string to specify a color for all points, or an array of (N) values for gradient color
     - colorscale defines the gradient colors to be used when color values are specified per point
     - opacity is a transparency of each points when drawn
     - hovertext is an additional text to be shown when a cursor hovers on a point (interactive legend)
    OUTPUT:
     - go.Scatter3d object
    """
    if hovertext is None and color is not None and not type(color) == type(str()):
        hovertext = ['%.2f' % float(v) for v in color]
    if cmin is None and color is not None and not type(color) == type(str()):
        cmin = min(color)
    if cmax is None and color is not None and not type(color) == type(str()):
        cmax = max(color)
    trace = go.Scatter3d(x=points[:,0], y=points[:,1], z=points[:,2],
                         mode='markers',
                         marker = dict(
                            size = markersize,
                            color=color,
                            colorscale=colorscale,
                            opacity=opacity,
                            cmin=cmin,
                            cmax=cmax
                         ),
                         hoverinfo = ['x','y','z'] if hovertext is None else ['x','y','z','text'],
                         hovertext = hovertext,
                        )
    return [trace]
