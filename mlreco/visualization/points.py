import plotly.graph_objs as go

def scatter_points(points, dim=3, markersize=5, color='orange', colorscale=None, opacity=0.8, hovertext=None, cmin=None, cmax=None):
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
    if hovertext is None and color is not None and not type(color) == type(str()):
        hovertext = ['%.2f' % float(v) for v in color]
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
        x=points[:,0],
        y=points[:,1],
        mode='markers',
        marker = dict(
            size = markersize,
            color=color,
            colorscale=colorscale,
            opacity=opacity,
            cmin=cmin,
            cmax=cmax,
        ),
        hoverinfo = ['x','y'] if hovertext is None else ['x','y','text'],
        hovertext = hovertext,
        )
    
    if dim == 3:
        args['z'] = points[:,2]
        args['hoverinfo'] = ['x','y','z'] if hovertext is None else ['x','y','z','text']
        return [go.Scatter3d(**args)]
    else:
        return [go.Scatter(**args)]
    
