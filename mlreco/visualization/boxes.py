import numpy as np
import plotly.graph_objs as go


def box_trace(lower, upper, draw_faces=False, **kwargs):
    '''
    Function which produces a plotly trace of a box given its 
    lower bounds and upper bounds in x, y and z.

    Parameters
    ----------
    lower : np.ndarray
        (3) Vector of lower boundaries in x, z and z
    upper : np.ndarray
        (3) Vector of upper boundaries in x, z and z
    **kwargs : dict, optional
        List of additional arguments to pass to plotly.graph_objs.Scatter3D or
        plotly.graph_objs.Mesh3D, depending on the `draw_faces` parameter.

    Returns
    -------
    Union[plotly.graph_objs.Scatter3D, plotly.graph_objs.Mesh3D]
        Box trace
    '''
    # Check the parameters
    assert len(lower) == len(upper) == 3,\
            'Must specify 3 values for both lower and upper boundaries'
    assert np.all(np.asarray(upper) > np.asarray(lower)),\
            'Each upper boundary should be greater than its lower counterpart'

    # List of box vertices in the edges that join them in the box mesh
    box_vertices   = np.array([[0, 0, 0, 0, 1, 1, 1, 1],
                               [0, 0, 1, 1, 0, 0, 1, 1],
                               [0, 1, 0, 1, 0, 1, 0, 1]]).T
    box_edge_index = np.array([[0, 0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 6],
                               [1, 2, 4, 3, 5, 3, 6, 7, 5, 6, 7, 7]])
    box_tri_index  = np.array([[0, 6, 3, 2, 7, 6, 1, 1, 5, 5, 6, 7],
                               [6, 0, 0, 0, 4, 4, 7, 7, 4, 0, 2, 3],
                               [2, 4, 1, 3, 5, 7, 5, 3, 0, 1, 7, 2]])

    # List of scaled vertices
    vertices = lower + box_vertices * (upper - lower)

    # Update hoverinfo style according to kwargs
    kwargs['hoverinfo'] = ['x', 'y', 'z'] if 'hovertext' not in kwargs  else ['x', 'y', 'z', 'text']

    if not draw_faces:
        # List of edges to draw (padded with None values to break them from each other)
        edges = np.full((3*box_edge_index.shape[1], 3), None)
        edges[np.arange(0, edges.shape[0], 3)] = vertices[box_edge_index[0]]
        edges[np.arange(1, edges.shape[0], 3)] = vertices[box_edge_index[1]]

        # Update color of the line specifically, if specified
        if 'color' in kwargs:
            if 'line' in kwargs:
                kwargs['line'].update({'color': kwargs['color']})
            else:
                kwargs['line'] = {'color': kwargs['color']}
            del kwargs['color']

        return go.Scatter3d(x = edges[:,0], y = edges[:,1], z = edges[:,2],
                            mode = 'lines', **kwargs)

    else:

        return go.Mesh3d(x = vertices[:,0], y = vertices[:,1], z = vertices[:,2],
                         i = box_tri_index[0], j = box_tri_index[1], k = box_tri_index[2],
                         **kwargs)

def box_traces(lowers, uppers, draw_faces=False, color=None, hovertext=None, **kwargs):
    '''
    Function which produces a list of plotly traces ofboxes given a list of
    lower bounds and upper bounds in x, y and z.

    Parameters
    ----------
    lower : np.ndarray
        (N, 3) List of vector of lower boundaries in x, z and z
    upper : np.ndarray
        (N, 3) List of vector of upper boundaries in x, z and z
    color : Union[int, str, np.ndarray]
        Color of boxes or list of color of boxes
    hovertext : Union[int, str, np.ndarray]
        Text associated with every box or each box
    **kwargs : dict, optional
        List of additional arguments to pass to plotly.graph_objs.Scatter3D or
        plotly.graph_objs.Mesh3D, depending on the `draw_faces` parameter.

    Returns
    -------
    Union[List[plotly.graph_objs.Scatter3D], List[plotly.graph_objs.Mesh3D]]
        Box traces
    '''
    assert len(lowers) == len(uppers),\
            'Provide as many upper boundary vector as their lower counterpart'
    assert color is None or np.isscalar(color) or len(color) == len(lowers),\
            'Either specify one color for all boxes, or one color per box'
    assert hovertext is None or np.isscalar(hovertext) or len(hovertext) == len(lowers),\
            'Either specify one hovertext for all boxes, or one hovertext per box'

    traces = []
    for i in range(len(lowers)):
        color     = color if (color is None or np.isscalar(color)) else color[i]
        hovertext = hovertext if (hovertext is None or np.isscalar(hovertext)) else hovertext[i]
        traces.append(box_trace(lowers[i], uppers[i], draw_faces, color=color, hovertext=hovertext, **kwargs))

    return traces


def scatter_boxes(coords, dimensions=[1.,1.,1.], color='orange', opacity=0.8, hovertext=None, colorscale=None, **kwargs):
    """
    Produces go.Mesh3d object to be plotted in plotly
    - coords is a list of boxes coordinates (Nx3 matrix)
    """
    base_x = np.array([0, 0, 1, 1, 0, 0, 1, 1]) * boxesize[0]
    base_y = np.array([0, 1, 1, 0, 0, 1, 1, 0]) * boxesize[1]
    base_z = np.array([0, 0, 0, 0, 1, 1, 1, 1]) * boxesize[2]
    trace = []
    cmin, cmax = None, None
    if not isinstance(color, str):
        cmin = min(color)
        cmax = max(color)
    for i in range(len(coords)):
        trace.append(
                go.Mesh3d(
                    x=(coords[i][0]-0.5) * boxesize[0] + base_x,
                    y=(coords[i][1]-0.5) * boxesize[1] + base_y,
                    z=(coords[i][2]-0.5) * boxesize[2] + base_z,
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
