import numpy as np
import plotly.graph_objs as go


def box_trace(lower, upper, draw_faces=False, linewidth=2, **kwargs):
    '''
    Function which produces a plotly trace of a box given its
    lower bounds and upper bounds in x, y and z.

    Parameters
    ----------
    lower : np.ndarray
        (3) Vector of lower boundaries in x, z and z
    upper : np.ndarray
        (3) Vector of upper boundaries in x, z and z
    draw_faces : bool, default False
        Weather or not to draw the box faces, or only the edges
    linewidth : int, default 2
        Width of the box edge lines
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

        # Update the width of the line, if specified
        if 'line' in kwargs and 'width' not in kwargs['line']:
            kwargs['line'].update({'width': linewidth})
        elif 'line' not in kwargs:
            kwargs['line'] = {'width': linewidth}

        # Return trace
        return go.Scatter3d(x = edges[:,0], y = edges[:,1], z = edges[:,2],
                            mode = 'lines', **kwargs)

    else:
        # Return trace
        return go.Mesh3d(x = vertices[:,0], y = vertices[:,1], z = vertices[:,2],
                         i = box_tri_index[0], j = box_tri_index[1], k = box_tri_index[2],
                         **kwargs)


def box_traces(lowers, uppers, draw_faces=False, color=None, hovertext=None, linewidth=2, **kwargs):
    '''
    Function which produces a list of plotly traces of boxes
    given a list of lower bounds and upper bounds in x, y and z.

    Parameters
    ----------
    lowers : np.ndarray
        (N, 3) List of vector of lower boundaries in x, z and z
    uppers : np.ndarray
        (N, 3) List of vector of upper boundaries in x, z and z
    draw_faces : bool, default False
        Weather or not to draw the box faces, or only the edges
    color : Union[str, np.ndarray], optional
        Color of boxes or list of color of boxes
    hovertext : Union[int, str, np.ndarray], optional
        Text associated with every box or each box
    linewidth : int, default 2
        Width of the box edge lines
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
        traces.append(box_trace(lowers[i], uppers[i], draw_faces, linewidth, color=color, hovertext=hovertext, **kwargs))

    return traces


def scatter_boxes(coords, dimensions=[1.,1.,1.], draw_faces=True, color='orange', hovertext=None, linewidth=2, **kwargs):
    """
    Function which produces a list of plotly traces of boxes
    given a list of coordinates and a box dimension.

    This function assumes that the coordinates are in a space
    where an offset of (1, 1, 1) corresponds to an offset of
    (b_x, b_y, b_z), with the latter the dimension of the boxes
    along the three axes. This can be used to represent the PPN
    regions of interest in a space compressed by a factor
    (b_x, b_y, b_z) from the original image resolution.

    Parameters
    ----------
    coords : np.ndarray
        (N, 3) Coordinates of in multiples of box lengths in each dimension
    dimensions : np.ndarray
        (3) Dimensions of the box in each dimension, i.e. (b_x, b_y, b_z)
    draw_faces : bool, default True
        Weather or not to draw the box faces, or only the edges
    color : Union[str, np.ndarray], default 'orange'
        Color of boxes or list of color of boxes
    hovertext : Union[int, str, np.ndarray], optional
        Text associated with every box or each box
    linewidth : int, default 2
        Width of the box edge lines
    **kwargs : dict, optional
        List of additional arguments to pass to plotly.graph_objs.Scatter3D or
        plotly.graph_objs.Mesh3D, depending on the `draw_faces` parameter.

    Returns
    -------
    Union[List[plotly.graph_objs.Scatter3D], List[plotly.graph_objs.Mesh3D]]
        Box traces
    """
    # Check the input
    assert len(dimensions) == 3, 'Must specify three dimensions for the box size'

    # Compute the lower and upper boundaries
    lowers = coords * np.asarray(dimensions)
    uppers = (coords + 1.) * np.asarray(dimensions)

    return box_traces(lowers, uppers, draw_faces, color, hovertext, linewidth, **kwargs)
