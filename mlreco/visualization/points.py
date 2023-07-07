import plotly.graph_objs as go
import numpy as np
from mlreco.utils.globals import *
from analysis.classes.data import *

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
    coords_col = (1, 4)
    if dim == 2:
        coords_col = (1, 3)
    if points.shape[1] == dim:
        coords_col = (0, dim)

    hoverinfo=['x','y','text'] if dim == 2 else ['x','y','z','text']
    if not type(color) == type(str()):
        if not len(points) == len(color):
            print('ERROR: size of the points (%d) not matching with the color (%d)' % (len(points),len(color)))
            raise ValueError
    if hovertext is None:
        if color is not None and not type(color) == type(str()):
            if dim == 2:
                hovertext = ['x: %.2f<br>y: %.2f<br>value: %.2f' % tuple(np.concatenate([points[i,coords_col[0]:coords_col[1]].flatten(),color[i].flatten()])) for i in range(len(points))]
            elif dim == 3:
                hovertext = ['x: %.2f<br>y: %.2f<br>z: %.2f<br>value: %.2f' % tuple(np.concatenate([points[i,coords_col[0]:coords_col[1]].flatten(),color[i].flatten()])) for i in range(len(points))]
            hoverinfo = 'text'
        else:
            if dim == 2:
                hovertext = ['x: %.2f<br>y: %.2f' % tuple(points[i,coords_col[0]:coords_col[1]].flatten()) for i in range(len(points))]
            if dim == 3:
                hovertext = ['x: %.2f<br>y: %.2f<br>z: %.2f' % tuple(points[i,coords_col[0]:coords_col[1]].flatten()) for i in range(len(points))]
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
        x=points[:,coords_col[0]],
        y=points[:,coords_col[0]+1],
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
        args['z'] = points[:,coords_col[0]+2]
        return [go.Scatter3d(**args)]
    else:
        return [go.Scatter(**args)]


class Scatter3D:

    def __init__(self):

        self._traces = []
        self._colors = {}

        self._color_bounds = [None, None]

    def clear_state(self):
        self._traces = []
        self._colors = {}
        self._color_bounds = [None, None]

    def scatter_start_points(self, particles, prefix=''):
        for p in particles:
            if p.start_point is not None and (np.abs(p.start_point) < 1e8).all():
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
                    name='{} {} Startpoint'.format(type(p).__name__, p.id))
                self._traces.append(plot)

    def scatter_end_points(self, particles, prefix=''):
        for p in particles:
            if p.end_point is not None and (np.abs(p.end_point) < 1e8).all():
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
                    name='{} {} Endpoint'.format(type(p).__name__, p.id))
                self._traces.append(plot)

    def scatter_vertices(self, interactions):
        for ia in interactions:
            if ia.vertex is not None and (np.abs(ia.vertex) < 1e8).all():
                plot = go.Scatter3d(x=np.array([ia.vertex[0]]),
                    y=np.array([ia.vertex[1]]),
                    z=np.array([ia.vertex[2]]),
                    mode='markers',
                    marker=dict(
                        size=5,
                        color='cyan',
                        # line=dict(width=2, color='red'),
                        # cmin=cmin, cmax=cmax,
                        # colorscale=colorscale,
                        opacity=0.6),
                        # hovertext=p.ppn_candidates[:, 4],
                    name='{} {} Vertex'.format(type(ia).__name__, ia.id))
                self._traces.append(plot)

    def set_pixel_color(self, objects, color, mode='points'):

        cmin, cmax = np.inf, -np.inf

        if 'depositions' not in color:
            for entry in objects:
                attribute = getattr(entry, color)
                assert np.isscalar(attribute)
                self._colors[int(entry.id)] = int(attribute) \
                    * np.ones(getattr(entry, mode).shape[0], dtype=np.int64)

                if int(attribute) < cmin:
                    cmin = int(attribute)
                if int(attribute) > cmax:
                    cmax = int(attribute)
        else:
            for entry in objects:
                depositions = getattr(entry, color)
                assert isinstance(depositions, np.ndarray)
                self._colors[int(entry.id)] = depositions
                dmin, dmax = depositions.min(), depositions.max()
                if dmin < cmin:
                    cmin = dmin
                if dmax > cmax:
                    cmax = dmax


        self._color_bounds = [cmin, cmax]
                
        # Define limits
        if color == 'pid':
            values = list(PID_LABELS.keys())
            self._color_bounds = [-1, max(values)]
        elif color == 'semantic_type':
            values = list(SHAPE_LABELS.keys())
            self._color_bounds = [-1, max(values)]
        elif color == 'is_primary':
            self._color_bounds = [-1, 1]
        elif 'depositions' in color:
            self._color_bounds = [0, cmax]


    def check_attribute_name(self, objects, color):

        attr_list = [att for att in dir(objects[0]) if att[0] != '_']
        if color not in attr_list:
            raise ValueError(f'"{color}" is not a valid attribute for object type {type(objects[0])}!')

    def __call__(self, objects, color='id', mode='points', colorscale='rainbow', size=1, **kwargs):
        
        self.check_attribute_name(objects, color)
        self.clear_state()

        self.set_pixel_color(objects, color, mode)

        for entry in objects:
            if getattr(entry, mode).shape[0] <= 0:
                continue
            c = self._colors[int(entry.id)]

            plot = go.Scatter3d(x=getattr(entry, mode)[:, 0],
                                y=getattr(entry, mode)[:, 1],
                                z=getattr(entry, mode)[:, 2],
                                mode='markers',
                                marker=dict(
                                    size=size,
                                    color=c,
                                    colorscale=colorscale,
                                    cmin=self._color_bounds[0], cmax=self._color_bounds[1]),
                                hovertext=c,
                        name='{} {}'.format(type(entry).__name__, entry.id)
                                )
            self._traces.append(plot)

        if isinstance(objects[0], Particle) and kwargs.get('scatter_start_points', False):
            self.scatter_start_points(objects)
        if isinstance(objects[0], Particle) and kwargs.get('scatter_end_points', False):
            self.scatter_end_points(objects)
        if isinstance(objects[0], Interaction) and kwargs.get('scatter_vertices', False):
            self.scatter_vertices(objects)
        
        return self._traces

get_event_displays = Scatter3D()
# Legacy
trace_particles = Scatter3D()
trace_interactions = Scatter3D()

def trace_objects(particles, color='id', size=1,
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
    
    # Randomize labels if requested (useful for coloring id)
    if randomize_labels:
        perm = np.random.permutation(len(color_dict))
        for i, key in enumerate(color_dict):
            color_dict[key] = perm[i]

    # Set color minmax boundaries
    colors = np.array(list(color_dict.values()))
    cmin, cmax = int(colors.min()), int(colors.max())
    if color == 'pid':
        values = list(PID_LABELS.keys())
        cmin, cmax = -1, max(values)
    elif color == 'semantic_type':
        values = list(SHAPE_LABELS.keys())
        cmin, cmax = min(values), max(values)
    elif color == 'is_primary':
        cmin, cmax = 0, 1

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
    return traces