import plotly
import plotly.graph_objs as go
import numpy as np

from mlreco.utils.globals import COORD_COLS, PID_LABELS, SHAPE_LABELS

from analysis.classes.data import *

def scatter_points(points, color=None, colorscale=None, cmin=None, cmax=None, opacity=None, markersize=None, hovertext=None, dim=3, **kwargs):
    '''
    Produces plotly.graph_objs.Scatter3d or plotly.graph_objs.Scatter
    trace object to be drawn in plotly. The object is nested to be fed
    directly to a plotly.graph_objs.Figure or plotly.offline.iplot. All
    of the regular plotly attribute are available.

    Parameters
    ----------
    points : np.ndarray
        (N, 2+) array of N points of (..., x, y, [z],...) coordinate information
    dim : int, default 3
        Dimension (can either be 2 or 3)
    color : Union[str, np.ndarray], optional
        Color of markers or (N) list of color of markers
    colorscale : Union[str, List[str], List[List[float, str]], optional
        Plotly colorscale specifier for the markers
    cmin : Union[int, float], optional
        Minimum of the color range
    cmax : Union[int, float], optional
        Maximum of the color range
    opacity : float
        Marker opacity
    markersize : int, optional
        Specified marker size
    hovertext : Union[List[str], List[int], optional
        (N) List of labels associated with each marker
    **kwargs : dict, optional
        List of additional arguments to pass to plotly.graph_objs.Scatter3D

    Returns
    -------
    List[go.Scatter3d]
        (1) List with one graph of the input points
    '''
    # Check the dimension for compatibility
    if not dim in [2, 3]:
        print('This function only supports dimension be 2 or 3')
        raise ValueError
    if points.shape[1] == 2:
        dim = 2

    # Get the coordinate column locations in the input tensor
    coord_cols = COORD_COLS
    if dim == 2:
        coord_cols = COORD_COLS[:2]
    if points.shape[1] == dim:
        coord_cols = np.arange(dim)

    # If there is hovertext, print the color as part of the hovertext
    if hovertext is None and color is not None and type(color) != str:
        hovertext = color

    # Update hoverinfo
    kwargs['hoverinfo'] = ['x', 'y', 'z'] if dim == 3 else ['x', 'y']
    if hovertext is not None:
        kwargs['hoverinfo'] += ['text']

    # If only cmin or cmax is defined, must figure out the other
    if (cmin is None) ^ (cmax is None) and color is not None and not np.isscalar(color):
        if not cmin: cmin = min(np.min(color), cmax*(1-1e-6))
        if not cmax: cmax = max(np.max(color), cmin*(1+1e-6))

    # Initialize and return
    trace_dict = dict(
            x = points[:,coord_cols[0]],
            y = points[:,coord_cols[1]],
            mode = 'markers',
            marker = dict(
                size = markersize,
                color = color,
                colorscale = colorscale,
                opacity = opacity,
                cmin = cmin,
                cmax = cmax
            ),
            text = hovertext,
            **kwargs
            )

    if dim == 3:
        trace_dict['z'] = points[:,coord_cols[2]]
        return [go.Scatter3d(**trace_dict)]
    else:
        return [go.Scatter(**trace_dict)]


class Scatter3D:

    def __init__(self):

        self._traces = []
        self._colors = {}

        self._color_bounds = [None, None]
        self._colorscale = None

    def clear_state(self):
        self._traces = []
        self._colors = {}
        self._color_bounds = [None, None]
        self._colorscale = None

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

    def set_pixel_color(self, objects, color, colorscale=None, precmin=None, precmax=None, mode='points'):

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

        # If manually specified, overrule
        if precmin is not None: self._color_bounds[0] = precmin
        if precmax is not None: self._color_bounds[1] = precmax

        # Define colorscale
        self._colorscale = colorscale
        if isinstance(colorscale, str) and hasattr(plotly.colors.qualitative, colorscale):
            self._colorscale = getattr(plotly.colors.qualitative, colorscale)
        if isinstance(colorscale, list) and isinstance(colorscale[0], str):
            count = np.round(self._color_bounds[1] - self._color_bounds[0]) + 1
            if count < len(colorscale):
                self._colorscale = colorscale[:count]
            if count > len(colorscale):
                repeat = int((count-1)/len(colorscale)) + 1
                self._colorscale = np.repeat(colorscale, repeat)[:count]


    def check_attribute_name(self, objects, color):

        attr_list = [att for att in dir(objects[0]) if att[0] != '_']
        if color not in attr_list:
            raise ValueError(f'"{color}" is not a valid attribute for object type {type(objects[0])}!')

    def __call__(self, objects, color='id', mode='points', colorscale='rainbow', cmin=None, cmax=None, size=1, scatter_start_points=False, scatter_end_points=False, scatter_vertices=False, **kwargs):

        if not len(objects):
            return []

        self.check_attribute_name(objects, color)
        self.clear_state()

        self.set_pixel_color(objects, color, colorscale, cmin, cmax, mode)

        for entry in objects:
            if getattr(entry, mode).shape[0] <= 0:
                continue
            c = self._colors[int(entry.id)].tolist()
            hovertext = [f'{color}: {ci}' for ci in c]

            plot = scatter_points(getattr(entry, mode)[:, :3],
                                  color = c, colorscale = self._colorscale,
                                  cmin = self._color_bounds[0], cmax = self._color_bounds[1],
                                  markersize = size, hovertext = hovertext,
                                  name = '{} {}'.format(type(entry).__name__, entry.id), **kwargs)

            self._traces += plot

        if isinstance(objects[0], Particle) and scatter_start_points:
            self.scatter_start_points(objects)
        if isinstance(objects[0], Particle) and scatter_end_points:
            self.scatter_end_points(objects)
        if isinstance(objects[0], Interaction) and scatter_vertices:
            self.scatter_vertices(objects)

        return self._traces

get_event_displays = Scatter3D()
# Legacy
trace_particles = Scatter3D()
trace_interactions = Scatter3D()
