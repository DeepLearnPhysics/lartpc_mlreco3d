import numpy as np
import plotly.graph_objs as go

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
