import plotly.graph_objs as go

def scatter_points(points, markersize=5, color='orange'):
    """
    scatter plot of voxels colored by labels
    - voxels is a list of voxel coordinates (Nx3-matrix)
    - labels is a list of voxel labels (N-vector)
    """
    trace = go.Scatter3d(x=points[:,0], y=points[:,1], z=points[:,2],
                        mode='markers',
                        marker = dict(
                            size = markersize,
                            color=color,
                            opacity=0.8
                        ))
    return [trace]