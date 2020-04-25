import numpy as np
import plotly.graph_objs as go

def scatter_label(voxels, labels, markersize=1):
    """
    scatter plot of voxels colored by labels
    - voxels is a list of voxel coordinates (Nx3-matrix)
    - labels is a list of voxel labels (N-vector)
    """
    _, cs = np.unique(labels, return_inverse=True)
    trace = go.Scatter3d(x=voxels[:,0], y=voxels[:,1], z=voxels[:,2],
                        mode='markers',
                        marker = dict(
                            size = markersize,
                            color = cs,
                            colorscale='Viridis',
                            opacity=0.8
                        ), 
                        hovertext=labels)
    return [trace]

def scatter_voxels(voxels, markersize=1):
    """
    scatter plot of voxels colored by labels
    - voxels is a list of voxel coordinates (Nx3-matrix)
    - labels is a list of voxel labels (N-vector)
    """
    trace = go.Scatter3d(x=voxels[:,0], y=voxels[:,1], z=voxels[:,2],
                        mode='markers',
                        marker = dict(
                            size = markersize,
                            colorscale='Viridis',
                            opacity=0.8
                        ))
    return [trace]