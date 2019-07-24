import numpy as np
import plotly.graph_objs as go

def scatter_clusters(voxels, labels, clusters, markersize=5):
    """
    scatter plot of cluster voxels colored by cluster order
    - voxels is a list of voxel coordinates (Nx3-matrix)
    - labels is a list of voxel labels (N-vector)
    - clusters is an array of clusters, each defined as an array of voxel ids
    """
    # first build voxel set
    voxels = np.concatenate([voxels[c] for c in clusters], axis=0)
    vfeats = np.concatenate([labels[c] for c in clusters], axis=0)
    _, cs = np.unique(vfeats, return_inverse=True)
    trace = go.Scatter3d(x=voxels[:,0], y=voxels[:,1], z=voxels[:,2],
                        mode='markers',
                        marker = dict(
                            size = markersize,
                            color = cs,
                            colorscale='Viridis',
                            opacity=0.8
                        ), 
                        hovertext=vfeats)
    return [trace]

def network_topology(voxels, clusters, primaries, edges, mode='sphere'):
    """
    Network 3D topological representation
    - voxels is a list of voxel coordinates (Nx3-matrix)
    - clusters is an array of clusters, each defined as an array of voxel ids
    - primaries is a list of the primary cluster ids
    - edges is list of pairs of cluster ids arranged in two vectors (2xM-vector)
    """
    # Define the arrays of node positions (barycenter of voxels in the cluster)
    pos = np.array([voxels[c].cpu().numpy().mean(0) for c in clusters])

    # Define the node features (label, color)
    n = len(clusters)
    node_labels = ['%d (%0.1f, %0.1f, %0.1f)' % (i, pos[i,0], pos[i,1], pos[i,2]) for i in range(n)]
    
    node_colors = ['#ff7f0e' if i in primaries else '#1f77b4' for i in range(n)]
    
    # Assert if there is edges to draw
    draw_edges = bool(edges.shape[1])

    # Define the nodes and their connections
    graph_data = []
    edge_vertices = []
    if mode == 'sphere':
        # Define the node size
        logn = np.array([np.log(len(c)) for c in clusters])
        node_sizes = np.interp(logn, (logn.min(), logn.max()), (5, 50))
        
        # Define the nodes as sphere of radius proportional to the log of the cluster voxel content
        graph_data.append(go.Scatter3d(x = pos[:,0], y = pos[:,1], z = pos[:,2],
                            name = 'clusters',
                            mode = 'markers',
                            marker = dict(
                                symbol = 'circle',
                                size = node_sizes,
                                color = node_colors,
                                colorscale = 'Viridis',
                                line = dict(color='rgb(50,50,50)', width=0.5)
                            ),
                            text = node_labels,
                            hoverinfo = 'text'
                        ))

        # Define the edges center to center
        if draw_edges:
            edge_vertices = np.concatenate([[pos[i], pos[j], [None, None, None]] for i, j in zip(edges[0], edges[1])])

    elif mode == 'hull':
        # For each cluster, add the convex hull of all its voxels
        graph_data += [go.Mesh3d(alphahull =10.0,
                            name = '',
                            x = voxels[c][:,0],
                            y = voxels[c][:,1],
                            z = voxels[c][:,2],
                            color = node_colors[i],
                            opacity = 0.3,
                            text = node_labels[i],
                            hoverinfo = 'text'
                        ) for i, c in enumerate(clusters)]

        # Define the edges closest pixel to closest pixel
        import scipy as sp
        edge_vertices = []
        for i, j in zip(edges[0], edges[1]):
            vi, vj = voxels[clusters[i]], voxels[clusters[j]]
            d12 = sp.spatial.distance.cdist(vi, vj, 'euclidean')
            i1, i2 = np.unravel_index(np.argmin(d12), d12.shape)
            edge_vertices.append([vi[i1].cpu().numpy(), vj[i2].cpu().numpy(), [None, None, None]])

        if draw_edges:
            edge_vertices = np.concatenate(edge_vertices)
        
    else:
        raise ValueError
            
    # Initialize a graph that contains the edges
    if draw_edges:
        graph_data.append(go.Scatter3d(x = edge_vertices[:,0], y = edge_vertices[:,1], z = edge_vertices[:,2],
                            mode = 'lines',
                            name = 'edges',
                            line = dict(
                                color = 'rgba(50, 50, 50, 0.5)',
                                width = 1
                            ),
                            hoverinfo = 'none'
                          ))

    # Return
    return graph_data

def network_schematic(clusters, primaries, edges):
    """
    Network 2D schematic representation
    - clusters is an array of clusters, each defined as an array of voxel ids
    - primaries is a list of the primary cluster ids
    - edges is list of pairs of cluster ids arranged in two vectors (2xM-vector)
    """
    # Define the node positions (primaries on the left, secondaries on the right)
    n = len(clusters)
    pos = np.array([[1.-float(i in primaries), i] for i in range(n)])

    # Define the node features (label, size, color)
    node_labels = [str(i) for i in range(n)]
    
    logn = np.array([np.log(len(c)) for c in clusters])
    node_sizes = np.interp(logn, (logn.min(), logn.max()), (5, 50))
    
    node_colors = ['#ff7f0e' if i in primaries else '#1f77b4' for i in range(n)]

    # Define the nodes as sphere of radius proportional to the log of the cluster voxel content
    graph_data = []
    graph_data.append(go.Scatter(
                        x = pos[:,0],
                        y = pos[:,1],
                        mode = 'markers',
                        name = 'clusters',
                        marker = dict(
                            color = node_colors,
                            size = node_sizes,
                        ),
                        text = node_labels,
                        hoverinfo = 'text'
                        ))
    
    # Assert if there is edges to draw
    draw_edges = bool(edges.shape[1])

    # Initialize the edges
    if draw_edges:
        edge_vertices = np.concatenate([[pos[i], pos[j], [None, None]] for i, j in zip(edges[0], edges[1])])
        graph_data.append(go.Scatter(x = edge_vertices[:,0], y = edge_vertices[:,1],
                            mode = 'lines',
                            name = 'edges',
                            line = dict(
                                color = 'rgba(50, 50, 50, 0.5)',
                                width = 1
                            ),
                            hoverinfo = 'none'
                          ))

    return graph_data
