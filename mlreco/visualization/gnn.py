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
    draw_edges = bool(edges.shape[1]) if len(edges) == 2 else False

    # Define the nodes and their connections
    graph_data = []
    edge_vertices = []
    if mode == 'sphere':
        # Define the node size as a linear function of the amount of voxels in the cluster
        sizes = np.array([len(c) for c in clusters])
        node_sizes = sizes * 50./sizes.max()

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
                                       hoverinfo = 'text'))

        # Define the edges center to center
        if draw_edges:
            edge_vertices = np.concatenate([[pos[i], pos[j], [None, None, None]] for i, j in zip(edges[0], edges[1])])

    elif mode == 'cone':
        # Evaluate the cone parameters
        from sklearn.decomposition import PCA
        import numpy.linalg as LA
        pca = PCA()
        axes, spos, epos = np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3))
        curv = lambda vox, pid, norm :\
            np.sum([np.abs(np.dot((v-vox[pid])/LA.norm(v-vox[pid]), norm)) for i, v in enumerate(vox) if i != pid])
        for c in clusters:
            # Get the voxels corresponding to the cluster
            vox = voxels[c].numpy()

            # Get the mean and the principal axis from the PCA
            pca.fit(vox)
            axis = np.array([pca.components_[0][i] for i in range(3)])

            # Order the point along the principal axis, get the end points
            pa_vals = np.dot(vox, axis)
            pids = np.argmax(pa_vals), np.argmin(pa_vals)

            # Identify the starting point as the point with the largest curvature
            curvs = [curv(vox, pid, axis) for pid in pids]
            start_id, end_id = pids[np.argmax(curvs)], pids[np.argmin(curvs)]
            spos = np.concatenate((spos, [vox[start_id]]))
            epos = np.concatenate((epos, [vox[end_id]]))

            # Get the full length of the principal axis
            pa_dist = pa_vals[start_id]-pa_vals[end_id]

            # Append the cone parameters
            axes = np.concatenate((axes, [2.*pa_dist*axis]))

        # Compute plotly's internal vector scale to undo it...
        vector_scale = np.inf
        for i, p in enumerate(spos):
            u = axes[i]
            if i > 0:
                vector_scale = min(vector_scale, 2*LA.norm(p2-p) / (LA.norm(u2) + LA.norm(u)))
            p2 = p
            u2 = u

        # Add a graph with a cone per cluster
        graph_data.append(go.Cone(x=spos[:,0], y=spos[:,1], z=spos[:,2],
                                  u=axes[:,0], v=axes[:,1], w=axes[:,2],
                                  name = 'clusters',
                                  opacity=.5,
                                  sizeref=.5/vector_scale,
                                  showscale=False,
                                  anchor='tip'))

        # Add a graph with the starting points
        graph_data.append(go.Scatter3d(x=spos[:,0], y=spos[:,1], z=spos[:,2],
                                       name = 'nodes',
                                       mode='markers',
                                       marker = dict(
                                           symbol = 'circle',
                                           color = node_colors,
                                           size = 5,
                                           colorscale = 'Viridis'
                                       ),
                                       text = node_labels,
                                       hoverinfo = 'text'))

        # Join end points of primary cones to starting points of secondary cones
        for e in edges:
            edge_vertices = np.concatenate([[epos[i], spos[j], [None, None, None]] for i, j in zip(edges[0], edges[1])])

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
                                 hoverinfo = 'text') for i, c in enumerate(clusters)]

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

    elif mode == 'scatter':
        # Simply draw all the voxels of each cluster, using primary as color
        cids = np.full(len(voxels), -1)
        for i, c in enumerate(clusters):
            cids[c] = i

        mask = np.where(cids != -1)[0]
        colors = [node_colors[i] for i in cids[mask]]
        labels = [node_labels[i] for i in cids[mask]]

        graph_data = [go.Scatter3d(x = voxels[mask][:,0],
                                   y = voxels[mask][:,1],
                                   z = voxels[mask][:,2],
                                   mode = 'markers',
                                   marker = dict(
                                     symbol = 'circle',
                                     color = colors,
                                     size = 1
                                   ),
                                   text = labels,
                                   hoverinfo = 'text')]

        # Define the edges closest pixel to closest pixel
        if draw_edges:
            import scipy as sp
            edge_vertices = []
            for i, j in zip(edges[0], edges[1]):
                vi, vj = voxels[clusters[i]], voxels[clusters[j]]
                d12 = sp.spatial.distance.cdist(vi, vj, 'euclidean')
                i1, i2 = np.unravel_index(np.argmin(d12), d12.shape)
                edge_vertices.append([vi[i1].cpu().numpy(), vj[i2].cpu().numpy(), [None, None, None]])

            edge_vertices = np.concatenate(edge_vertices)

    else:
        raise ValueError("Network topology mode not supported")

    # Initialize a graph that contains the edges
    if draw_edges:
        graph_data.append(go.Scatter3d(x = edge_vertices[:,0], y = edge_vertices[:,1], z = edge_vertices[:,2],
                                       mode = 'lines',
                                       name = 'edges',
                                       line = dict(
                                           color = 'rgba(50, 50, 50, 0.5)',
                                           width = 5
                                       ),
                                       hoverinfo = 'none'))

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

    sizes = np.array([len(c) for c in clusters])
    node_sizes = sizes * 50./sizes.max()

    node_colors = ['#ff7f0e' if i in primaries else '#1f77b4' for i in range(n)]

    # Define the nodes as sphere of radius proportional to the log of the cluster voxel content
    graph_data = []
    graph_data.append(go.Scatter(x = pos[:,0], y = pos[:,1],
                                 mode = 'markers',
                                 name = 'clusters',
                                 marker = dict(
                                     color = node_colors,
                                     size = node_sizes
                                 ),
                                 text = node_labels,
                                 hoverinfo = 'text'))

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
                                     hoverinfo = 'none'))

    return graph_data
