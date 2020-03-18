import numpy as np
import plotly.graph_objs as go

def scatter_clusters(voxels, labels, clusts, markersize=5, colorscale='Viridis'):
    """
    Scatter plot of cluster voxels colored by cluster order

    Args:
        voxels (np.ndarray)  : (N,3) List of voxel coordinate
        labels (np.ndarray)  : (N) List of voxels labels
        clusts ([np.ndarray]): (C) List of arrays of voxel IDs in each cluster
        markersize (int)     : Size of the voxel markersize
        colorscale (str)     : Plotly color scale name
    Returns:
        [plotly.graph_objs.Scatter3d]: Scatter plot
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
                             colorscale = colorscale,
                             opacity = 0.8
                         ),
                         hovertext=vfeats)
    return [trace]

def network_topology(voxels, clusters, edge_index=[], clust_labels=[], edge_labels=[], mode='scatter', markersize=3, linewidth=2, colorscale='Inferno'):
    """
    Network 3D topological representation

    Args:
        voxels (np.ndarray)      : (N,3) List of voxel coordinate
        clusts ([np.ndarray])    : (C) List of arrays of voxel IDs in each cluster
        edge_index (np.ndarray)  : (E,2) List of connections
        clust_labels (np.ndarray): (C) Cluster labels
        edge_labels (np.ndarray) : (E) Edge labels
        mode (str)               : Draw mode ('sphere', 'cone', 'hull', 'scatter')
        markersize (int)         : Size of the voxel markersize in pixels
        linewidth (int)          : Width of the edge lines in pixels
        colorscale (str)         : Plotly color scale name
    Returns:
        [plotly.graph_objs.Scatter3d]: (2) 3D Scatter plots of [nodes, edges]
    """
    # Define the arrays of node positions (barycenter of voxels in the cluster)
    pos = np.array([voxels[c].mean(0) for c in clusters])

    # Define the node features (label, color)
    n = len(clusters)
    if not len(clust_labels): clust_labels = np.zeros(n)
    node_labels = ['Cluster ID: %d<br>Cluster label: %0.3f<br>Centroid: (%0.1f, %0.1f, %0.1f)' % (i, clust_labels[i], pos[i,0], pos[i,1], pos[i,2]) for i in range(n)]

    # Assert if there is edges to draw
    draw_edges = bool(len(edge_index))

    # Define the nodes and their connections
    graph_data = []
    edge_vertices = []
    if mode == 'sphere':
        # Define the node size as a linear function of the amount of voxels in the cluster
        sizes = np.array([len(c) for c in clusters])
        node_sizes = sizes * 50./sizes.max()

        # Define the nodes as sphere of radius proportional to the log of the cluster voxel content
        graph_data.append(go.Scatter3d(x = pos[:,0], y = pos[:,1], z = pos[:,2],
                                       name = 'Graph nodes',
                                       mode = 'markers',
                                       marker = dict(
                                           symbol = 'circle',
                                           size = node_sizes,
                                           color = clust_labels,
                                           colorscale = colorscale,
                                           line = dict(color='rgb(50,50,50)', width=0.5)
                                       ),
                                       text = node_labels,
                                       hoverinfo = 'text'))

        # Define the edges center to center
        if draw_edges:
            edge_vertices = np.concatenate([[pos[i], pos[j], [None, None, None]] for i, j in edge_index])

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
            vox = voxels[c]

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
                                  name = 'Graph node cones',
                                  opacity=.5,
                                  sizeref=.5/vector_scale,
                                  showscale=False,
                                  anchor='tip'))

        # Add a graph with the starting points
        graph_data.append(go.Scatter3d(x=spos[:,0], y=spos[:,1], z=spos[:,2],
                                       name = 'Graph node starts',
                                       mode='markers',
                                       marker = dict(
                                           symbol = 'circle',
                                           color = clust_labels,
                                           size = 5,
                                           colorscale = colorscale
                                       ),
                                       text = node_labels,
                                       hoverinfo = 'text'))

        # Join end points of primary cones to starting points of secondary cones
        for e in edge_index:
            edge_vertices = np.concatenate([[epos[i], spos[j], [None, None, None]] for i, j in edge_index])

    elif mode == 'hull':
        # For each cluster, add the convex hull of all its voxels
        graph_data += [go.Mesh3d(alphahull =10.0,
                                 name = 'Graph nodes',
                                 x = voxels[c][:,0],
                                 y = voxels[c][:,1],
                                 z = voxels[c][:,2],
                                 color = clust_labels[i],
                                 opacity = 0.3,
                                 text = node_labels[i],
                                 hoverinfo = 'text') for i, c in enumerate(clusters)]

        # Define the edges closest pixel to closest pixel
        import scipy as sp
        edge_vertices = []
        for i, j in edge_index:
            vi, vj = voxels[clusters[i]], voxels[clusters[j]]
            d12 = sp.spatial.distance.cdist(vi, vj, 'euclidean')
            i1, i2 = np.unravel_index(np.argmin(d12), d12.shape)
            edge_vertices.append([vi[i1], vj[i2], [None, None, None]])

        if draw_edges:
            edge_vertices = np.concatenate(edge_vertices)

    elif mode == 'scatter':
        # Simply draw all the voxels of each cluster, using labels as color
        cids = np.full(len(voxels), -1)
        for i, c in enumerate(clusters):
            cids[c] = i

        mask = np.where(cids != -1)[0]
        colors = [clust_labels[i] for i in cids[mask]]
        node_labels = [node_labels[i] for i in cids[mask]]

        graph_data = [go.Scatter3d(x = voxels[mask][:,0],
                                   y = voxels[mask][:,1],
                                   z = voxels[mask][:,2],
                                   mode = 'markers',
                                   name = 'Graph nodes',
                                   marker = dict(
                                     symbol = 'circle',
                                     color = colors,
                                     colorscale = colorscale,
                                     size = markersize
                                   ),
                                   text = node_labels,
                                   hoverinfo = 'text')]

        # Define the edges closest pixel to closest pixel
        if draw_edges:
            import scipy as sp
            edge_vertices = []
            for i, j in edge_index:
                vi, vj = voxels[clusters[i]], voxels[clusters[j]]
                d12 = sp.spatial.distance.cdist(vi, vj, 'euclidean')
                i1, i2 = np.unravel_index(np.argmin(d12), d12.shape)
                edge_vertices.append([vi[i1], vj[i2], [None, None, None]])

            edge_vertices = np.concatenate(edge_vertices)

    else:
        raise ValueError("Network topology mode not supported")

    # Initialize a graph that contains the edges
    if draw_edges:
        if not len(edge_labels): edge_labels = np.zeros(len(edge_index))
        edge_colors = np.concatenate([[edge_labels[i]]*3 for i in range(len(edge_index))])
        graph_data.append(go.Scatter3d(x = edge_vertices[:,0], y = edge_vertices[:,1], z = edge_vertices[:,2],
                                       mode = 'lines',
                                       name = 'Graph edges',
                                       line = dict(
                                           color = edge_colors,
                                           width = linewidth,
                                           colorscale = 'Blues'
                                       ),
                                       hoverinfo = 'none'))

    # Return
    return graph_data

def network_schematic(clusters, edge_index, clust_labels=[], edge_labels=[], linewidth=1, colorscale='Inferno'):
    """
    Network 2D schematic representation

    Args:
        clusts ([np.ndarray])    : (C) List of arrays of voxel IDs in each cluster
        edge_index (np.ndarray)  : (E,2) List of connections
        clust_labels (np.ndarray): (C) Node labels
        edge_labels (np.ndarray) : (E) Edge labels
        linewidth (int)          : Width of the edge lines in pixels
        colorscale (str)         : Plotly color scale name
    Returns:
        [plotly.graph_objs.Scatter]: (2) Scatter plots of [nodes, edges]
    """
    # Get the cluster sizes (will determine the node size)
    sizes = np.array([len(c) for c in clusters])
    node_sizes = sizes * 100./sizes.max()

    # Define the node features (label, color)
    n = len(clusters)
    if not len(clust_labels): clust_labels = np.zeros(n)
    node_labels = ['Cluster ID: %d<br>Cluster label: %0.3f<br>Cluster size: %d' % (i, clust_labels[i], sizes[i]) for i in range(n)]

    # Define the node positions (primaries on the left, secondaries on the right)
    pos = np.array([[l, i] for i, l in enumerate(clust_labels)])

    # Define the nodes as sphere of radius proportional to the log of the cluster voxel content
    graph_data = []
    graph_data.append(go.Scatter(x = pos[:,0], y = pos[:,1],
                                 mode = 'markers',
                                 name = 'Graph nodes',
                                 marker = dict(
                                    color = clust_labels,
                                    size = node_sizes,
                                    colorscale = colorscale,
                                    reversescale = True
                                 ),
                                 text = node_labels,
                                 hoverinfo = 'text'))

    # Assert if there is edges to draw
    draw_edges = bool(len(edge_index))

    # Initialize the edges
    if draw_edges:
        edge_vertices = np.concatenate([[pos[i], pos[j], [None, None]] for i, j in edge_index])
        #if not len(edge_labels): edge_labels = np.zeros(len(edge_index))
        #edge_colors = np.concatenate([[edge_labels[i]]*3 for i in range(len(edge_index))])
        graph_data.append(go.Scatter(x = edge_vertices[:,0], y = edge_vertices[:,1],
                                     mode = 'lines',
                                     name = 'Graph edges',
                                     line = dict(
                                        color = 'gray', # Cannot use multiple colors...
                                        width = linewidth
                                        #colorscale = 'Greys'
                                     ),
                                     hoverinfo = 'none'))

    return graph_data
