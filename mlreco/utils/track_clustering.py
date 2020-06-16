# Utils to do track clustering
import numpy as np
import scipy
import sklearn
from scipy.spatial.distance import cdist

def track_clustering(voxels, points, method='masked_dbscan', **kwargs):
    if method == 'masked_dbscan':
        dist_mat  = cdist(points, voxels)
        dist_mask = np.all((dist_mat > kwargs['mask_radius']), axis=0)
        labels = sklearn.cluster.DBSCAN(eps=kwargs['eps'], min_samples=kwargs['min_samples']).fit(voxels).labels_
        for i in np.unique(labels):
            global_mask = labels == i
            active_mask = dist_mask & global_mask
            passive_mask = ~dist_mask & global_mask
            if np.sum(active_mask):
                res = sklearn.cluster.DBSCAN(eps=kwargs['eps'], min_samples=kwargs['min_samples']).fit(voxels[active_mask])
                labels[active_mask] = np.max(labels)+1+res.labels_
                if np.sum(passive_mask):
                    dist_mat = cdist(voxels[active_mask], voxels[passive_mask])
                    args = np.argmin(dist_mat, axis=0)
                    labels[passive_mask] = labels[active_mask][args]

        return labels

    elif method == 'closest_path':
        dist_mat = cdist(voxels, points)
        labels = sklearn.cluster.DBSCAN(eps=kwargs['eps'], min_samples=kwargs['min_samples']).fit(voxels).labels_
        for l in np.unique(labels):
            group_mask = np.where(labels == l)[0]
            point_dists = np.min(dist_mat[group_mask], axis=0)
            group_points = points[point_dists < kwargs['eps']]
            if len(group_points) > 2:
                # Build a graph on the group voxels that respect the DBSCAN distance scale
                graph_voxels = np.vstack((group_points, voxels[group_mask]))
                graph = cdist(graph_voxels, graph_voxels)
                graph *= (graph < kwargs['eps'])
                cs_graph = scipy.sparse.csr_matrix(graph)

                # Find the shortest path between each of the breaking points, idenify optimal segments
                shortest_mat, predecessors = scipy.sparse.csgraph.shortest_path(csgraph=cs_graph, directed=False, return_predecessors=True)
                mst_mat = scipy.sparse.csgraph.minimum_spanning_tree(shortest_mat[:len(group_points),:len(group_points)]).toarray()
                mst_edges = np.vstack(np.where(mst_mat > 0)).T

                # Find the closest path to each voxel, relabel
                paths = [[] for _ in range(len(mst_edges))]
                for i, e in enumerate(mst_edges):
                    k, l = e
                    paths[i].append(l)
                    while l != k:
                        l = predecessors[k,l]
                        paths[i].append(l)
                if np.all([len(p) == 2 for p in paths]):
                    labels[group_mask] = max(labels)+1
                    continue
                mindists = np.vstack([np.min(shortest_mat[len(group_points):,p[1:-1]],axis=1) for p in paths if len(p) > 2])
                sublabels = np.argmin(mindists, axis=0)
                labels[group_mask] = max(labels)+1+sublabels

        return np.unique(labels, return_inverse=True)[1]

    else:
        raise ValueError('Track clustering method not recognized:', method)
