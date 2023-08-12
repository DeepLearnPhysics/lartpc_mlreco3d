import numpy as np
import scipy
import sklearn
from scipy.spatial.distance import cdist

def track_clustering(voxels, points, method='masked_dbscan', **kwargs):
    if method == 'masked_dbscan':
        pair_mat  = cdist(points, voxels, metric=kwargs['metric'])
        dist_mask = np.all((pair_mat > kwargs['mask_radius']), axis=0)
        labels = sklearn.cluster.DBSCAN(eps=kwargs['eps'],
                                        min_samples=kwargs['min_samples'],
                                        metric=kwargs['metric']).fit(voxels).labels_
        for i in np.unique(labels):
            global_mask  = labels == i
            active_mask  = dist_mask & global_mask
            passive_mask = ~dist_mask & global_mask
            if np.sum(active_mask):
                res = sklearn.cluster.DBSCAN(eps=kwargs['eps'],
                                             min_samples=kwargs['min_samples'],
                                             metric=kwargs['metric']).fit(voxels[active_mask])
                labels[active_mask] = np.max(labels)+1+res.labels_
                if np.sum(passive_mask):
                    dist_mat = cdist(voxels[active_mask], voxels[passive_mask], metric=kwargs['metric'])
                    args = np.argmin(dist_mat, axis=0)
                    labels[passive_mask] = labels[active_mask][args]

        return labels

    elif method == 'closest_path':
        pair_mat = cdist(voxels, points, metric=kwargs['metric'])
        labels   = sklearn.cluster.DBSCAN(eps=kwargs['eps'],
                                          min_samples=kwargs['min_samples'],
                                          metric=kwargs['metric']).fit(voxels).labels_
        for l in np.unique(labels):
            group_mask  = labels == l
            point_mask  = np.min(pair_mat[group_mask], axis=0) < kwargs['eps']
            point_ids   = np.unique(np.argmin(pair_mat[np.ix_(group_mask, point_mask)], axis=0))
            if len(point_ids) > 2:
                # Build a graph on the group voxels that respect the DBSCAN distance scale
                dist_mat  = cdist(voxels[group_mask], voxels[group_mask], metric=kwargs['metric'])
                graph     = dist_mat * (dist_mat < kwargs['eps'])
                cs_graph  = scipy.sparse.csr_matrix(graph)

                # Find the shortest between each of the breaking points, identify segments that minimize absolute excursion
                graph_mat, predecessors = scipy.sparse.csgraph.shortest_path(csgraph=cs_graph, directed=False, return_predecessors=True)
                break_ix  = np.ix_(point_ids, point_ids)
                chord_mat = dist_mat[break_ix]
                mst_mat   = scipy.sparse.csgraph.minimum_spanning_tree(graph_mat[break_ix]-chord_mat+1e-6).toarray()
                mst_edges = np.vstack(np.where(mst_mat > 0)).T

                # Construct graph paths along the tree
                paths = [[] for _ in range(len(mst_edges))]
                for i, e in enumerate(mst_edges):
                    k, l = point_ids[e]
                    paths[i].append(l)
                    while l != k:
                        l = predecessors[k,l]
                        paths[i].append(l)

                # Find the path closest to each of the voxels in the group. If a path does not improve reachability, remove
                mindists  = np.vstack([np.min(graph_mat[:,p],axis=1) for p in paths])
                mst_tort  = mst_mat[mst_mat > 0]/chord_mat[mst_mat > 0] # tau - 1
                tort_ord  = np.argsort(-mst_tort)
                least_r   = np.max(np.min(mindists, axis=0)) # Least reachable point distance
                path_mask = np.ones(len(mst_tort), dtype=np.bool)
                for i in range(len(mst_tort)):
                    if np.sum(path_mask) == 1: break
                    path_mask[tort_ord[i]] = False
                    reach = np.max(np.min(mindists[path_mask], axis=0))
                    if reach > least_r:
                        path_mask[tort_ord[i]] = True

                # Associate voxels with closest remaining path
                mindists  = np.vstack([np.min(graph_mat[:,p[min(1,len(p)-2):max(len(p)-1,2)]],axis=1) for i, p in enumerate(paths) if path_mask[i]])
                sublabels = np.argmin(mindists, axis=0)
                labels[group_mask] = max(labels)+1+sublabels

        return np.unique(labels, return_inverse=True)[1]

    else:
        raise ValueError('Track clustering method not recognized:', method)
