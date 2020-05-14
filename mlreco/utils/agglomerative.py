import numpy as np
from mlreco.utils.gnn.compton import filter_compton
from mlreco.utils.gnn.cluster import form_clusters
from mlreco.utils.gnn.cluster import get_cluster_features

def find_parent(parent, i):
    """
    find parent of index i
    """
    if i != parent[i]:
        parent[i] = find_parent(parent, parent[i])
    return parent[i]


def merge_diagram(f, edges):
    """
    f: np array of function values on voxels
    edges: np array of edges connecting voxels
    
    computes 0-dimensional persistence pairs using lower-star filtration
    """
    n = len(f)
    fe = np.max(f[edges], axis=1) # function value on edges
    birth = f.copy()
    death = np.empty(n)
    death.fill(np.inf) # fill with deaths
    sortperm = np.argsort(fe)
    parent = np.arange(n)
    elist = []
    for ei in sortperm:
        # get nodes
        i, j = edges[ei]
        fij = fe[ei]
        pi = find_parent(parent, i)
        pj = find_parent(parent, j)
        if pi == pj:
            # nothing to do
            continue
        bi = birth[pi] # birth of component containing i
        bj = birth[pj] # birth of component containing j
        if bi < bj:
            # component with j merges
            death[pj] = fij
            # merge components
            parent[pj] = pi
            elist.append([i,j])
        else:
            # bj <= bi
            death[pi] = fij
            # merge components
            parent[pi] = pj
            elist.append([i,j])
    return birth, death, elist

def get_lifetimes(data):
    """
    data: np array of DBSCAN-parsed data with shape (N, 5)
    returns: np array of shape (N,) with the label corresponding to the lifetime of the voxel
        lifetime will be infinity if a voxel is outside a cluster or in a compton scatter
    """
    all_lifetimes = np.inf * np.ones(len(data))
    clusts = form_clusters(data)

    # remove compton clusters
    selection = filter_compton(clusts)
    clusts = clusts[selection]

    non_compton = np.concatenate(clusts)
    cluster_features = get_cluster_features(data, clusts)
    for i in range(len(clusts)):
        clust = clusts[i]
        mean = cluster_features[:, :3][i]
        direction = cluster_features[:, -3:][i]
        
        coords = data[clust][:, :3]
        f = np.dot(coords - mean, direction)
        box_dim = 1
        edges = []
        for i in range(len(coords)):
            point = coords[i][:3]
            x, y, z = point
            region = coords
            indices = np.arange(len(coords))
            indices = indices[np.searchsorted(region[:, 2], z - box_dim):]
            region = coords[indices]
            indices = indices[:np.searchsorted(region[:, 2], z + box_dim, side='right')]
            region = coords[indices]
            indices = indices[np.where((region[:, 1] >= y - box_dim) & (region[:, 1] <= y + box_dim) & (region[:, 0] >= x - box_dim) & (region[:, 0] <= x + box_dim))]
            region = coords[indices]
            for j in indices:
                if i != j:
                    entry = sorted((i, j))
                    if entry not in edges:
                        edges.append(entry)
        edges = np.array(edges)
        
        births, deaths, edge_list = merge_diagram(f, edges)
        lifetimes = deaths - births
        print(lifetimes)
        all_lifetimes[clust] = lifetimes
        print(all_lifetimes[clust])
    return all_lifetimes
