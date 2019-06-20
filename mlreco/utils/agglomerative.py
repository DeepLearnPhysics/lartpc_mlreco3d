import numpy as np

def find_parent(parent, i):
    """
    find parent of index i
    """
    if parent[i] == i:
        return i
    else:
        return find_parent(parent, parent[i])

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
            death[j] = fij
            # merge components
            parent[pj] = pi
            pass
        else:
            # bj <= bi
            death[i] = fij
            # merge components
            parent[pi] = pj
    return birth, death