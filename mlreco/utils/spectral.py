# Some Utilities for Graph Laplacians
# for linear system solves should use Krylov methods - probably Minres, with preconditioner
# sparse matrices should be scipy sparse
# Should also have a linear operator constructed for direction laplacian
# TODO: weighted versions as well


import numpy as np

def degrees(edges):
    """
    return degrees of nodes
    ASSUME: nodes are numbered sequentially starting at 0.
    ASSUME: every node has at least 1 edge
    """
    # we start knowing number of edges
    m = len(edges)
    degdict = {}
    for e in edges:
        for k in e:
            if k in degdict:
                degdict[k] += 1
            else:
                degdict[k] = 1
    # finally know number of nodes
    n = len(degdict)
    degs = np.empty(n, dtype=np.int)
    for i, ct in degdict.items():
        degs[i] = ct
    return degs


def weighted_degrees(edges, weights):
    """
    return weighted degrees of nodes
    """
    pass



def incidence_dense(edges):
    """
    form incidence matrix B of graph
    as dense numpy array
    edges[k] is edge from i to j
    B[i,k] = -1
    B[j,k] = +1
            0 otherwise
    """
    pass


def weighted_incidence_dense(edges, weights):
    pass


def incidence_sparse(edges):
    """
    form incidence matrix B of graph
    as sparse scipy array
    edges[k] is edge from i to j
    B[i,k] = -1
    B[j,k] = +1
            0 otherwise
    """
    pass


def weighted_incidence_sparse(edges, weights):
    pass








def laplacian_dense(edges):
    """
    form graph laplacian given a list of edges
    edges[k] = [i,j] is edge from i to j
    """
    pass


def laplacian_sparse(edges):
    pass


def direction_adjacency_dense(edges, frames, dtype=np.double):
    """
    form adjacency matrix for direction laplacian as dense numpy array
    Block sparse in same pattern as standard laplacian
    L[i,j] = deg(i) kron(id, id) if i == j
    L[i,j] = -kron(frame, frame) if i != j
    
    ASSUME: edges are unique
    """
    # get degrees and number of nodes
    degs = degrees(edges)
    n = len(degs)
    # get dimension of frames
    k = frames[0].shape[0]
    # size of direction_laplacian
    k2 = k * k
    nD = n * k2
    
    AD = np.zeros((nD, nD), dtype=dtype)
    # iterate over edges
    for k, e in enumerate(edges):
        i,j = e
        AD[(k2*i):(k2*(i+1)), (k2*j):(k2*(j+1))] = -np.kron(frames[k], frames[k])
        AD[(k2*j):(k2*(j+1)), (k2*i):(k2*(i+1))] = -np.kron(frames[k], frames[k])
    return AD


def direction_laplacian_dense(edges, frames, dtype=np.double):
    """
    form direction laplacian as dense numpy array
    Block sparse in same pattern as standard laplacian
    L[i,j] = deg(i) kron(id, id) if i == j
    L[i,j] = -kron(frame, frame) if i != j
    
    ASSUME: edges are unique
    """
    # get degrees and number of nodes
    degs = degrees(edges)
    n = len(degs)
    # get dimension of frames
    k = frames[0].shape[0]
    # size of direction_laplacian
    k2 = k * k
    nD = n * k2
    
    LD = np.zeros(nD, nD, dtype=dtype)
    # set diagonal
    for i in range(n):
        LD[(k2*i):(k2*(i+1)), (k2*i):(k2*(i+1))] = degs[i] * np.eye(k2)
    # iterate over edges
    for k, e in enumerate(edges):
        i,j = e
        LD[(k2*i):(k2*(i+1)), (k2*j):(k2*(j+1))] = -frames[k]
        LD[(k2*j):(k2*(j+1)), (k2*i):(k2*(i+1))] = -frames[k]
    return LD


def direction_laplacian_dense2(edges, frames, dtype=np.double):
    """
    form direction laplacian as dense numpy array
    different degree matrix definition than above
    Block sparse in same pattern as standard laplacian
    L[i,j] = deg(i) kron(id, id) if i == j
    L[i,j] = -kron(frame, frame) if i != j
    
    ASSUME: edges are unique
    """
    AD = direction_adjacency_dense(edges, frames, dtype=dtype)
    nD = AD.shape[0]
    v = np.ones(nD, dtype=dtype)
    degsD = AD.dot(v) # note that this need not be positive
    LD = np.diag(degsD) - AD
    return LD


def direction_laplacian_dense3(edges, frames, dtype=np.double):
    """
    form direction laplacian as dense numpy array
    different degree matrix definition
    Block sparse in same pattern as standard laplacian
    L[i,j] = deg(i) kron(id, id) if i == j
    L[i,j] = -kron(frame, frame) if i != j
    
    ASSUME: edges are unique
    """
    AD = direction_adjacency_dense(edges, frames, dtype=dtype)
    nD = AD.shape[0]
    v = np.ones(nD, dtype=dtype)
    degsD = np.abs(AD).dot(v) # absolute value degree
    LD = np.diag(degsD) - AD
    return LD


def weighted_direction_laplacian(edges, frames, weights):
    pass



def direction_laplacian_sparse(edges, frames):
    """
    Scipy sparse matrix format
    """
    pass


def direction_laplacian_lop(edges, frames):
    """
    Scipy Linear Operator
    """
    pass