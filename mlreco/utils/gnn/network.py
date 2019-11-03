# defines incidence matrix for primaries
import numpy as np
import torch
from scipy.spatial import Delaunay
from scipy.sparse.csgraph import minimum_spanning_tree

def primary_bipartite_incidence(batches, primaries, dist=None, max_dist=float('inf'), device=None, cuda=True):
    """
    incidence matrix of bipartite graph between primary clusters and non-primary clusters
    """
    others = np.where([ not(x in primaries) for x in np.arange(len(batches))])[0]
    ret = torch.tensor([[i, j] for i in primaries for j in others if batches[i] == batches[j]], dtype=torch.long, requires_grad=False).t().contiguous().reshape(2,-1)

    # If requested, remove the edges above a certain length threshold
    if max_dist < float('inf'):
        dists = np.array([dist[i, j] for i in primaries for j in others if batches[i] == batches[j]])
        ret = ret[:,np.where(dists < max_dist)[0]]

    if not device is None:
        ret = ret.to(device)
    elif cuda:
        ret = ret.cuda()
    return ret

def complete_graph(batches, dist=None, max_dist=float('inf'), device=None, cuda=True):
    """
    incidence matrix of bipartite graph between primary clusters and non-primary clusters
    """
    ret = torch.tensor([[i, j] for i in np.arange(len(batches)) for j in np.arange(len(batches)) if (batches[i] == batches[j] and j > i)], dtype=torch.long, requires_grad=False).t().contiguous().reshape(2,-1)

    # If requested, remove the edges above a certain length threshold
    if max_dist < float('inf'):
        dists = np.array([dist[i, j] for i in np.arange(len(batches)) for j in np.arange(len(batches)) if (batches[i] == batches[j] and j > i)])
        ret = ret[:,np.where(dists < max_dist)[0]]

    if not device is None:
        ret = ret.to(device)
    elif cuda:
        ret = ret.cuda()
    return ret

def delaunay_graph(batches, centers, max_dist=float('inf'), device=None, cuda=None):
    """
    incidence matrix of graph between clusters that are connected by a distance-based Delaunay Graph
    """
    # For each batch, find the list of edges, append it
    ret = np.zeros(shape=(0, 2), dtype=np.int32)
    for i in np.unique(batches):
        where = np.where(batches == i)[0]
        tri = Delaunay(centers[where])
        edges = np.unique(np.array([[where[i], where[j]] for s in tri.simplices for i in s for j in s if i < j]), axis=0)
        ret = np.vstack((ret, edges))

    # If requested, remove the edges above a certain length threshold
    if max_dist < float('inf'):
        dists = np.linalg.norm(centers[ret[:,1]]-centers[ret[:,0]], axis=1)
        ret = ret[dists < max_dist]

    ret = torch.tensor(ret.transpose())
    if not device is None:
        ret = ret.to(device)
    elif cuda:
        ret = ret.cuda()
    return ret

def mst_graph(batches, dist, max_dist=float('inf'), device=None, cuda=None):
    """
    incidence matrix of graph between clusters that are connected by a distance-based Minimum Spanning Tree
    """
    mst_mat = minimum_spanning_tree(dist).toarray().astype(float)
    ret = torch.tensor(np.unravel_index(np.where(mst_mat.flatten() > 0.)[0], mst_mat.shape))

    # If requested, remove the edges above a certain length threshold
    if max_dist < float('inf'):
        dists = mst_mat.flatten()[np.where(mst_mat.flatten() > 0.)[0]]
        ret = ret[:,np.where(dists < max_dist)[0]]

    if not device is None:
        ret = ret.to(device)
    elif cuda:
        ret = ret.cuda()
    return ret

