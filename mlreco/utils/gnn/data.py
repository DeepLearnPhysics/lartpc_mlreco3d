# creates inputs to GNN networks
import torch
from .cluster import get_cluster_centers, get_cluster_voxels, get_cluster_features, get_cluster_energies
import numpy as np
import scipy as sp


# TODO: add vertex orientation information


def cluster_vtx_features_old(data, cs, cuda=True):
    """
    Cluster vertex features - center
    returned as a pytorch tensor of size (n_clusts, 3)
    optional flag to put features on gpu
    """
    f = torch.tensor(np.concatenate((get_cluster_centers(data, cs), (get_cluster_energies(data, cs)).reshape(-1,1)), axis=1), dtype=torch.float, requires_grad=False)
    if cuda:
        f = f.cuda()
    return f

def cluster_vtx_features(data, cs, cuda=True):
    """
    Cluster vertex features - center, orientation, and direction
    returned as pytorch tensor of size (n_clusts, 15)
    optional flag to put features on gpu
    """
    f = torch.tensor(get_cluster_features(data, cs), dtype=torch.float, requires_grad=False)
    if cuda:
        f = f.cuda()
    return f
    


def cluster_edge_feature(data, c1, c2):
    """
    feature that includes closest points, displacement, and length of edge
    """
    x1 = get_cluster_voxels(data, c1)
    x2 = get_cluster_voxels(data, c2)
    d12 = sp.spatial.distance.cdist(x1, x2,'euclidean')
    imin = np.argmin(d12)
    i1, i2 = np.unravel_index(imin, d12.shape)
    v1 = x1[i1,:] # closest point in c1
    v2 = x2[i2,:] # closest point in c2
    out = np.append(v1, v2)
    disp = v1 - v2 # displacement
    out = np.append(out, disp)
    lend = np.linalg.norm(disp) # length of displacement
    out = np.append(out, lend)
    return out


def cluster_edge_features(data, clusts, edge_index, cuda=True):
    """
    Cluster edge features
    returned as a pytorch tensor of size (n_edges, 10)
    """
    if isinstance(data, torch.Tensor):
        data = data.cpu().detach().numpy()
    e = torch.tensor([cluster_edge_feature(data, clusts[edge_index[0,k]], clusts[edge_index[1,k]]) for k in range(edge_index.shape[1])], dtype=torch.float, requires_grad=False)
    if cuda:
        e = e.cuda()
    return e


def edge_assignment(edge_index, batches, groups, cuda=True):
    """
    edge assignment as same group/different group
    """
    edge_assn = torch.tensor([np.logical_and(
        batches[edge_index[0,k]] == batches[edge_index[1,k]],
        groups[edge_index[0,k]] == groups[edge_index[1,k]]) for k in range(edge_index.shape[1])], 
                             dtype=torch.float, requires_grad=False)
    if cuda:
        edge_assn = edge_assn.cuda()
    return edge_assn
    
    
    
    
  
    
