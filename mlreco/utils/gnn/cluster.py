import numpy as np
import torch

def get_cluster_label(data, clusts):
    """
    get cluster id
    """
    if isinstance(data, torch.Tensor):
        data = data.cpu().detach().numpy()
    labels = []
    for c in clusts:
        v, cts = np.unique(data[c,5], return_counts=True)
        labels.append(v[np.argmax(cts)])
    return np.array(labels)


def get_cluster_group(data, clusts):
    """
    get cluster group
    """
    if isinstance(data, torch.Tensor):
        data = data.cpu().detach().numpy()
    labels = []
    for c in clusts:
        v, cts = np.unique(data[c,6], return_counts=True)
        labels.append(v[np.argmax(cts)])
    return np.array(labels)


def get_cluster_batch(data, clusts):
    """
    get cluster batch
    """
    if isinstance(data, torch.Tensor):
        data = data.cpu().detach().numpy()
    batch = []
    for c in clusts:
        v, cts = np.unique(data[c,3], return_counts=True)
        batch.append(v[np.argmax(cts)])
    return np.array(batch)


def get_cluster_voxels(data, clust):
    """
    return voxels in cluster
    """
    return data[clust, :3]


def get_cluster_centers(data, clusts):
    """
    get centers of clusters
    """
    centers = []
    if isinstance(data, torch.Tensor):
        data = data.cpu().detach().numpy()
    for c in clusts:
        x = get_cluster_voxels(data, c)
        centers.append(np.mean(x, axis=0))
    return np.array(centers)
    

def get_cluster_energies(data, clusts):
    """
    get energy for each cluster
    """
    energy = []
    #if isinstance(data, torch.Tensor):
    #    data = data.cpu().detach().numpy()
    for c in clusts:
        energy.append(len(c))
    return np.array(energy)


def get_cluster_dirs(data, clusts, delta=0.0):
    """
    get (N, 9) array of cluster directions
    
    Optional arguments:
        delta = orientation matrix regularization
    """
    # first make sure data is numpy array
    if isinstance(data, torch.Tensor):
        data = data.cpu().detach().numpy()
    
    feats = []
    for c in clusts:
        # get center of cluster
        x = get_cluster_voxels(data, c)
        if len(c) < 2:
            # don't waste time with computations
            # default to regularized orientation matrix, zero direction
            center = x.flatten()
            B = delta * np.eye(3)
            v0 = np.zeros(3)
            feats.append(np.concatenate((center, B.flatten(), v0)))
            continue
            
        center = np.mean(x, axis=0)
        # center data
        x = x - center
        
        # get orientation matrix
        A = x.T.dot(x)
        # get eigenvectors - convention with eigh is that eigenvalues are ascending
        w, v = np.linalg.eigh(A)
        w = w / w[2] # normalize top eigenvalue to be 1
        # orientation matrix with regularization
        B = (1-delta) * v.dot(np.diag(w)).dot(v.T) + delta*np.eye(3)
        
        feats.append(B.flatten())
        
    return np.array(feats)
    
    
def get_cluster_features(data, clusts, delta=0.0):
    """
    get features for N clusters:
    * center (N, 3) array
    * orientation (N, 9) array
    * direction (N, 3) array
    output is (N, 15) matrix
    
    Optional arguments:
        delta = orientation matrix regularization
    
    """
    # first make sure data is numpy array
    if isinstance(data, torch.Tensor):
        data = data.cpu().detach().numpy()
    # types of data
    feats = []
    for c in clusts:
        # get center of cluster
        x = get_cluster_voxels(data, c)
        if len(c) < 2:
            # don't waste time with computations
            # default to regularized orientation matrix, zero direction
            center = x.flatten()
            B = delta * np.eye(3)
            v0 = np.zeros(3)
            feats.append(np.concatenate((center, B.flatten(), v0)))
            continue
            
        center = np.mean(x, axis=0)
        # center data
        x = x - center
        
        # get orientation matrix
        A = x.T.dot(x)
        # get eigenvectors - convention with eigh is that eigenvalues are ascending
        w, v = np.linalg.eigh(A)
        dirwt = 0.0 if w[2] == 0 else 1.0 - w[1] / w[2] # weight for direction
        w = w + delta # regularization
        w = w / w[2] # normalize top eigenvalue to be 1
        # orientation matrix
        B = v.dot(np.diag(w)).dot(v.T)
        
        # get direction - look at direction of spread orthogonal to v[:,2]
        v0 = v[:,2]
        # projection of x along v0 
        x0 = x.dot(v0)
        # projection orthogonal to v0
        xp0 = x - np.outer(x0, v0)
        np0 = np.linalg.norm(xp0, axis=1)
        # spread coefficient
        sc = np.dot(x0, np0)
        if sc < 0:
            # reverse 
            v0 = -v0
        # weight direction
        v0 = dirwt*v0
        # append, center, B.flatten(), v0
        feats.append(np.concatenate((center, B.flatten(), v0, [len(x)])))
    return np.array(feats)
        
    
def form_clusters_new(data):
    """
    input dbscan image data
    returns clusters
    ASSUME:
    data is in [x,y,z, batchid, val, cid] form
    """
    if isinstance(data, torch.Tensor):
        data = data.cpu().detach().numpy()
    clusts = []
    
    for b in np.unique(data[:, 3]):
        batch_sel = data[:, 3] == b
        binds = np.where(batch_sel)[0]
        data_batch = data[binds,:]
        for c in np.unique(data_batch[:,5]):
            if c < 0:
                continue
            c_sel = data_batch[:,5] == c
            clust = np.where(c_sel)[0]
            # go back to original indices
            clusts.append(binds[clust])
    return np.array(clusts)

            
        
        
        
    
    
    
    
    
    
    
