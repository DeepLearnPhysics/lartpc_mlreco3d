import numpy as np
from sklearn.cluster import DBSCAN

def dbscan_points(voxels, epsilon = 1.01, minpts = 3):
    """
    input:
        voxels  : (N,3) array of voxel locations
        epsilon : (optional) DBSCAN radius (default = 1.99)
        minpts  : (optional) DBSACN min pts (default = 1)
    """
    index = np.arange(len(voxels))
    res=DBSCAN(eps=epsilon,
               min_samples=minpts,
               metric='euclidean'
              ).fit(voxels)
    clusters = [ index[np.where(res.labels_ == i)[0]].astype(np.int64) for i in range(np.max(res.labels_)+1) ]
    clusters_nb    = np.empty(len(clusters), dtype=object)
    clusters_nb[:] = clusters
    return clusters_nb


def dbscan_types(voxels, types, epsilon = 1.01, minpts = 3, typemin=2, typemax=5):
    """
    input:
        voxels  : (N,3) array of voxel locations
        types   : (N,) vector of voxel type
        epsilon : (optional) DBSCAN radius (default = 1.99)
        minpts  : (optional) DBSACN min pts (default = 1)
        typemin : (optional) minimum type value (default = 2 for only EM)
        typemax : (optional) maximum type value (default = 5)
    """
    clusts = []
    # loop over classes
    for c in range(typemin, typemax):
        cinds = types == c
        selection = np.where(cinds == 1)[0]
        if len(selection) == 0:
            continue
        # perform DBSCAN
        sel_vox = voxels[selection]

        res=DBSCAN(eps=epsilon,
                   min_samples=minpts,
                   metric='euclidean'
                  ).fit(sel_vox)
        cls_idx = [ selection[np.where(res.labels_ == i)[0]] for i in range(np.max(res.labels_)+1) ]
        clusts.extend(cls_idx)
    return np.array(clusts)


def dbscan_groups(voxels, groups, types, epsilon = 1.01, minpts = 3, typemin=2, typemax=5):
    """
    input:
        voxels : (N,3) array of voxel locations
        groups : (N,) vector of voxel groups
        types : (N,) vector of group types
        epsilon : (optional) DBSCAN radius (default = 1.99)
        minpts : (optional) DBSACN min pts (default = 1)
        typemin : (optional) minimum type value (default = 2 for only EM)
        typemax : (optional) maximum type value (default = 5)
    """
    clusts = []
    # loop over classes
    for c in range(len(types)):
        if types[c] < typemin or types[c] > typemax:
            continue
        cinds = groups == c
        selection = np.where(cinds == 1)[0]
        if len(selection) == 0:
            continue
        # perform DBSCAN
        sel_vox = voxels[selection]

        res=DBSCAN(eps=epsilon,
                   min_samples=minpts,
                   metric='euclidean'
                  ).fit(sel_vox)
        cls_idx = [ selection[np.where(res.labels_ == i)[0]] for i in range(np.max(res.labels_)+1) ]
        clusts.extend(cls_idx)
    return np.array(clusts)
