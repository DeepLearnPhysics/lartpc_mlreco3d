# Defines cluster formation and feature extraction
import numpy as np
import torch

def form_clusters(data, min_size=-1):
    """
    Function that returns a list of of arrays of voxel IDs
    that make up each of the clusters in the input tensor.

    Args:
        data (np.ndarray): (N,8) [x, y, z, batchid, value, id, groupid, shape]
        min_size (int)   : Minimal cluster size
    Returns:
        [np.ndarray]: (C) List of arrays of voxel IDs in each cluster
    """
    clusts = []
    for b in data[:, 3].unique():
        binds = torch.nonzero(data[:, 3] == b).flatten()
        for c in data[binds,5].unique():
            # Skip if the cluster ID is -1 (not defined)
            if c < 0:
                continue
            clust = torch.nonzero(data[binds,5] == c).flatten()
            if len(clust) < min_size:
                continue
            clusts.append(binds[clust])

    return clusts


def reform_clusters(data, clust_ids, batch_ids):
    """
    Function that returns a list of of arrays of voxel IDs
    that make up the requested clusters.

    Args:
        data (np.ndarray)     : (N,8) [x, y, z, batchid, value, id, groupid, shape]
        clust_ids (np.ndarray): (C) List of cluster ids
        batch_ids (np.ndarray): (C) List of batch ids
    Returns:
        [np.ndarray]: (C) List of arrays of voxel IDs in each cluster
    """
    return np.array([np.where((data[:,3] == batch_ids[j]) & (data[:,5] == clust_ids[j]))[0] for j in range(len(batch_ids))])


def get_cluster_batch(data, clusts):
    """
    Function that returns the batch ID of each cluster.
    This should be unique for each clustert, assert that it is.

    Args:
        data (np.ndarray)    : (N,8) [x, y, z, batchid, value, id, groupid, shape]
        clusts ([np.ndarray]): (C) List of arrays of voxel IDs in each cluster
    Returns:
        np.ndarray: (C) List of batch IDs
    """
    labels = []
    for c in clusts:
        assert len(data[c,3].unique()) == 1
        labels.append(int(data[c[0],3].item()))

    return np.array(labels)


def get_cluster_label(data, clusts):
    """
    Function that returns the cluster label of each cluster.
    This should be unique for each clustert, assert that it is.

    Args:
        data (np.ndarray)    : (N,8) [x, y, z, batchid, value, id, groupid, shape]
        clusts ([np.ndarray]): (C) List of arrays of voxel IDs in each cluster
    Returns:
        np.ndarray: (C) List of cluster IDs
    """
    labels = []
    for c in clusts:
        v, cts = data[c,5].unique(return_counts=True)
        labels.append(int(v[cts.argmax()].item()))

    return np.array(labels)


def get_cluster_group(data, clusts):
    """
    Function that returns the group label of each cluster.
    This does not have to be unique, depending on the cluster
    formation method. Use majority vote.

    Args:
        data (np.ndarray)    : (N,8) [x, y, z, batchid, value, id, groupid, shape]
        clusts ([np.ndarray]): (C) List of arrays of voxel IDs in each cluster
    Returns:
        np.ndarray: (C) List of group IDs
    """
    labels = []
    for c in clusts:
        v, cts = data[c,6].unique(return_counts=True)
        labels.append(int(v[cts.argmax()].item()))

    return np.array(labels)


def get_cluster_primary(clust_ids, group_ids):
    """
    Function that returns the group label of each cluster.
    This function assumes that if clust_id == group_id,
    the cluster is a primary.

    Args:
        clust_ids (np.ndarray): (C) List of cluster ids
        group_ids (np.ndarray): (C) List of cluster group ids
    Returns:
        np.ndarray: (P) List of primary cluster ids
    """
    return np.where(clust_ids == group_ids)[0]


def get_cluster_voxels(data, clust):
    """
    Function that returns the voxel coordinates associated
    with the listed voxel IDs.

    Args:
        data (np.ndarray) : (N,8) [x, y, z, batchid, value, id, groupid, shape]
        clust (np.ndarray): (M) Array of voxel IDs in the cluster
    Returns:
        np.ndarray: (Mx3) tensor of voxel coordinates
    """
    return data[clust, :3]


def get_cluster_centers(data, clusts):
    """
    Function that returns the coordinate of the centroid
    associated with the listed clusters.

    Args:
        data (np.ndarray)    : (N,8) [x, y, z, batchid, value, id, groupid, shape]
        clusts ([np.ndarray]): (C) List of arrays of voxel IDs in each cluster
    Returns:
        np.ndarray: (C,3) tensor of cluster centers
    """
    centers = []
    for c in clusts:
        x = get_cluster_voxels(data, c)
        centers.append(np.mean(x, axis=0))
    return np.vstack(centers)


def get_cluster_sizes(data, clusts):
    """
    Function that returns the sizes of
    each of the listed clusters.

    Args:
        data (np.ndarray)    : (N,5) [x, y, z, batchid, value]
        clusts ([np.ndarray]): (C) List of arrays of voxel IDs in each cluster
    Returns:
        np.ndarray: (C) List of cluster sizes
    """
    return np.array([len(c) for c in clusts])


def get_cluster_energies(data, clusts):
    """
    Function that returns the energies deposited by
    each of the listed clusters.

    Args:
        data (np.ndarray)    : (N,5) [x, y, z, batchid, value]
        clusts ([np.ndarray]): (C) List of arrays of voxel IDs in each cluster
    Returns:
        np.ndarray: (C) List of cluster energies
    """
    return np.array([np.sum(data[c,4]) for c in clusts])


def get_cluster_dirs(voxels, clusts, delta=0.0):
    """
    Function that returns the direction of the listed clusters,
    expressed as its normalized covariance matrix.

    Args:
        voxels (np.ndarray)  : (N,3) Voxel coordinates [x, y, z]
        clusts ([np.ndarray]): (C) List of arrays of voxel IDs in each cluster
        delta (float)        : Orientation matrix regularization
    Returns:
        np.ndarray: (C,9) Tensor of cluster directions
    """
    dirs = []
    for c in clusts:
        # Get list of voxels in the cluster
        x = get_cluster_voxels(voxels, c)

        # Handle size 1 clusters seperately
        if len(c) < 2:
            # Don't waste time with computations, default to regularized
            # orientation matrix
            B = delta * np.eye(3)
            dirs.append(B.flatten())
            continue

        # Center data
        center = np.mean(x, axis=0)
        x = x - center

        # Get orientation matrix
        A = x.T.dot(x)

        # Get eigenvectors - convention with eigh is that eigenvalues are ascending
        w, v = np.linalg.eigh(A)
        w = w + delta
        w = w / w[2]

        # Orientation matrix with regularization
        B = (1.-delta) * v.dot(np.diag(w)).dot(v.T) + delta * np.eye(3)

        # Append (dirs)
        dirs.append(B.flatten())

    return np.vstack(dirs)


def get_cluster_features(voxels, clusts, delta=0.0):
    """
    Function that returns the an array of 16 features for
    each of the clusters in the provided list.

    Args:
        voxels (np.ndarray)  : (N,3) Voxel coordinates [x, y, z]
        clusts ([np.ndarray]): (C) List of arrays of voxel IDs in each cluster
        delta (float)        : Orientation matrix regularization
    Returns:
        np.ndarray: (C,16) tensor of cluster features (center, orientation, direction, size)
    """
    feats = []
    for c in clusts:
        # Get list of voxels in the cluster
        x = get_cluster_voxels(voxels, c)

        # Handle size 1 clusters seperately
        if len(c) < 2:
            # Don't waste time with computations, default to regularized
            # orientation matrix, zero direction
            center = x.flatten()
            B = delta * np.eye(3)
            v0 = np.zeros(3)
            feats.append(np.concatenate((center, B.flatten(), v0, [len(c)])))
            continue

        # Start point estimate
        #firstid, lastid = cluster_start_point(x)
        #first = x[firstid]
        #last  = x[lastid]

        # Direction estimate
        #fdir = cluster_direction(x, firstid, max_dist=5)
        #ldir = cluster_direction(x, lastid, max_dist=5)

        # Center data
        center = np.mean(x, axis=0)
        x = x - center

        # Get orientation matrix
        A = x.T.dot(x)

        # Get eigenvectors
        w, v = np.linalg.eigh(A)
        dirwt = 0.0 if w[2] == 0 else 1.0 - w[1] / w[2]
        w = w + delta
        w = w / w[2]

        # Orientation matrix with regularization
        B = (1.-delta) * v.dot(np.diag(w)).dot(v.T) + delta * np.eye(3)

        # get direction - look at direction of spread orthogonal to v[:,maxind]
        v0 = v[:,2]

        # Projection of x along v0
        x0 = x.dot(v0)

        # Projection orthogonal to v0
        xp0 = x - np.outer(x0, v0)
        np0 = np.linalg.norm(xp0, axis=1)

        # spread coefficient
        sc = np.dot(x0, np0)
        if sc < 0:
            # Reverse
            v0 = -v0

        # Weight direction
        v0 = dirwt * v0

        # Append (center, B.flatten(), v0, size)
        feats.append(np.concatenate((center, B.flatten(), v0, [len(c)])))
        #feats.append(np.concatenate((center, B.flatten(), v0, [len(c)], first, fdir)))
        #feats.append(np.concatenate((center, B.flatten(), v0, [len(c)], first, fdir, last, ldir)))

    return np.vstack(feats)


def umbrella_curv(vox, voxid):
    """
    Computes the umbrella curvature as in equation 9 of "Umbrella Curvature:
    A New Curvature Estimation Method for Point Clouds" by A.Foorginejad and K.Khalili
    (https://www.sciencedirect.com/science/article/pii/S2212017313006828)

    Args:
        voxels (np.ndarray): (N,3) Voxel coordinates [x, y, z]
        voxid  (int)       : Voxel ID in which to compute the curvature
    Returns:
        int: Value of the curvature in voxid with respect to the rest of the point cloud
    """
    # Find the mean direction from that point
    import numpy.linalg as LA
    refvox = vox[voxid]
    axis = np.mean([v-refvox for v in vox], axis=0)
    axis /= LA.norm(axis)

    # Find the umbrella curvature (mean angle from the mean direction)
    return abs(np.mean([np.dot((vox[i]-refvox)/LA.norm(vox[i]-refvox), axis) for i in range(len(vox)) if i != voxid]))


def cluster_start_point(voxels):
    """
    Finds the start point of a cluster by:
    1. Find the principal axis a of the point cloud
    2. Find the coordinate a_i of each point along this axis
    3. Find the points with minimum and maximum coordinate
    4. Find the point that has the largest umbrella curvature

    Args:
        voxels (np.ndarray): (N,3) Voxel coordinates [x, y, z]
    Returns:
        int: ID of the start voxel
    """
    from sklearn.decomposition import PCA
    pca = PCA()
    pca.fit(voxels)
    axis = pca.components_[0,:]

    # Compute coord values along that axis
    coords = [np.dot(v, axis) for v in voxels]
    ids = np.array([np.argmin(coords), np.argmax(coords)])

    # Compute curvature of the
    curvs = [umbrella_curv(voxels, ids[0]), umbrella_curv(voxels, ids[1])]

    # Return ID of the point
    return ids[np.argsort(curvs)]


def get_cluster_start_points(data, clusts):
    """
    Function that estimates the start point of clusters
    based on their PCA and local curvature.

    Args:
        data (np.ndarray)    : (N,3) Voxel coordinates [x, y, z]
        clusts ([np.ndarray]): (C) List of arrays of voxel IDs in each cluster
    Returns:
        np.ndarray: (C,3) tensor of cluster start points
    """
    points = []
    for c in clusts:
        # Get list of voxels in the cluster
        x = get_cluster_voxels(data, c)
        vid = cluster_start_point(x)[0]
        points.append(x[vid])

    return np.vstack(points)


def cluster_direction(data, i, max_dist=-1):
    """
    Finds the orientation of the cluster by computing the
    mean direction from the start point.

    Args:
        data (np.ndarray): (N,3) Voxel coordinates [x, y, z]
        i (int)          : ID of the start voxel
        max_dist (float) : Max distance between start voxel and other voxels
    Returns:
        np.ndarray: (3) Orientation
    """
    # If max_dist is set, limit the set of voxels to those within
    # a sphere of radius max_dist
    voxels = data[:,:3]
    svoxel = voxels[i]
    if max_dist > 0:
        from scipy.spatial.distance import cdist
        dist_mat = cdist(svoxel.reshape(1,-1), voxels).reshape(-1)
        voxels = voxels[dist_mat < max_dist]

    # Compute mean direction with respect to start point, normalize it
    mean = np.mean(np.vstack([v-svoxel for v in voxels]), axis=0)
    if np.linalg.norm(mean):
        return mean/np.linalg.norm(mean)
    return mean


def get_cluster_directions(data, clusts, ids, max_dist=-1):
    """
    Finds the orientation of all the clusters by computing the
    mean direction from the start point.

    Args:
        data (np.ndarray)    : (N,3) Voxel coordinates [x, y, z]
        clusts ([np.ndarray]): (C) List of arrays of voxel IDs in each cluster
        ids (np.ndarray)     : (C) IDs of the start voxel in each cluster
        max_dist (float)     : Max distance between start voxel and other voxels
    Returns:
        np.ndarray: (3) Orientation
    """
    # If max_dist is set, limit the set of voxels to those within
    # a sphere of radius max_dist
    dirs = []
    for i, c in enumerate(clusts):
        # Get list of voxels in the cluster
        x = get_cluster_voxels(data, c)
        dir = cluster_direction(x, ids[i], max_dist)
        dirs.append(dir)

    return np.vstack(dirs)
