# Defines cluster formation and feature extraction
import numpy as np
import torch

def form_clusters(data, min_size=-1, column=5):
    """
    Function that returns a list of of arrays of voxel IDs
    that make up each of the clusters in the input tensor.

    Args:
        data (np.ndarray): (N,6-10) [x, y, z, batchid, value, id(, groupid, intid, nuid, shape)]
        min_size (int)   : Minimal cluster size
        column (int)     : Specifies on which column to base the clusters
    Returns:
        [np.ndarray]: (C) List of arrays of voxel IDs in each cluster
    """
    clusts = []
    for b in data[:, 3].unique():
        binds = torch.nonzero(data[:, 3] == b).flatten()
        for c in data[binds,column].unique():
            # Skip if the cluster ID is -1 (not defined)
            if c < 0:
                continue
            clust = torch.nonzero(data[binds,column] == c).flatten()
            if len(clust) < min_size:
                continue
            clusts.append(binds[clust])

    return clusts


def reform_clusters(data, clust_ids, batch_ids, column=5):
    """
    Function that returns a list of of arrays of voxel IDs
    that make up the requested clusters.

    Args:
        data (np.ndarray)     : (N,8) [x, y, z, batchid, value, id, groupid, shape]
        clust_ids (np.ndarray): (C) List of cluster ids
        batch_ids (np.ndarray): (C) List of batch ids
        column (int)          : Specifies on which column to base the clusters
    Returns:
        [np.ndarray]: (C) List of arrays of voxel IDs in each cluster
    """
    return np.array([np.where((data[:,3] == batch_ids[j]) & (data[:,column] == clust_ids[j]))[0] for j in range(len(batch_ids))])


def get_cluster_batch(data, clusts):
    """
    Function that returns the batch ID of each cluster.
    This should be unique for each cluster, assert that it is.

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


def get_cluster_label(data, clusts, column=5):
    """
    Function that returns the majority label of each cluster,
    as specified in the requested data column.

    Args:
        data (np.ndarray)    : (N,8) [x, y, z, batchid, value, id, groupid, shape]
        clusts ([np.ndarray]): (C) List of arrays of voxel IDs in each cluster
    Returns:
        np.ndarray: (C) List of cluster IDs
    """
    labels = []
    for c in clusts:
        v, cts = data[c,column].unique(return_counts=True)
        labels.append(int(v[cts.argmax()].item()))

    return np.array(labels)


def get_momenta_labels(data, clusts, columns=[7,8,9]):
    """
    Function that returns the momentum unit vector of each cluster.

    Args:
        data (np.ndarray)    : (N,12) [x, y, z, batchid, value, id, groupid, px, py, pz, p, pdg]
        clusts ([np.ndarray]): (C) List of arrays of voxel IDs in each cluster
    Returns:
        np.ndarray: (C) List of cluster IDs
    """
    labels = []
    for c in clusts:
        v = data[c,:]
        # print(v[:, columns].mean(dim=0))
        labels.append(v[:, columns].mean(dim=0))
    labels = torch.stack(labels, dim=0)
    return labels.to(dtype=torch.float32)


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


def get_cluster_features(data, clusts, delta=0.0, whether_adjust_direction=False):
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
        x = get_cluster_voxels(data, c)

        # Handle size 1 clusters seperately
        if len(c) < 2:
            # Don't waste time with computations, default to regularized
            # orientation matrix, zero direction
            center = x.flatten()
            B = delta * np.eye(3)
            v0 = np.zeros(3)
            feats.append(np.concatenate((center, B.flatten(), v0, [len(c)])))
            continue

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

        # If adjust the direction
        if whether_adjust_direction:
            if np.dot(
                    x[find_start_point(x.detach().cpu().numpy())],
                    v0
            ) > 0:
                v0 = -v0

        # Append (center, B.flatten(), v0, size)
        feats.append(np.concatenate((center, B.flatten(), v0, [len(c)])))

    return np.vstack(feats)


def get_cluster_features_extended(data_values, data_sem_types, clusts):
    """
    Function that returns the an array of 16 features for
    each of the clusters in the provided list.

    Args:
        data_values (np.ndarray)    : (N) value
        data_sem_types (np.ndarray) : (N) sem_type
        clusts ([np.ndarray])       : (C) List of arrays of voxel IDs in each cluster
    Returns:
        np.ndarray: (C,3) tensor of cluster features (mean value, std value, major sem_type)
    """
    feats = []
    for c in clusts:
        # Get values for the clusts
        vs = data_values[c]
        ts = data_sem_types[c]

        # mean value
        mean_value = np.mean(vs)
        std_value = np.std(vs)

        # get majority of semantic types
        types, cnts = np.unique(ts, return_counts=True)
        major_sem_type = types[np.argmax(cnts)]

        feats.append([mean_value, std_value, major_sem_type])

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


def get_cluster_points_label(data, particles, clusts, groupwise=False):
    """
    Function that gets label points for each cluster.
    - If fragments (groupwise=False), returns start point only
    - If particle instance (groupwise=True), returns start point of primary shower fragment
      twice if shower, delta or Michel and end points of tracks if track.

    Args:
        data (torch.tensor)     : (N,6) Voxel coordinates [x, y, z, batch_id, value, clust_id, group_id]
        particles (torch.tensor): (N,8) Point coordinates [start_x, start_y, start_z, last_x, last_y, last_z, batch_id, start_t]
        clusts ([np.ndarray])   : (C) List of arrays of voxel IDs in each cluster
        groupwise (bool)        : Whether or not to get a single point per group (merges shower fragments)
    Returns:
        np.ndarray: (N,3/6) particle wise start (and end points in RANDOMIZED ORDER)
    """
    # Get batch_ids and group_ids
    batch_ids = get_cluster_batch(data, clusts)
    points = []
    if not groupwise:
        clust_ids = get_cluster_label(data, clusts)
        for i, c in enumerate(clusts):
            batch_mask = torch.nonzero(particles[:,3] == batch_ids[i]).flatten()
            idx = batch_mask[clust_ids[i]]
            points.append(particles[idx,:3])
    else:
        for i, c in enumerate(clusts):
            batch_mask = torch.nonzero(particles[:,3] == batch_ids[i]).flatten()
            clust_ids  = data[c,5].unique().long()
            maxid = torch.argmin(particles[batch_mask][clust_ids,-1])
            order = [0, 1, 2, 4, 5, 6] if np.random.choice(2) else [4, 5, 6, 0, 1, 2]
            points.append(particles[batch_mask][clust_ids[maxid],order])

    return torch.stack(points)


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
        vid = cluster_start_point(x)[-1]
        points.append(x[vid])

    return np.vstack(points)


def cluster_direction(data, start, max_dist=-1, optimize=False):
    """
    Finds the orientation of the cluster by computing the
    mean direction from the start point.

    Args:
        data (torch.tensor) : (N,3) Voxel coordinates [x, y, z]
        start (torch.tensor): (3) Start voxel coordinates [x, y, z]
        max_dist (float)    : Max distance between start voxel and other voxels in the mean
        optimize (bool)      : Optimizes the number of points involved in the estimate
    Returns:
        torch.tensor: (3) Orientation
    """
    # If max_dist is set, limit the set of voxels to those within
    # a sphere of radius max_dist
    voxels = data[:,:3]
    if max_dist > 0 and not optimize:
        from mlreco.utils import local_cdist
        dist_mat = local_cdist(start.reshape(1,-1), voxels).reshape(-1)
        voxels = voxels[dist_mat <= max_dist]
        if len(voxels) < 2:
            return start-start
    elif optimize:
        # Order the cluster points by increasing distance to the start point
        from mlreco.utils import local_cdist
        dist_mat = local_cdist(start.reshape(1,-1), voxels).reshape(-1)
        order = torch.argsort(dist_mat)
        voxels = voxels[order]
        dist_mat = dist_mat[order]

        # Find the PCA relative secondary spread for each point
        labels = torch.zeros(len(voxels))
        meank = torch.mean(voxels[:3], dim=0)
        covk = (voxels[:3]-meank).t().mm(voxels[:3]-meank)/3
        for i in range(2, len(voxels)):
            # Get the eigenvalues and eigenvectors, identify point of minimum secondary spread
            w, _ = torch.eig(covk)
            w, _ = w[:,0].reshape(-1).sort()
            labels[i] = torch.sqrt(w[2]/(w[0]+w[1])) if (w[0]+w[1]) else 0.
            if dist_mat[i] == dist_mat[i-1]:
                labels[i-1] = 0.

            # Increment mean and matrix
            if i != len(voxels)-1:
                meank = ((i+1)*meank+voxels[i+1])/(i+2)
                covk = (i+1)*covk/(i+2) + (voxels[i+1]-meank).reshape(-1,1)*(voxels[i+1]-meank)/(i+1)

        # Subselect voxels that are most track-like
        max_id = torch.argmax(labels)
        voxels = voxels[:max_id+1]

    # Compute mean direction with respect to start point, normalize it
    mean = torch.mean(torch.stack([v-start for v in voxels]), dim=0)
    if torch.norm(mean):
        return mean/torch.norm(mean)
    return mean


def get_cluster_directions(data, starts, clusts, max_dist=-1, optimize=False):
    """
    Finds the orientation of all the clusters by computing the
    mean direction from the start point.

    Args:
        data (torch.tensor)  : (N,3) Voxel coordinates [x, y, z]
        starts (torch.tensor): (C,3) Coordinates of the start points
        clusts ([np.ndarray]): (C) List of arrays of voxel IDs in each cluster
        max_dist (float)     : Max distance between start voxel and other voxels
        optimize (bool)      : Optimizes the number of points involved in the estimate
    Returns:
        torch.tensor: (3) Orientation
    """
    # If max_dist is set, limit the set of voxels to those within
    # a sphere of radius max_dist
    dirs = []
    for i, c in enumerate(clusts):
        # Get list of voxels in the cluster
        x = get_cluster_voxels(data, c)
        dir = cluster_direction(x, starts[i], max_dist, optimize)
        dirs.append(dir)

    return torch.stack(dirs)


def relabel_groups(data, clusts, groups, new_array=True):
    """
    Function that resets the value of the group data column according
    to the cluster value specified for each cluster.

    Args:
        data (torch.Tensor)    : N_GPU array of (N,8) [x, y, z, batchid, value, id, groupid, shape]
        clusts ([[np.ndarray]]): N_GPU array of (C) List of arrays of voxel IDs in each cluster
        groups ([np.ndarray])  : N_GPU array of (C) List of group ids for each cluster
        new_array (bool)       : Whether or not to deep copy the data array
    Returns:
        torch.Tensor: (N,8) Relabeled [x, y, z, batchid, value, id, groupid, shape]
    """
    data_new = data
    if new_array:
        import copy
        data_new = copy.deepcopy(data)

    device = data[0].device
    dtype  = data[0].dtype
    for i in range(len(data)):
        batches = data[i][:,3]
        for b in batches.unique():
            batch_mask = torch.nonzero(batches == b).flatten()
            labels = data[i][batch_mask]
            batch_clusts = clusts[i][b.int().item()]
            if not len(batch_clusts):
                continue
            clust_ids = get_cluster_label(labels, batch_clusts, column=5)
            group_ids = groups[i][b.int().item()]
            true_group_ids = get_cluster_label(labels, batch_clusts, column=6)
            primary_mask   = clust_ids == true_group_ids
            new_id = max(clust_ids)+1
            for g in np.unique(group_ids):
                group_mask     = group_ids == g
                primary_labels = np.where(primary_mask & group_mask)[0]
                group_id = -1
                if len(primary_labels) != 1:
                    group_id = new_id
                    new_id += 1
                else:
                    group_id = clust_ids[primary_labels[0]]
                for c in batch_clusts[group_mask]:
                    new_groups = torch.full([len(c)], group_id, dtype=dtype).to(device)
                    data_new[i][batch_mask[c], 6] = new_groups

    return data_new
