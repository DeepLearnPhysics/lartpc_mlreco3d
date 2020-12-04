# Defines cluster formation and feature extraction
import numpy as np
import torch

def form_clusters(data, min_size=-1, column=5, batch_index=3):
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
    for b in data[:, batch_index].unique():
        binds = torch.nonzero(data[:, batch_index] == b, as_tuple=True)[0]
        for c in data[binds,column].unique():
            # Skip if the cluster ID is -1 (not defined)
            if c < 0:
                continue
            clust = torch.nonzero(data[binds,column] == c, as_tuple=True)[0]
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


def get_cluster_batch(data, clusts, batch_index=3):
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
        assert len(data[c,batch_index].unique()) == 1
        labels.append(int(data[c[0],batch_index].item()))

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


def get_cluster_label_np(data, clusts, column=5):
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
        v, cts = np.unique(data[c,column], return_counts=True)
        labels.append(int(v[cts.argmax()].item()))

    return np.array(labels)


def get_momenta_label(data, clusts, column=8):
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
        labels.append(v[:, column].mean(dim=0))
    labels = torch.stack(labels, dim=0)
    return labels.to(dtype=torch.float32)


def get_momenta_label_np(data, clusts, column=8):
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
        labels.append(v[:, column].mean(axis=0))
    labels = np.vstack(labels)
    return labels


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


def get_cluster_dirs(voxels, clusts):
    """
    Function that returns the direction of the listed clusters,
    expressed as its normalized covariance matrix.

    Args:
        voxels (np.ndarray)  : (N,3) Voxel coordinates [x, y, z]
        clusts ([np.ndarray]): (C) List of arrays of voxel IDs in each cluster
    Returns:
        np.ndarray: (C,9) Tensor of cluster directions
    """
    dirs = []
    for c in clusts:

        # Get list of voxels in the cluster
        x = get_cluster_voxels(voxels, c)

        # Do not waste time with computations with size 1 clusters, default to zeros
        if len(c) < 2:
            return dirs.append(np.concatenate(np.zeros(9)))
            continue

        # Center data
        center = np.mean(x, axis=0)
        x = x - center

        # Get orientation matrix
        A = x.T.dot(x)

        # Get eigenvectors, normalize orientation matrix
        # This step assumes points are not superimposed, i.e. that largest eigenvalue != 0
        w, v = np.linalg.eigh(A)
        B = A / w[2]

        # Append (dirs)
        dirs.append(B.flatten())

    return np.vstack(dirs)


def get_cluster_features(data, clusts, whether_adjust_direction=False):
    """
    Function that returns the an array of 16 features for
    each of the clusters in the provided list.

    Args:
        voxels (np.ndarray)  : (N,3) Voxel coordinates [x, y, z]
        clusts ([np.ndarray]): (C) List of arrays of voxel IDs in each cluster
    Returns:
        np.ndarray: (C,16) tensor of cluster features (center, orientation, direction, size)
    """
    feats = []
    for c in clusts:

        # Get list of voxels in the cluster
        x = get_cluster_voxels(data, c)

        # Do not waste time with computations with size 1 clusters, default to zeros
        if len(c) < 2:
            return feats.append(np.concatenate((x.flatten(), np.zeros(9), [len(c)])))
            continue

        # Center data
        center = np.mean(x, axis=0)
        x = x - center

        # Get orientation matrix
        A = x.T.dot(x)

        # Get eigenvectors, normalize orientation matrix and eigenvalues to largest
        # This step assumes points are not superimposed, i.e. that largest eigenvalue != 0
        w, v = np.linalg.eigh(A)
        dirwt = 1.0 - w[1] / w[2]
        B = A / w[2]
        w = w / w[2]

        # Get the principal direction, identify the direction of the spread
        v0 = v[:,2]

        # Projection all points, x, along the principal axis
        x0 = x.dot(v0)

        # Evaluate the distance from the points to the principal axis
        xp0 = x - np.outer(x0, v0)
        np0 = np.linalg.norm(xp0, axis=1)

        # Flip the principal direction if it is not pointing towards the maximum spread
        sc = np.dot(x0, np0)
        if sc < 0:
            v0 = -v0

        # Weight direction
        v0 = dirwt * v0

        # Adjust direction to the start direction if requested
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
            batch_mask = torch.nonzero(particles[:,3] == batch_ids[i], as_tuple=True)[0]
            idx = batch_mask[clust_ids[i]]
            points.append(particles[idx,:3])
    else:
        for i, c in enumerate(clusts):
            batch_mask = torch.nonzero(particles[:,3] == batch_ids[i], as_tuple=True)[0]
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


def cluster_direction(data, start, max_dist=-1, optimize=False, use_cpu=False):
    """
    Finds the orientation of the cluster by computing the
    mean direction from the start point.

    Args:
        data (torch.tensor) : (N,3) Voxel coordinates [x, y, z]
        start (torch.tensor): (3) Start voxel coordinates [x, y, z]
        max_dist (float)    : Max distance between start voxel and other voxels in the mean
        optimize (bool)     : Optimizes the number of points involved in the estimate
        use_cpu (bool)      : Bring data to CPU to hasten optimization
    Returns:
        torch.tensor: (3) Orientation
    """
    # If max_dist is set, limit the set of voxels to those within
    # a sphere of radius max_dist
    device = data.device
    if use_cpu:
        data = data.detach().cpu()
        start = start.detach().cpu()
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
    if use_cpu:
        mean = mean.to(device)
    if torch.norm(mean):
        return mean/torch.norm(mean)
    return mean


def get_cluster_directions(data, starts, clusts, max_dist=-1, optimize=False, use_cpu=False):
    """
    Finds the orientation of all the clusters by computing the
    mean direction from the start point.

    Args:
        data (torch.tensor)  : (N,3) Voxel coordinates [x, y, z]
        starts (torch.tensor): (C,3) Coordinates of the start points
        clusts ([np.ndarray]): (C) List of arrays of voxel IDs in each cluster
        max_dist (float)     : Max distance between start voxel and other voxels
        optimize (bool)      : Optimizes the number of points involved in the estimate
        use_cpu (bool)       : Bring data to CPU to hasten optimization
    Returns:
        torch.tensor: (3) Orientation
    """
    # If max_dist is set, limit the set of voxels to those within
    # a sphere of radius max_dist
    dirs = []
    for i, c in enumerate(clusts):
        # Get list of voxels in the cluster
        x = get_cluster_voxels(data, c)
        dir = cluster_direction(x, starts[i], max_dist, optimize, use_cpu)
        dirs.append(dir)

    return torch.stack(dirs)


def relabel_groups(clust_ids, true_group_ids, pred_group_ids):
    """
    Function that resets the value of the group ids according
    to the predicted group ids, enforcing that clus_id=group_id
    if the cluster corresponds to a primary

    Args:
        clust_ids (np.ndarray)       : (C) List of label cluster ids
        true_group_ids (np.ndarray)  : (C) List of label group ids
        pred_groups_ids (np.ndarray) : (C) List of predicted group ids
    Returns:
        torch.Tensor: (C) Relabeled group ids
    """
    new_group_ids = np.empty(len(pred_group_ids))
    primary_mask = clust_ids == true_group_ids
    new_id = max(clust_ids)+1
    for g in np.unique(pred_group_ids):
        group_mask     = pred_group_ids == g
        primary_labels = np.where(primary_mask & group_mask)[0]
        group_id = -1
        if len(primary_labels) != 1:
            group_id = new_id
            new_id += 1
        else:
            group_id = clust_ids[primary_labels[0]]
        new_group_ids[group_mask] = group_id

    return new_group_ids
