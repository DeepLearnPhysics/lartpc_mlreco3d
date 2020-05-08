# Defines voxel feature extraction
import numpy as np

def get_voxel_features(voxels, max_dist=5.0, delta=0.0):
    """
    Function that returns the an array of 16 features for
    each of the voxels in the provided tensor.

    Args:
        voxels (np.ndarray)  : (N,3) Voxel coordinates [x, y, z]
        max_dist (float)     : Defines "local", max distance to look at
        delta (float)        : Orientation matrix regularization
    Returns:
        np.ndarray: (N,16) tensor of voxel features (coords, local orientation, local direction, local count)
    """
    # Compute intervoxel distance matrix
    from scipy.spatial.distance import cdist
    dist_mat = cdist(voxels, voxels)

    # Get local geometrical features for each voxel
    feats = []
    for i in range(len(voxels)):

        # Restrict the points to the neighborood of the voxel
        x = voxels[dist_mat[i] < max_dist,:3]
        voxel = voxels[i,:3]

        # Handle size 1 neighbourhoods seperately
        if len(x) < 2:
            # Don't waste time with computations, default to regularized
            # orientation matrix, zero direction
            center = x.flatten()
            B = delta * np.eye(3)
            v0 = np.zeros(3)
            feats.append(np.concatenate((voxel, B.flatten(), v0, [len(x)])))
            continue

        # Center data around voxel
        x = x - voxel

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
        feats.append(np.concatenate((voxel, B.flatten(), v0, [len(x)])))

    return np.vstack(feats)
