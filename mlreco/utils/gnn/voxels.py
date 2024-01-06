import numpy as np
import numba as nb

import mlreco.utils.numba_local as nbl
from mlreco.utils.decorators import numbafy


@numbafy(cast_args=['data'], keep_torch=True, ref_arg='data')
def get_voxel_features(data, max_dist=5.0):
    """
    Function that returns the an array of 16 features for
    each of the voxels in the provided tensor.

    Args:
        data (np.ndarray)    : (N,8) [x, y, z, batchid, value, id, groupid, shape]
        max_dist (float)     : Defines "local", i.e. max distance to look at
    Returns:
        np.ndarray: (N,16) tensor of voxel features (coords, local orientation, local direction, local count)
    """
    return _get_voxel_features(voxels, max_dist)

@nb.njit(parallel=True, cache=True)
def _get_voxel_features(data: nb.float32[:,:], max_dist=5.0):

    # Compute intervoxel distance matrix
    voxels = data[:,:3]
    dist_mat = nbl.cdist(voxels, voxels)

    # Get local geometrical features for each voxel
    feats = np.empty((len(voxels), 16), dtype=data.dtype)
    for k in nb.prange(len(voxels)):

        # Restrict the points to the neighborood of the voxel
        voxel = voxels[k]
        x = voxels[dist_mat[k] < max_dist,:3]

        # Do not waste time with computations with size 1 clusters, default to zeros
        if len(x) < 2:
            feats[k] = np.concatenate((voxel, np.zeros(12), np.array([len(x)])))
            continue

        # Center data around voxel
        x = x - voxel

        # Get orientation matrix
        A = x.T.dot(x)

        # Get eigenvectors, normalize orientation matrix and eigenvalues to largest
        # This step assumes points are not superimposed, i.e. that largest eigenvalue != 0
        w, v = np.linalg.eigh(A)
        dirwt = 1.0 - w[1] / w[2]
        B = A / w[2]

        # get direction - look at direction of spread orthogonal to v[:,maxind]
        v0 = v[:,2]

        # Projection of x along v0
        x0 = x.dot(v0)

        # Projection orthogonal to v0
        xp0 = x - np.outer(x0, v0)
        np0 = np.empty(len(xp0), dtype=data.dtype)
        for i in range(len(xp0)):
            np0[i] = np.linalg.norm(xp0[i])

        # Flip the principal direction if it is not pointing towards the maximum spread
        sc = np.dot(x0, np0)
        if sc < 0:
            v0 = np.zeros(3, dtype=data.dtype)-v0 # (-)/negative doesn't work with numba for now...

        # Weight direction
        v0 = dirwt * v0

        # Append
        feats[k] = np.concatenate((voxel, B.flatten(), v0, np.array([len(x)])))

    return np.vstack(feats)
