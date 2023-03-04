import numpy as np
import numba as nb


@nb.njit(cache=True)
def submatrix(x: nb.float32[:,:],
              index1: nb.int32[:],
              index2: nb.int32[:]) -> nb.float32[:,:]:
    """
    Numba implementation of matrix subsampling.

    Parameters
    ----------
    x : np.ndarray
        (N,M) array of values
    index1 : np.ndarray
        (N') array of indices along axis 0 in the input matrix
    index2 : np.ndarray
        (M') array of indices along axis 1 in the input matrix

    Returns
    -------
    np.ndarray
        (N',M') array of values from the original matrix
    """
    subx = np.empty((len(index1), len(index2)), dtype=x.dtype)
    for i, i1 in enumerate(index1):
        for j, i2 in enumerate(index2):
            subx[i,j] = x[i1,i2]
    return subx


@nb.njit(cache=True)
def unique(x: nb.int32[:]) -> (nb.int32[:], nb.int32[:]):
    """
    Numba implementation of `np.unique(x, return_counts=True)`.

    Parameters
    ----------
    x : np.ndarray
        (N) array of values

    Returns
    -------
    np.ndarray
        (U) array of unique values
    np.ndarray
        (U) array of counts of each unique value in the original array
    """
    b = np.sort(x.flatten())
    unique = list(b[:1])
    counts = [1 for _ in unique]
    for x in b[1:]:
        if x != unique[-1]:
            unique.append(x)
            counts.append(1)
        else:
            counts[-1] += 1
    return unique, counts


@nb.njit(cache=True)
def mean(x: nb.float32[:,:],
         axis: nb.int32) -> nb.float32[:]:
    """
    Numba implementation of `np.mean(x, axis)`.

    Parameters
    ----------
    x : np.ndarray
        (N,M) array of values
    axis : int
        Array axis ID

    Returns
    -------
    np.ndarray
        (N) or (M) array of `mean` values
    """
    assert axis == 0 or axis == 1
    mean = np.empty(x.shape[1-axis], dtype=x.dtype)
    if axis == 0:
        for i in range(len(mean)):
            mean[i] = np.mean(x[:,i])
    else:
        for i in range(len(mean)):
            mean[i] = np.mean(x[i])
    return mean


@nb.njit(cache=True)
def argmin(x: nb.float32[:,:],
           axis: nb.int32) -> nb.int32[:]:
    """
    Numba implementation of `np.argmin(x, axis)`.

    Parameters
    ----------
    x : np.ndarray
        (N,M) array of values
    axis : int
        Array axis ID

    Returns
    -------
    np.ndarray
        (N) or (M) array of `argmin` values
    """
    assert axis == 0 or axis == 1
    argmin = np.empty(x.shape[1-axis], dtype=np.int32)
    if axis == 0:
        for i in range(len(argmin)):
            argmin[i] = np.argmin(x[:,i])
    else:
        for i in range(len(argmin)):
            argmin[i] = np.argmin(x[i])
    return argmin


@nb.njit(cache=True)
def argmax(x: nb.float32[:,:],
           axis: nb.int32) -> nb.int32[:]:
    """
    Numba implementation of `np.argmax(x, axis)`.

    Parameters
    ----------
    x : np.ndarray
        (N,M) array of values
    axis : int
        Array axis ID

    Returns
    -------
    np.ndarray
        (N) or (M) array of `argmax` values
    """
    assert axis == 0 or axis == 1
    argmax = np.empty(x.shape[1-axis], dtype=np.int32)
    if axis == 0:
        for i in range(len(argmax)):
            argmax[i] = np.argmax(x[:,i])
    else:
        for i in range(len(argmax)):
            argmax[i] = np.argmax(x[i])
    return argmax


@nb.njit(cache=True)
def min(x: nb.float32[:,:],
        axis: nb.int32) -> nb.float32[:]:
    """
    Numba implementation of `np.max(x, axis)`.

    Parameters
    ----------
    x : np.ndarray
        (N,M) array of values
    axis : int
        Array axis ID

    Returns
    -------
    np.ndarray
        (N) or (M) array of `min` values
    """
    assert axis == 0 or axis == 1
    xmin = np.empty(x.shape[1-axis], dtype=np.int32)
    if axis == 0:
        for i in range(len(xmin)):
            xmin[i] = np.min(x[:,i])
    else:
        for i in range(len(xmax)):
            xmin[i] = np.min(x[i])
    return xmin


@nb.njit(cache=True)
def max(x: nb.float32[:,:],
        axis: nb.int32) -> nb.float32[:]:
    """
    Numba implementation of `np.max(x, axis)`.

    Parameters
    ----------
    x : np.ndarray
        (N,M) array of values
    axis : int
        Array axis ID

    Returns
    -------
    np.ndarray
        (N) or (M) array of `max` values
    """
    assert axis == 0 or axis == 1
    xmax = np.empty(x.shape[1-axis], dtype=np.int32)
    if axis == 0:
        for i in range(len(xmax)):
            xmax[i] = np.max(x[:,i])
    else:
        for i in range(len(xmax)):
            xmax[i] = np.max(x[i])
    return xmax


@nb.njit(cache=True)
def all(x: nb.float32[:,:],
        axis: nb.int32) -> nb.boolean[:]:
    """
    Numba implementation of `np.all(x, axis)`.

    Parameters
    ----------
    x : np.ndarray
        (N,M) array of values
    axis : int
        Array axis ID

    Returns
    -------
    np.ndarray
        (N) or (M) array of `all` outputs
    """
    assert axis == 0 or axis == 1
    all = np.empty(x.shape[1-axis], dtype=np.bool_)
    if axis == 0:
        for i in range(len(all)):
            all[i] = np.all(x[:,i])
    else:
        for i in range(len(all)):
            all[i] = np.all(x[i])
    return all


@nb.njit(cache=True)
def softmax(x: nb.float32[:,:],
            axis: nb.int32) -> nb.float32[:,:]:
    """
    Numba implementation of `scipy.special.softmax(x, axis)`.

    Parameters
    ----------
    x : np.ndarray
        (N,M) array of values
    axis : int
        Array axis ID

    Returns
    -------
    np.ndarray
        (N,M) Array of softmax scores
    """
    assert axis == 0 or axis == 1
    if axis == 0:
        xmax = max(x, axis=0)
        logsumexp = np.log(np.sum(np.exp(x-xmax), axis=0)) + xmax
        return np.exp(x - logsumexp)
    else:
        xmax = max(x, axis=1).reshape(-1,1)
        logsumexp = np.log(np.sum(np.exp(x-xmax), axis=1)).reshape(-1,1) + xmax
        return np.exp(x - logsumexp)


@nb.njit(cache=True)
def log_loss(label: nb.boolean[:],
             pred: nb.float32[:]) -> nb.float32:
    """
    Numba implementation of cross-entropy loss.
    
    Parameters
    ----------
    label : np.ndarray
        (N) array of boolean labels (0 or 1)
    pred : np.ndarray
        (N) array of float scores (between 0 and 1)

    Returns
    -------
    float
        Cross-entropy loss
    """
    if len(label) > 0:
        return -(np.sum(np.log(pred[label])) + np.sum(np.log(1.-pred[~label])))/len(label)
    else:
        return 0.


@nb.njit(cache=True)
def pdist(x: nb.float32[:,:]) -> nb.float32[:,:]:
    """
    Numba implementation of Eucleadian `scipy.spatial.distance.pdist(x, p=2)` in 3D.

    Parameters
    ----------
    x : np.ndarray
        (N,3) array of point coordinates in the set

    Returns
    -------
    np.ndarray
        (N,N) array of pair-wise Euclidean distances
    """
    res = np.zeros((x.shape[0], x.shape[0]), dtype=x.dtype)
    for i in range(x.shape[0]):
        for j in range(i+1, x.shape[0]):
            res[i,j] = res[j,i] = np.sqrt((x[i][0]-x[j][0])**2+(x[i][1]-x[j][1])**2+(x[i][2]-x[j][2])**2)
    return res


@nb.njit(cache=True)
def cdist(x1: nb.float32[:,:],
          x2: nb.float32[:,:]) -> nb.float32[:,:]:
    """
    Numba implementation of Eucleadian `scipy.spatial.distance.cdist(x, p=2)` in 3D.

    Parameters
    ----------
    x1 : np.ndarray
        (N,3) array of point coordinates in the first set
    x2 : np.ndarray
        (M,3) array of point coordinates in the second set

    Returns
    -------
    np.ndarray
        (N,M) array of pair-wise Euclidean distances
    """
    res = np.empty((x1.shape[0], x2.shape[0]), dtype=x1.dtype)
    for i1 in range(x1.shape[0]):
        for i2 in range(x2.shape[0]):
            res[i1,i2] = np.sqrt((x1[i1][0]-x2[i2][0])**2+(x1[i1][1]-x2[i2][1])**2+(x1[i1][2]-x2[i2][2])**2)
    return res


@nb.njit(cache=True)
def farthest_pair(x: nb.float32[:,:],
                  algorithm: bool = 'brute') -> (nb.int32, nb.int32, nb.float32):
    '''
    Algorithm which finds the two points which are
    farthest from each other in a set.

    Two algorithms:
    - `brute`: compute pdist, use argmax
    - `recursive`: Start with the first point in one set, find the farthest
                   point in the other, move to that point, repeat. This
                   algorithm is *not* exact, but a good and very quick proxy.

    Parameters
    ----------
    x : np.ndarray
        (Nx3) array of point coordinates
    algorithm : str
        Name of the algorithm to use: `brute` or `recursive`

    Returns
    -------
    int
        ID of the first point that makes up the pair
    int
        ID of the second point that makes up the pair
    float
        Distance between the two points
    '''
    if algorithm == 'brute':
        dist_mat = pdist(x)
        index = np.argmax(dist_mat)
        idxs = [index//x.shape[0], index%x.shape[0]]
        dist = dist_mat[idxs[0], idxs[1]]
    elif algorithm == 'recursive':
        idxs, subidx, dist, tempdist = [0, 0], False, 1e9, 1e9+1.
        while dist < tempdist:
            tempdist = dist
            dists = cdist(np.ascontiguousarray(x[idxs[int(subidx)]]).reshape(1,-1), x).flatten()
            idxs[int(~subidx)] = np.argmax(dists)
            dist = dists[idxs[int(~subidx)]]
            subidx = ~subidx
    else:
        raise ValueError('Algorithm not supported')

    return idxs[0], idxs[1], dist
