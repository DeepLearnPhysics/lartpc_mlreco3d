import numpy as np
import numba as nb


@nb.njit(cache=True)
def unique(x: nb.int32[:]) -> (nb.int32[:], nb.int32[:]):
    """
    Numba implementation of np.unique
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
def submatrix(x: nb.float32[:,:],
              index1: nb.int32[:],
              index2: nb.int32[:]) -> nb.float32[:,:]:
    """
    Numba implementation of matrix subsampling
    """
    subx = np.empty((len(index1), len(index2)), dtype=x.dtype)
    for i, i1 in enumerate(index1):
        for j, i2 in enumerate(index2):
            subx[i,j] = x[i1,i2]
    return subx


@nb.njit(cache=True)
def pdist(x: nb.float32[:,:]) -> nb.float32[:,:]:
    """
    Numba implementation of Eucleadian pdist in 3D.
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
    Numba implementation of Eucleadian cdist in 3D.
    """
    res = np.empty((x1.shape[0], x2.shape[0]), dtype=x1.dtype)
    for i1 in range(x1.shape[0]):
        for i2 in range(x2.shape[0]):
            res[i1,i2] = np.sqrt((x1[i1][0]-x2[i2][0])**2+(x1[i1][1]-x2[i2][1])**2+(x1[i1][2]-x2[i2][2])**2)
    return res


@nb.njit(cache=True)
def mean(x: nb.float32[:,:],
         axis: nb.int32) -> nb.float32[:]:
    """
    Numba implementation of np.mean(x, axis)
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
    Numba implementation of np.argmin(x, axis)
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
    Numba implementation of np.argmax(x, axis)
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
    Numba implementation of np.max(x, axis)
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
    Numba implementation of np.max(x, axis)
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
        axis: nb.int32) -> nb.int32[:]:
    """
    Numba implementation of np.all(x, axis)
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
    Numba implementation of SciPy's softmax(x, axis)
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
def log_loss(x1: nb.boolean[:],
             x2: nb.float32[:]) -> nb.float32:
    """
    Numba implementation of cross-entropy loss
    """
    if len(x1) > 0:
        return -(np.sum(np.log(x2[x1])) + np.sum(np.log(1.-x2[~x1])))/len(x1)
    else:
        return 0.
