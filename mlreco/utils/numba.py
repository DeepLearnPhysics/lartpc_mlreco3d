import numpy as np
import numba as nb
import torch
import inspect
from functools import wraps

def numba_wrapper(cast_args=[], list_args=[], keep_torch=False, ref_arg=None):
    '''
    Function which wraps a *numba* function with some checks on the input
    to make the relevant conversions to numpy where necessary.

    Args:
        cast_args ([str]): List of arguments to be cast to numpy
        list_args ([str]): List of arguments which need to be cast to a numba typed list
        keep_torch (bool): Make the output a torch object, if the reference argument is one
        ref_arg (str)    : Reference argument used to assign a type and device to the torch output
    Returns:
        Function
    '''
    def outer(fn):
        @wraps(fn)
        def inner(*args, **kwargs):
            # Convert the positional arguments in args into key:value pairs in kwargs
            keys = list(inspect.signature(fn).parameters.keys())
            for i, val in enumerate(args):
                kwargs[keys[i]] = val

            # Extract the default values for the remaining parameters
            for key, val in inspect.signature(fn).parameters.items():
                if key not in kwargs and val.default != inspect.Parameter.empty:
                    kwargs[key] = val.default

            # If a torch output is request, register the input dtype and device location
            if keep_torch:
                assert ref_arg in kwargs
                dtype, device = None, None
                if isinstance(kwargs[ref_arg], torch.Tensor):
                    dtype = kwargs[ref_arg].dtype
                    device = kwargs[ref_arg].device

            # If the cast data is not a numpy array, cast it
            for arg in cast_args:
                assert arg in kwargs
                if not isinstance(kwargs[arg], np.ndarray):
                    assert isinstance(kwargs[arg], torch.Tensor)
                    kwargs[arg] = kwargs[arg].detach().cpu().numpy() # For now cast to CPU only

            # If there is a reflected list in the input, type it
            for arg in list_args:
                assert arg in kwargs
                kwargs[arg] = nb.typed.List(kwargs[arg])

            # Get the output
            ret = fn(**kwargs)
            if keep_torch and dtype:
                if isinstance(ret, np.ndarray):
                    ret = torch.tensor(ret, dtype=dtype, device=device)
                elif isinstance(ret, list):
                    ret = [torch.tensor(r, dtype=dtype, device=device) for r in ret]
                else:
                    raise TypeError('Return type not recognized, cannot cast to torch')
            return ret
        return inner
    return outer


@nb.njit(cache=True)
def unique_nb(x: nb.int32[:]) -> (nb.int32[:], nb.int32[:]):
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
def submatrix_nb(x:nb.float32[:,:],
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
def cdist_nb(x1: nb.float32[:,:],
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
def mean_nb(x: nb.float32[:,:],
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
def argmin_nb(x: nb.float32[:,:],
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
def argmax_nb(x: nb.float32[:,:],
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
def min_nb(x: nb.float32[:,:],
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
def max_nb(x: nb.float32[:,:],
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
def all_nb(x: nb.float32[:,:],
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
def softmax_nb(x: nb.float32[:,:],
               axis: nb.int32) -> nb.float32[:,:]:
    assert axis == 0 or axis == 1
    if axis == 0:
        xmax = max_nb(x, axis=0)
        logsumexp = np.log(np.sum(np.exp(x-xmax), axis=0)) + xmax
        return np.exp(x - logsumexp)
    else:
        xmax = max_nb(x, axis=1).reshape(-1,1)
        logsumexp = np.log(np.sum(np.exp(x-xmax), axis=1)).reshape(-1,1) + xmax
        return np.exp(x - logsumexp)


@nb.njit(cache=True)
def log_loss_nb(x1: nb.boolean[:], x2: nb.float32[:]) -> nb.float32:
    if len(x1) > 0:
        return -(np.sum(np.log(x2[x1])) + np.sum(np.log(1.-x2[~x1])))/len(x1)
    else:
        return 0.
