import numpy as np
import torch

from .globals import COORD_COLS


def to_numpy(array):
    '''
    Function which casts an array-like object
    to a `numpy.ndarray`.

    Parameters
    ----------
    array : object
        Array-like object (can be either `np.ndarray`, `torch.Tensor` or `ME.SparseTensor`)

    Returns
    -------
    np.ndarray
        Array cast to np.ndarray
    '''
    import MinkowskiEngine as ME

    if isinstance(array, np.ndarray):
        return array
    elif isinstance(array, torch.Tensor):
        return array.cpu().detach().numpy()
    elif isinstance(array, tuple):
        return np.array(array)
    elif isinstance(array, ME.SparseTensor):
        return torch.cat([array.C.float(), array.F], dim=1).detach().cpu().numpy()
    else:
        raise TypeError('Unknown return type %s' % type(array))


def local_cdist(v1, v2):
    '''
    Function which computes the pairwise distances between two
    collections of points stored as `torch.Tensor` objects.

    This is necessary because the torch.cdist implementation is either
    slower (with the `donot_use_mm_for_euclid_dist` option) or produces
    dramatically wrong answers under certain situations (with the
    `use_mm_for_euclid_dist_if_necessary` option).

    Parameters
    ----------
    v1 : torch.Tensor
        (N, D) tensor of coordinates
    v2 : torch.Tensor
        (M, D) tensor of coordinates

    Returns
    -------
    torch.Tensor
        (N, M) tensor of pairwise distances
    '''
    v1_2 = v1.unsqueeze(1).expand(v1.size(0), v2.size(0), v1.size(1))
    v2_2 = v2.unsqueeze(0).expand(v1.size(0), v2.size(0), v1.size(1))
    return torch.sqrt(torch.pow(v2_2 - v1_2, 2).sum(2))


def pixel_to_cm(coords, meta, translate=True):
    '''
    Converts the pixel indices in a tensor to detector coordinates
    using the metadata information.

    The metadata is assumed to have the following structure:
    [lower_x, lower_y(, lower_z), upper_x, upper_y, (upper_z), size_x, size_y(, size_z)],
    i.e. lower and upper bounds of the volume and pixel/voxel size.

    Parameters
    ----------
    coords : np.ndarray
        (N, 2/3) Input pixel indices
    meta : np.ndarray
        (6/9) Array of metadata information
    translate : bool, default True
        If set to `False`, this function returns the input unchanged
    '''
    if not translate or not len(coords):
        return coords

    lower, upper, size = np.split(np.asarray(meta).reshape(-1), 3)
    out = lower + (coords + .5) * size
    return out.astype(np.float32)


def cm_to_pixel(coords, meta, translate=True):
    '''
    Converts the detector coordinates in a tensor to pixel indices
    using the metadata information.

    The metadata is assumed to have the following structure:
    [lower_x, lower_y(, lower_z), upper_x, upper_y, (upper_z), size_x, size_y(, size_z)],
    i.e. lower and upper bounds of the volume and pixel/voxel size.

    Parameters
    ----------
    coords : np.ndarray
        (N, 2/3) Input detector coordinates
    meta : np.ndarray
        (6/9) Array of metadata information
    translate : bool, default True
        If set to `False`, this function returns the input unchanged
    '''
    if not translate or not len(coords):
        return coords

    lower, upper, size = np.split(np.asarray(meta).reshape(-1), 3)
    return (coords - lower) / size - .5
