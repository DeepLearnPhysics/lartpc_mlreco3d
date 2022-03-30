import numpy as np
from mlreco.utils.groups import filter_duplicate_voxels, filter_duplicate_voxels_ref, filter_nonimg_voxels


def clean_data(grp_voxels, grp_data, img_voxels, img_data, meta):
    """
    Helper that factorizes common cleaning operations required
    when trying to match true sparse3d and cluster3d data products.

    1) lexicographically sort group data (images are lexicographically sorted)

    2) remove voxels from group data that are not in image

    3) choose only one group per voxel (by lexicographic order)

    Parameters
    ----------
    grp_voxels: np.ndarray
    grp_data: np.ndarray
    img_voxels: np.ndarray
    img_data: np.ndarray
    meta: larcv::Meta

    Returns
    -------
    grp_voxels: np.ndarray
    grp_data: np.ndarray
    """
    # step 1: lexicographically sort group data
    perm = np.lexsort(grp_voxels.T)
    grp_voxels = grp_voxels[perm,:]
    grp_data = grp_data[perm]

    perm = np.lexsort(img_voxels.T)
    img_voxels = img_voxels[perm,:]
    img_data = img_data[perm]

    # step 2: remove duplicates
    sel1 = filter_duplicate_voxels_ref(grp_voxels, grp_data[:,-1],meta, usebatch=True, precedence=[0,2,1,3,4])
    inds1 = np.where(sel1)[0]
    grp_voxels = grp_voxels[inds1,:]
    grp_data = grp_data[inds1]

    # step 3: remove voxels not in image
    sel2 = filter_nonimg_voxels(grp_voxels, img_voxels, usebatch=False)
    inds2 = np.where(sel2)[0]
    grp_voxels = grp_voxels[inds2,:]
    grp_data = grp_data[inds2]
    return grp_voxels, grp_data
