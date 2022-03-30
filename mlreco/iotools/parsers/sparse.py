import numpy as np
from larcv import larcv
from mlreco.utils.groups import filter_duplicate_voxels, filter_duplicate_voxels_ref, filter_nonimg_voxels


def parse_sparse2d_scn(data):
    """
    A function to retrieve sparse tensor input from larcv::EventSparseTensor2D object

    Returns the data in format to pass to SCN

    .. code-block:: yaml

        schema:
          segment_label:
            - parse_sparse2d_scn
            - sparse2d_pcluster

    Configuration
    -------------
    sparse2d_pcluster: larcv::EventSparseTensor2D
        Optionally, give an array of (larcv::EventSparseTensor2D, int) for projection id

    Returns
    -------
    voxels: np.ndarray(int32)
        Coordinates with shape (N,2)
    data: np.ndarray(float32)
        Pixel values/channels with shape (N,C)
    """
    meta = None
    output = []
    np_voxels = None
    for event_tensor2d in data:
        projection_id = 0  # default
        if isinstance(event_tensor2d, tuple):
            projection_id = event_tensor2d[1]
            event_tensor2d = event_tensor2d[0]

        tensor2d = event_tensor2d.sparse_tensor_2d(projection_id)
        num_point = tensor2d.as_vector().size()

        if meta is None:

            meta = tensor2d.meta()
            np_voxels = np.empty(shape=(num_point, 2), dtype=np.int32)
            larcv.fill_2d_voxels(tensor2d, np_voxels)

        # else:
        #     assert meta == tensor2d.meta()
        np_data = np.empty(shape=(num_point, 1), dtype=np.float32)
        larcv.fill_2d_pcloud(tensor2d, np_data)
        output.append(np_data)
    return np_voxels, np.concatenate(output, axis=-1)


def parse_sparse3d_scn(data):
    """
    A function to retrieve sparse tensor input from larcv::EventSparseTensor3D object

    Returns the data in format to pass to DataLoader

    .. code-block:: yaml

        schema:
          segment_label:
            - parse_sparse3d_scn
            - sparse3d_pcluster

    Configuration
    -------------
    sparse3d_pcluster: larcv::EventSparseTensor3D
        Can be repeated to load more features (one per feature).

    Returns
    -------
    voxels: numpy array(int32) with shape (N,3)
        Coordinates
    data: numpy array(float32) with shape (N,C)
        Pixel values/channels, as many channels as specified larcv::EventSparseTensor3D.
    """
    meta = None
    output = []
    np_voxels = None
    for event_tensor3d in data:
        num_point = event_tensor3d.as_vector().size()
        if meta is None:
            meta = event_tensor3d.meta()
            np_voxels = np.empty(shape=(num_point, 3), dtype=np.int32)
            larcv.fill_3d_voxels(event_tensor3d, np_voxels)
        else:
            assert meta == event_tensor3d.meta()
        np_data = np.empty(shape=(num_point, 1), dtype=np.float32)
        larcv.fill_3d_pcloud(event_tensor3d, np_data)
        output.append(np_data)
    return np_voxels, np.concatenate(output, axis=-1)


def parse_sparse3d(data):
    """
    A function to retrieve sparse tensor from larcv::EventSparseTensor3D object
    and return it in concatenated form (shape (N, 3+C)) instead of voxels and
    features arrays separately.

    .. code-block:: yaml

        schema:
          segment_label:
            - parse_sparse3d
            - sparse3d_pcluster

    Configuration
    -------------
    sparse3d_pcluster: larcv::EventSparseTensor3D
        Can be repeated to load more features (one per feature).

    Returns
    -------
    np.ndarray
        a numpy array with the shape (N,3+C) where 3+C represents
        (x,y,z) coordinate and C stored pixel values (channels).
    """
    meta = None
    output = []
    for event_tensor3d in data:
        num_point = event_tensor3d.as_vector().size()
        if meta is None:
            meta = event_tensor3d.meta()
            np_voxels = np.empty(shape=(num_point, 3), dtype=np.int32)
            larcv.fill_3d_voxels(event_tensor3d, np_voxels)
            output.append(np_voxels)
        else:
            assert meta == event_tensor3d.meta()
        np_values = np.empty(shape=(num_point, 1), dtype=np.float32)
        larcv.fill_3d_pcloud(event_tensor3d, np_values)
        output.append(np_values)
    return np.concatenate(output, axis=-1)


def parse_weights(data):
    """
    A function to generate weights from larcv::EventSparseTensor3D and larcv::Particle list

    For each voxel belonging to a particle :math:`p`, if the particle has :math:`N_p` voxels,
    the weight is computed as

    .. math::
        w_p = 1. / (N_p + 1)

    .. code-block:: yaml

        schema:
          weights:
            - parse_weights
            - sparse3d_pcluster
            - sparse3d_index
            - particle_pcluster

    Configuration
    -------------
    sparse3d_pcluster: larcv::EventSparseTensor3D
    sparse3d_index: larcv::EventSparseTensor3D
        Contains index information (to which particle each voxel belongs)
    particle_pcluster: larcv::Particle

    Returns
    -------
    np_voxels: np.ndarray
    np_values: np.ndarray
        Weight values for each voxel
    """
    event_tensor3d = data[0]
    num_point = event_tensor3d.as_vector().size()
    np_voxels = np.empty(shape=(num_point, 3), dtype=np.int32)
    larcv.fill_3d_voxels(event_tensor3d, np_voxels)

    event_index = data[1]
    assert num_point == event_index.as_vector().size()
    np_index = np.empty(shape=(num_point, 1), dtype=np.float32)
    larcv.fill_3d_pcloud(event_index, np_index)

    particles = data[2]
    num_voxels = np.array([1. / (p.num_voxels()+1) for p in particles.as_vector()])

    return np_voxels, num_voxels[np_index.astype(int)]


def parse_sparse3d_clean(data):
    """
    A function to retrieve clusters tensor.  Do the following cleaning:

    1) lexicographically sort coordinates

    2) choose only one group per voxel (by lexicographic order)

    3) get labels from the image labels for each voxel in addition to groups

    .. code-block:: yaml

        schema:
          segment_label:
            - parse_sparse3d_clean
            - sparse3d_mcst_reco
            - sparse3d_mcst_reco_group
            - sparsed_fivetypes_reco

    Configuration
    -------------
    sparse3d_mcst_reco: larcv::EventSparseTensor3D
    sparse3d_mcst_reco_group: larcv::EventSparseTensor3D
    sparse3d_fivetypes_reco: larcv::EventSparseTensor3D

    Returns
    -------
    grp_voxels: np.ndarray
        a numpy array with the shape (N,3) where 3 represents (x,y,z)
        coordinate
    grp_data: np.ndarray
        a numpy array with the shape (N,3) where 3 is energy + cluster id + label

    See Also
    --------
    parse_sparse3d_scn
    """
    img_voxels, img_data = parse_sparse3d_scn([data[0]])
    perm = np.lexsort(img_voxels.T)
    img_voxels = img_voxels[perm]
    #img_data = img_data[perm]
    img_voxels, unique_indices = np.unique(img_voxels, axis=0, return_index=True)
    #img_data = img_data[unique_indices]

    grp_voxels, grp_data = parse_sparse3d_scn([data[1]])
    perm = np.lexsort(grp_voxels.T)
    grp_voxels = grp_voxels[perm]
    grp_data = grp_data[perm]
    grp_voxels, unique_indices = np.unique(grp_voxels, axis=0, return_index=True)
    grp_data = grp_data[unique_indices]

    label_voxels, label_data = parse_sparse3d_scn([data[2]])
    perm = np.lexsort(label_voxels.T)
    label_voxels = label_voxels[perm]
    label_data = label_data[perm]
    label_voxels, unique_indices = np.unique(label_voxels, axis=0, return_index=True)
    label_data = label_data[unique_indices]

    sel2 = filter_nonimg_voxels(grp_voxels, label_voxels[(label_data<5).reshape((-1,)),:], usebatch=False)
    inds2 = np.where(sel2)[0]
    grp_voxels = grp_voxels[inds2]
    grp_data = grp_data[inds2]

    sel2 = filter_nonimg_voxels(img_voxels, label_voxels[(label_data<5).reshape((-1,)),:], usebatch=False)
    inds2 = np.where(sel2)[0]
    img_voxels = img_voxels[inds2]
    img_data = img_data[inds2]
    return grp_voxels, np.concatenate([img_data, grp_data, label_data[label_data<5][:, None]], axis=1)


def parse_sparse3d_scn_scales(data):
    """
    Retrieves sparse tensors at different spatial sizes.

    .. code-block:: yaml

        schema:
          segment_label:
            - parse_sparse3d_scn_scales
            - sparse3d_pcluster

    Configuration
    -------------
    sparse3d_pcluster: larcv::EventSparseTensor3D
        Can be repeated to load more features (one per feature).

    Returns
    -------
    list of tuples
    """
    grp_voxels, grp_data = parse_sparse3d_scn(data)
    perm = np.lexsort(grp_voxels.T)
    grp_voxels = grp_voxels[perm]
    grp_data = grp_data[perm]

    spatial_size = data[0].meta().num_voxel_x()
    max_depth = int(np.floor(np.log2(spatial_size))-1)
    scales = []
    for d in range(max_depth):
        scale_voxels = np.floor(grp_voxels/2**d)#.astype(int)
        scale_voxels, unique_indices = np.unique(scale_voxels, axis=0, return_index=True)
        scale_data = grp_data[unique_indices]
        # perm = np.lexsort(scale_voxels.T)
        # scale_voxels = scale_voxels[perm]
        # scale_data = scale_data[perm]
        scales.append((scale_voxels, scale_data))
    return scales
